import os, time, argparse, os.path as osp, numpy as np
import torch
import gc
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm


from gpocc.utils.iou_eval import IOUEvalBatch
from gpocc.utils.iou_as_iso import SSCMetrics, SSCMetricsTorch
from gpocc.utils.loss_record import LossRecord
from gpocc.utils.load_save_util import revise_ckpt, revise_ckpt_2
from gpocc.loss.depth_loss import DepthLoss

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import open3d as o3d
import cv2
from einops import einsum
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")
from PIL import Image


def pass_print(*args, **kwargs):
    pass


def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def save_model(
    model, optimizer=None, scheduler=None, epoch=None, save_epoch=False, **kwargs
):
    """保存可恢复训练的 checkpoint。

    每轮都会覆盖 latest.pth；达到 save_freq 时额外保留 epoch_N.pth。
    除模型参数外，还保存 optimizer、scheduler、epoch、global_iter 和最佳指标，
    以便中断后继续训练，而不只是做纯模型权重加载。
    """
    dict_to_save = {"state_dict": model.state_dict()}

    if optimizer is not None:
        dict_to_save["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        dict_to_save["scheduler"] = scheduler.state_dict()

    if len(kwargs):
        dict_to_save.update(kwargs)

    if epoch is not None:
        dict_to_save["epoch"] = epoch

    # 1. latest.pth 每次都保存
    latest_file = os.path.join(os.path.abspath(args.work_dir), "latest.pth")
    torch.save(dict_to_save, latest_file)

    # 2. epoch_x.pth 按需保存
    if save_epoch and epoch is not None:
        epoch_file = os.path.join(os.path.abspath(args.work_dir), f"epoch_{epoch}.pth")
        torch.save(dict_to_save, epoch_file)


def main(args):
    # ====================================================================== #
    # 1. 读取实验配置并初始化全局运行参数
    # ====================================================================== #
    # benchmark=True 会为固定输入尺寸选择更快的 cuDNN kernel；如果输入尺寸
    # 频繁变化，首次搜索 kernel 的额外成本可能抵消收益。
    torch.backends.cudnn.benchmark = True

    # py-config 同时定义模型、数据集、损失、优化器以及训练轮数等全部实验参数。
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed) # seed:1
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs # 10
    eval_freq = cfg.eval_freq

    save_freq = cfg.get("save_freq", 1) # 每隔几个epoch保存一次model

    print_freq = cfg.print_freq
    vis_freq = cfg.get('vis_freq', 10)
    if args.vis_freq:
        vis_freq = args.vis_freq

    # ====================================================================== #
    # 2. 初始化单卡或 torchrun DDP 环境
    # ====================================================================== #
    # torchrun 会注入 RANK/WORLD_SIZE/LOCAL_RANK；普通 python 启动时则走单卡。
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        # init DDP
        # distributed = True
        world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
        rank = int(os.environ["RANK"])  # node id

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(
            backend="nccl",
            # init_method=f"env://",
            world_size=world_size,
        )
        rank, world_size = get_dist_info()
    else:
        rank = 0
        world_size = 1
        # torch.cuda.set_device(0)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # ====================================================================== #
    # 3. 创建工作目录和日志
    # ====================================================================== #
    # 只由 rank 0 保存一份最终配置，便于复现实验。
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    exp_name = os.path.basename(args.work_dir)
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(
        name='embodied',
        logger_name=exp_name,
        log_file=log_file,
        log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # ====================================================================== #
    # 4. 根据 cfg.model 构建 Mono Gaussian OCC 模型
    # ====================================================================== #
    # DPT 和 VGGT 配置当前都构建 VGGTGaussianSegmentor，主要区别是
    # use_depthanything 是否选择 DepthAnything/DINOv2 作为图像 backbone。
    from gpocc.model import build_model
    my_model = build_model(cfg.model).cuda()

    if cfg.flag_depthanything_as_gt:
        # 仅旧独立深度辅助分支使用；当前 release custom 配置通常为 False。
        my_model.depthanything.requires_grad_(False)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}') # 可学习参数 944,309,688

    if distributed:
        # DDP 为每个进程复制一份模型并在 backward 时同步参数梯度。
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model,
            # device_ids=[gpu],
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    logger.info('done ddp model')

    # ====================================================================== #
    # 5. 构建基础 Dataset、外层 Wrapper 和 DataLoader
    # ====================================================================== #
    # build_dataloader 内部先用 *_dataset_config 读取样本，再用
    # *_wrapper_config 做图像增强/布局转换，最后按 *_loader_config 组成 batch。
    from gpocc.dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    # ====================================================================== #
    # 6. 构建优化器、混合精度、OCC 损失和学习率调度器
    # ====================================================================== #
    amp = cfg.get('amp', True)
    # MMEngine OptimWrapper 根据配置构建 AdamW，并支持 backbone 的 lr_mult。
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    # amp=False 时 GradScaler 自动退化为普通 FP32 更新。
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    from gpocc.loss import GPD_LOSS
    # 配置中的 MultiLoss：Focal + Lovasz + Semantic Scaling + Geometric Scaling。
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    # 以 iteration 而非 epoch 更新 cosine LR；warmup 固定为 1000 iterations。
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=1000, # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False
    )

    # OCC 验证指标：统计几何 IoU、各语义类 IoU 和语义 mIoU。
    CalMeanIou = SSCMetricsTorch(n_classes=12)

    # ====================================================================== #
    # 7. 确定断点续训/预训练权重来源
    # ====================================================================== #
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info(f'resume from: {cfg.resume_from}')
    logger.info(f'work dir: {args.work_dir}')

    if cfg.resume_from and osp.exists(cfg.resume_from):
        # resume：恢复模型和完整训练状态。evaluate 时只需要恢复模型及计数信息。
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        if not args.evaluate:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        # load_from：只加载模型权重，常用于从发布 checkpoint 开始评估/微调。
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))

    # 不在该列表里的 metadata 会在每个 iteration 中转成 Tensor 并移到 GPU。
    # img_depthbranch 被脚本另行显式搬到 GPU，但当前 VGGT/DPT forward 不读取它。
    metas_tensor_keys_inv = ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix']

    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    if args.evaluate:
        # ================================================================== #
        # 8A. 单独评估模式：只跑 val loader，不计算 loss、不更新参数
        # ================================================================== #
        my_model.eval()
        CalMeanIou.reset()
        # loss_record = LossRecord(loss_func=loss_func)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        num_gaussians = []
        with torch.no_grad():
            for i_iter_val, data in enumerate(tqdm(val_dataset_loader)):

                # custom_collate_fn 返回可修改 list；先将 batch 顶层 Tensor 搬到 GPU。
                for i in range(len(data)):
                    if isinstance(data[i], torch.Tensor):
                        data[i] = data[i].cuda()
                (imgs, metas, label) = data

                device = imgs.device
                # 把模型 forward 所需的数值 metadata 统一转换成当前 GPU Tensor。
                for meta in metas:
                    for k, v in meta.items():
                        if k not in metas_tensor_keys_inv:
                            if isinstance(v, torch.Tensor):
                                meta[k] = v.to(device)
                            else:
                                meta[k] = torch.as_tensor(v, device=device)
                    meta["img_depthbranch"] = meta["img_depthbranch"].to(device)

                with torch.cuda.amp.autocast(enabled=amp):
                    # 单帧推理输出：result_dict 是局部 OCC；其余返回值在评估中不用。
                    result_dict, my_occ, predtoreturn = my_model(
                        imgs=imgs,
                        metas=metas,
                        points=None,
                        label=label,
                        grad_frames=None,
                        test_mode=True
                    )

                # ce_input=[B,C,60,60,36]，在类别维 argmax 得到体素类别。
                voxel_predict = result_dict['ce_input'].argmax(dim=1).long()
                voxel_label = result_dict['ce_label'].long()

                # 模型标签定义：0=unknown/ignore、12=empty；SSCMetrics 定义：
                # 255=ignore、0=empty，因此评估前做类别重映射。
                voxel_predict[voxel_predict == 0] = 255
                voxel_predict[voxel_predict == 12] = 0
                voxel_label[voxel_label == 0] = 255
                voxel_label[voxel_label == 12] = 0

                if args.eval_fov:
                    # 可选：视锥外体素置为 ignore，只评估当前相机实际可见范围。
                    fov_mask = metas[0]['fov_mask']
                    voxel_label[0][fov_mask == 0] = 255
                    voxel_predict[0][fov_mask == 0] = 255

                CalMeanIou.add_batch(voxel_predict, voxel_label)

                if i_iter_val % print_freq == 0:
                    stats = CalMeanIou.get_stats(distributed=False)
                    info_sem_cls = stats["iou_ssc"]
                    info_sem = stats["iou_ssc_mean"]
                    info_geo = stats["iou"]

                    if is_main_process():
                        logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                        logger.info(f'Current val iou of sem is {info_sem}')
                        logger.info(f'Current val iou of geo is {info_geo}')

                gc.collect()
                torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats(distributed=distributed)

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            if is_main_process():
                logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                logger.info(f'Current val iou of sem is {info_sem}')
                logger.info(f'Current val iou of geo is {info_geo}')

        return

    # ====================================================================== #
    # 8B. 训练主循环
    # ====================================================================== #
    while epoch < max_num_epochs:

        my_model.train()
        # DDP sampler 每个 epoch 使用不同随机种子，确保各 rank 数据划分正确洗牌。
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = LossRecord(loss_func=loss_func)
        time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            # -------------------- 8B-1. 数据搬运与 metadata 整理 --------------------
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data

            device = imgs.device
            for meta in metas:
                for k, v in meta.items():
                    if k not in metas_tensor_keys_inv:
                        if isinstance(v, torch.Tensor):
                            meta[k] = v.to(device)
                        else:
                            meta[k] = torch.as_tensor(v, device=device)
                meta['img_depthbranch'] = meta['img_depthbranch'].to(device)

            # -------------------- 8B-2. 模型前向 --------------------
            data_time_e = time.time()

            with torch.cuda.amp.autocast(enabled=amp):
                result_dict, sparse_result_dict, predtoreturn = my_model(
                    imgs=imgs, metas=metas,
                    points=None, label=label,
                    grad_frames=cfg.grad_frames,
                    test_mode=False)

            # -------------------- 8B-3. 组合训练损失 --------------------
            # DPT/VGGT 分支在配置 MultiLoss 外额外加入深度 Huber Loss。
            # 对 mono_dpt_bin16_release_custom.py，总损失为：
            # 0.2*Depth + 100*Focal + Lovasz + SemScal + GeoScal。
            assert cfg.ignore_label == 0
            if 'vggt' in args.py_config.lower() or 'dpt' in args.py_config.lower():
                total_loss = 0.
                all_loss_dict = {}
                # 深度预测损失
                # 深度预测由 anchor_depth_pred 产生；GT 使用 meta['depth_gt']。
                depth_loss = DepthLoss(
                    input_dict=dict(
                        depth_preds='depth_pred',
                        depth_labels='depth_gt',
                    ),
                    weight=cfg.get('depth_loss_weight', 2.0),
                )(result_dict)

                total_loss = total_loss + depth_loss
                all_loss_dict['depth_loss'] = depth_loss.detach().item()

                if 'ce_label' in result_dict:
                    # 对局部 60x60x36 OCC 计算配置中的四项 MultiLoss。
                    loss, loss_dict = loss_func(result_dict)
                    all_loss_dict.update(loss_dict)
                    total_loss = total_loss + loss

                if sparse_result_dict is not None:
                    # 预留的稀疏 Gaussian/OCC 辅助监督。当前模型通常返回 None，
                    # 因而 release custom 配置实际不会产生 sparse loss。
                    if isinstance(sparse_result_dict, dict) and 'ce_label' in sparse_result_dict:
                        sparse_loss, sparse_loss_dict = loss_func(sparse_result_dict)
                        for k, v in sparse_loss_dict.items():
                            all_loss_dict[f'sparse_{k}'] = v
                    else:
                        sparse_loss = sparse_result_dict
                    total_loss = total_loss + sparse_loss

                loss_record.update(loss=total_loss.item(), loss_dict=all_loss_dict)
            else:
                total_loss, loss_dict = loss_func(result_dict)
                loss_record.update(loss=total_loss.item(), loss_dict=loss_dict)

            # -------------------- 8B-4. 反向传播与参数更新 --------------------
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            # clip_grad_norm_ 前必须 unscale，确保裁剪的是实际梯度而非缩放后梯度。
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)

            valid_grad = True

            scaler.step(optimizer)
            scaler.update()
            # 每次 iteration 更新一次 cosine learning rate。
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and is_main_process():
                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                # logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                #             'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))
                logger.info(
                    "[TRAIN] Epoch %d Iter %5d/%d   "
                    % (epoch + 1, i_iter, len(train_dataset_loader))
                    + loss_info
                    + "GradNorm: %.3f,   lr: %.7f,   memory: %.2f GB,   time: %.3f (%.3f)"
                    % (
                        grad_norm,
                        lr,
                        torch.cuda.memory_allocated() / 1024**3,
                        time_e - time_s,
                        data_time_e - data_time_s,
                    )
                )
                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()

        # -------------------- 8B-5. 每个 epoch 保存 checkpoint --------------------
        if is_main_process():
            save_model(
                my_model,
                optimizer,
                scheduler,
                epoch + 1,
                save_epoch=((epoch + 1) % save_freq == 0),
                global_iter=global_iter,
                best_val_iou=best_val_iou,
                best_val_miou=best_val_miou,
            )

        epoch += 1
        # -------------------- 8B-6. 按 eval_freq 周期验证 --------------------
        # 验证只统计局部 OCC 指标，不计算 loss，也不更新 best checkpoint。
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou.reset()
            loss_record = LossRecord(loss_func=loss_func)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            with torch.no_grad():
                for i_iter_val, data in enumerate(val_dataset_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], torch.Tensor):
                            data[i] = data[i].cuda()
                    (imgs, metas, label) = data

                    device = imgs.device
                    for meta in metas:
                        for k, v in meta.items():
                            if k not in metas_tensor_keys_inv:
                                if isinstance(v, torch.Tensor):
                                    meta[k] = v.to(device)
                                else:
                                    meta[k] = torch.as_tensor(v, device=device)
                        meta['img_depthbranch'] = meta['img_depthbranch'].to(device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict, my_occ, predtoreturn = my_model(
                            imgs=imgs,
                            metas=metas,
                            points=None,
                            label=label,
                            grad_frames=None,
                            test_mode=True
                        )

                    # loss, loss_dict = loss_func(result_dict)
                    # loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                    voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                    voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]

                    voxel_predict[voxel_predict == 0] = 255
                    voxel_predict[voxel_predict == 12] = 0
                    voxel_label[voxel_label == 0] = 255
                    voxel_label[voxel_label == 12] = 0

                    CalMeanIou.add_batch(voxel_predict, voxel_label)

                    gc.collect()
                    torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats(distributed=distributed)

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
            logger.info(f'Current val iou of sem is {info_sem}')
            logger.info(f'Current val iou of geo is {info_geo}')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_mono')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=10)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--eval_fov', action='store_true')

    args, _ = parser.parse_known_args()
    main(args)
