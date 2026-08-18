# 遗留训练配置：当前 train_embodied.py 只实现评估，--evaluate 模式不会构建优化器。
optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)}
    ),
)
# 遗留训练参数：当前 train_embodied.py 只支持 --evaluate，非评估分支会
# assert False，因此不会执行梯度裁剪。
grad_max_norm = 35
amp = False
seed = 1
print_freq = 1
# 遗留训练参数：脚本会读取它们，但 embodied 评估流程不使用 epoch 循环。
eval_freq = 10
max_epochs = 10
load_from = "/c20250502/wangyushen/Outputs/gpocc/dav2_bin16/train/latest.pth"
load_from = None
find_unused_parameters = True
track_running_stats = True
flag_depthanything_as_gt = False

ignore_label = 0
empty_idx = 12   # 0 ignore, 1~11 objects, 12 empty
cls_dims = 13

# ===== 以下主要是旧 GaussianFormer anchor/refine/spconv 配置的遗留变量 =====
# 当前 VGGTGaussianSegmentorOnline 不构建下面的 anchor_encoder、refine_layer
# 和 spconv_layer，因此这些变量不会控制 OccScanNet 的实际感知范围或网络结构。
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # 遗留，无实际作用；不是 OccScanNet 场景范围
scale_range = [0.01, 0.08]  # 此顶层变量未传入 model；模型当前使用构造函数中的同值默认参数
image_size = [480, 640]  # 遗留，无引用；实际尺寸由 wrapper 的 final_dim 单独指定
resize_lim = [1.0, 1.0] 
num_frames = 1  # 遗留接线：dataset 只保存 self.num_frames，后续没有读取，实际使用场景全部有效帧
offset = 0  # 遗留，无引用；dataset 配置中的 offset 已被注释
grad_frames = None  # 遗留接口：虽传给 Online forward，但当前 VGGT/DPT 单帧 forward 未使用

_dim_ = 96  # 遗留，仅被下方未构建的旧模块配置引用
num_cams = 1  # 遗留，无引用
num_heads = 3  # 遗留，无引用
num_levels = 4  # 遗留，无引用
num_anchor = 16200  # 遗留，无引用；当前 Gaussian 数量由 num_bins 和图像特征分辨率决定
num_anchor_init = 8100  # 遗留接线：仅写入 meta['num_depth']，当前 VGGT/DPT 模型不读取
num_cross_layer = 3  # 遗留，无引用
num_self_layer = 3  # 遗留，无引用
num_decoder_fillhead = 2  # 遗留，无引用
semantics_activation = 'identity'  # 遗留，仅被下方未构建的 refine_layer 引用
use_camera_embed = False  # 遗留，无引用

# 遗留模块：该字典不在 model=dict(...) 内，build_model(cfg.model) 不会构建它。
anchor_encoder = dict(
    type='SparseGaussian3DEncoder',
    embed_dims=_dim_,
    semantic_dim=cls_dims-1,
)

# 遗留模块：旧 GaussianFormer 用它迭代细化 anchor；当前 DPT 分支直接通过
# gs_pred_layer 从图像特征预测 Gaussian，不会构建或调用该模块。
refine_layer = dict(
    type='SparseGaussian3DDeltaRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[0.1, 0.1, 0.06], 
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims-1,
    semantics_activation=semantics_activation,
)

# 遗留模块：旧 GaussianFormer 的 anchor 间稀疏 3D 卷积；当前模型不会构建。
spconv_layer=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.08]*3, 
    kernel_size=3,
)

model = dict(
    type='VGGTGaussianSegmentorOnline',
    frozen_backbone=False,
    freeze_blocks=0,  # DPT 分支不使用；这是 VGGT aggregator 分块冻结的遗留参数
    flag_depthbranch=False,
    flag_depthanything_as_gt=flag_depthanything_as_gt,
    use_depthanything=True,
    render_feat=False,  # 遗留无效：构造函数没有消费该参数，经 **kwargs 传递后被丢弃
    detach_local=True,
    num_bins=16,

    semantic_dim=cls_dims-1,

    merge_kernel_dist_thresh=0.02,
    opacities_threshold=0.01,
    opa_to_thickness=False,  # 遗留实验开关：仅保存为属性，当前融合代码未读取
    use_markley_avg_quat=False,  # 遗留实验开关：当前四元数融合固定使用符号对齐后加权平均
    use_log_avg=False,  # 遗留实验开关：仅保存为属性，当前融合代码未读取
    weight_use_p=False,
    weight_use_t=True,
    # 当前接线未生效：构造函数没有保存 temporal_alpha；融合代码通过
    # getattr(self, 'temporal_alpha', 0.1) 固定回退到 0.1。修改此值不会改变行为。
    temporal_alpha=0.1,

    cuda_kwargs=dict(
        scale_multiplier=3,
        H=60, W=60, D=36,
        # 遗留占位：LocalAggregator 会注册该值，但 forward 实际使用每帧
        # meta['vox_origin'] 作为局部网格原点。
        pc_min=[-51.2, -51.2, -5.0],
        grid_size=0.08),
    global_cuda_kwargs=dict(
        scale_multiplier=3,
        H=200, W=220, D=90,
        # 遗留占位：全局聚合实际使用每个场景的 global_scene_origin。
        pc_min=[-51.2, -51.2, -5.0],
        grid_size=0.08),
)


# 遗留训练配置：当前 train_embodied.py 的 --evaluate 路径不会构建 loss；
# 保留它仅用于将来补回 embodied 训练流程。
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='FocalLoss',
            weight=100.0, 
            gamma=2.0,
            alpha=0.25,
            cls_freq=[5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393, 30949122, 9871618, 3196722886],
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'target': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            ignore_label=ignore_label,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Sem_Scal_Loss',
            weight=1.0,
            ignore_label=ignore_label,
            sem_cls_range=[1, 12],
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Geo_Scal_Loss',
            weight=1.0,
            empty_idx=empty_idx,
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label',
                'fov_mask': 'fov_mask'}),
    ]
)

# data_path = './data/occscannet' # path/to/your/data/occscannet
monoocc_root = "/c20250502/wangyushen/Datasets/occscannet"    # occscannet路径  
occscannet_root = "/c20250502/wangyushen/Datasets/scene_occ"  # embodiedocc路径

train_dataset_config = dict(
    type="Scannet_Online_SceneOcc_Dataset",
    monoocc_root = monoocc_root,
    occscannet_root=occscannet_root,
    num_frames=num_frames,
    empty_idx=empty_idx,
    phase="train",
    num_pts=num_anchor_init,
    data_tag="base",  # 'mini' for mini-set
    vggt_image_preprocess=True,
)

val_dataset_config = dict(
    type="Scannet_Online_SceneOcc_Dataset",
    monoocc_root=monoocc_root,
    occscannet_root=occscannet_root,  #! 增加数据集根目录
    num_frames=num_frames,
    # offset = offset,
    empty_idx=empty_idx,
    phase="test",
    num_pts=num_anchor_init,
    data_tag="base",  # 'mini' for mini-set
    vggt_image_preprocess=True,
)

train_wrapper_config = dict(
    type='Scannet_Online_SceneOcc_DatasetWrapper_VGGT',
    final_dim = [480, 640], 
    resize_lim = resize_lim,
    phase='train', 
)

val_wrapper_config = dict(
    type='Scannet_Online_SceneOcc_DatasetWrapper_VGGT',
    final_dim = [480, 640],
    resize_lim = resize_lim,
    phase='test', 
    # phase='train', # for vis
)

train_loader_config = dict(
    batch_size = 2,
    shuffle = True,
    num_workers = 5,
)

val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
)
