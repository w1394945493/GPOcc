import numpy as np
import torch
from torch.utils import data
from . import OPENOCC_DATAWRAPPER
from gpocc.dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D

from .vggt_load_fn import load_and_preprocess_images


img_norm_cfg = dict(
    mean=[t * 255 for t in [0.485, 0.456, 0.406]],
    std=[t * 255 for t in [0.229, 0.224, 0.225]],
    to_rgb=True)


@OPENOCC_DATAWRAPPER.register_module()
class Scannet_Scene_Occ_DatasetWrapper(data.Dataset):
    """旧 GaussianFormer 风格 wrapper：输出 channel-first 图像。

    基础 Dataset 已完成磁盘读取和初始 resize；这里继续做几何/颜色增强、
    ImageNet 归一化、padding，并把图像从 HWC 转为 CHW。该输出格式不适合
    当前期望 channel-last 输入的 VGGTGaussianSegmentor。
    """
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], phase='train', size_divisor=32):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                ImageAug3D(
                    final_dim=final_dim,
                    resize_lim=resize_lim,
                    is_train=True
                ),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=size_divisor)
            ]
        else:
            transforms = [
                ImageAug3D(
                    final_dim=final_dim,
                    resize_lim=resize_lim,
                    is_train=False
                ),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=size_divisor)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 基础 Dataset 返回 imgs: [F,N,H,W,C]，数值约为 0~255。
        data = self.dataset[index]
        imgs, metas, occ = data

        # 将帧维 F 和相机维 N 合并，使同一套 transform 逐张处理图像。
        F, N, H, W, C = imgs.shape
        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
        # train：几何增强 + 颜色增强 + ImageNet normalize + pad；
        # test：确定性几何变换 + ImageNet normalize + pad。
        for t in self.transforms:
            imgs_dict = t(imgs_dict)

        # 普通 wrapper 最终输出 channel-first：[F,N,C,H,W]。
        imgs = imgs_dict['img']
        imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)

        FN, C, H, W = imgs.shape
        imgs = imgs.reshape(F, N, C, H, W)
        metas['img_shape'] = imgs_dict['img_shape']
        if imgs_dict.get('img_aug_matrix'):
            img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0)
            metas['img_aug_matrix'] = img_aug_matrix.reshape(F, N, 4, 4)

        for k in ['cam2world', 'vox_origin', 'occ_xyz', 'cam_vox_range', 'world2cam', 'scene_size', 'cam_k', 'fov_mask', 'depth_gt']:
            value = metas[k]

            if isinstance(value, (tuple, list)):
                value = np.array(value)

            value = torch.from_numpy(value.astype(np.float32)) # .cuda()
            metas[k] = value

        data_tuple = (imgs, metas, occ)
        return data_tuple



@OPENOCC_DATAWRAPPER.register_module()
class Scannet_Scene_Occ_DatasetWrapper_VGGT(data.Dataset):
    """VGGTGaussianSegmentor 共用 wrapper，DPT 与 VGGT 配置都会使用。

    名字虽为 VGGT，但其关键职责是保留 channel-last 布局。基础 Dataset
    已把图像长边缩放到 518、短边对齐到 14 的倍数；本 wrapper 当前不再做
    resize、ImageNet normalize 或 padding，只在训练时做颜色增强，然后除以
    255 并转成 Tensor。
    """
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], phase='train'):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                # ImageAug3D(
                #     final_dim=final_dim,
                #     resize_lim=resize_lim,
                #     is_train=True
                # ),
                PhotoMetricDistortionMultiViewImage(
                    brightness_delta=8,
                    contrast_range=(0.8, 1.2),
                    saturation_range=(0.8, 1.2),
                    hue_delta=4,
                ),
                # NormalizeMultiviewImage(**img_norm_cfg),
                # PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                # ImageAug3D(
                #     final_dim=final_dim,
                #     resize_lim=resize_lim,
                #     is_train=False
                # ),
                # NormalizeMultiviewImage(**img_norm_cfg),
                # PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 输入来自基础 Dataset：imgs=[F,N,H,W,C]，float32，范围约 0~255。
        data = self.dataset[index]
        imgs, metas, occ = data

        # 第 1 步：暂时合并 F/N，方便按图像执行 transform。
        # imgs = load_and_preprocess_images(imgs)
        F, N, H, W, C = imgs.shape

        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}

        # 第 2 步：训练阶段当前只做轻量颜色扰动；测试阶段 transforms 为空。
        # 下面构造函数中的 ImageAug3D、Normalize、Pad 均已注释，因此配置里的
        # final_dim/resize_lim 在这个 _VGGT wrapper 中目前不会改变主图像。
        for t in self.transforms:
            imgs_dict = t(imgs_dict)

        # 第 3 步：恢复 [F,N,H,W,C]，保持 channel-last；再由 0~255 缩放到
        # [0,1] 并转成 Tensor。这里没有执行 ImageNet mean/std 归一化。
        imgs = np.stack(imgs_dict['img']).reshape(F, N, H, W, C)
        imgs = torch.from_numpy(imgs) / 255.

        cam_intrin = metas['cam_k']

        # FN, C, H, W = imgs.shape
        # imgs = imgs.reshape(F, N, C, H, W)
        # metas['img_shape'] = imgs_dict['img_shape']

        # if imgs_dict.get('img_aug_matrix'):
        #     img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0)
        #     metas['img_aug_matrix'] = img_aug_matrix.reshape(F, N, 4, 4)
        # 第 4 步：把模型 forward 会使用的几何 metadata 从 NumPy 转成 Tensor。
        # 最终 DataLoader 再增加 batch 维，主图像成为 [B,F,N,H,W,C]；模型内
        # 部 squeeze N 并 rearrange 为 [B,F,C,H,W] 后送入 DPT 或 VGGT backbone。
        for k in ['cam2world', 'vox_origin', 'occ_xyz', 'cam_vox_range', 'world2cam', 'scene_size', 'cam_k', 'fov_mask', 'depth_gt']:
            value = metas[k]

            if isinstance(value, (tuple, list)):
                value = np.array(value)

            value = torch.from_numpy(value.astype(np.float32)) # .cuda()
            metas[k] = value

        data_tuple = (imgs, metas, occ)
        return data_tuple
