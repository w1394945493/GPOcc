#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opas,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, bin_logits, density, probability, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo

        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer,
            binningBuffer,
            imgBuffer,
            means3D,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            probability
        )
        return logits, bin_logits, density

    @staticmethod # todo
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opas, semantics, logits, bin_logits, density, probability = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            probability,
            logits_grad,
            bin_logits_grad,
            density_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opas_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opas_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1):
        """构造 Gaussian-to-voxel 聚合器。

        参数来源分为两类：

        1. ``H/W/D``、``grid_size`` 和 ``scale_multiplier`` 当前仍真实参与
           体素索引、边界检查及 Gaussian 影响半径计算；
        2. ``pc_min`` 和 ``radii_min`` 是旧 GaussianFormer 接口遗留参数。
           ``pc_min`` 仅被注册为 buffer，forward 不读取它；``radii_min``
           对应的 clamp 已被注释，当前同样不影响计算。

        当前代码在 forward 时通过 ``origin_use`` 接收真实网格原点：局部 OCC
        传入每帧的 ``vox_origin``，全局 OCC 传入场景的
        ``global_scene_origin``。
        """
        super().__init__()
        # 以下参数仍有效：
        # - scale_multiplier：把 Gaussian 最大 scale 换算成 CUDA splat 半径；
        # - H/W/D：体素网格容量和 CUDA kernel 的空间尺寸；
        # - grid_size：世界坐标到体素坐标的离散化步长。
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D

        # GaussianFormer 遗留占位参数：配置通常仍传入 nuScenes 风格的
        # [-51.2, -51.2, -5.0]，这里只为兼容旧构造接口而注册 buffer。
        # 当前 forward 没有使用 self.pc_min，实际原点由 origin_use 覆盖。
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size

        # GaussianFormer 遗留参数：原本用于 radii.clamp(min=radii_min)，但
        # 对应代码目前已注释，所以改变该配置不会影响实际 splat 半径。
        self.radii_min = radii_min

    def forward(
        self,
        pts,
        means3D,
        opas,
        semantics,
        scales,
        cov3D,
        metas,
        origin_use):
        """把一组连续空间 Gaussian 聚合到给定体素采样点。

        ``metas`` 是为兼容旧调用签名保留的形式参数，当前函数体没有读取；
        ``origin_use`` 才是本次聚合实际采用的动态网格原点。
        """

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opas = opas.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        # 不使用构造阶段保存的 self.pc_min。调用方在运行时传入真实原点：
        # - 局部聚合 self.aggregator：origin_use = meta['vox_origin']；
        # - 全局聚合 self.global_aggregator：
        #   origin_use = meta['global_scene_origin']。
        nyu_pc_min = origin_use.cuda()

        # 使用动态原点和真实有效的 grid_size，把采样点及 Gaussian 中心
        # 从世界坐标离散化为当前 OCC 网格的整数索引。
        points_int = ((pts - nyu_pc_min) / self.grid_size).to(torch.int)

        assert points_int.min() >= 0 and points_int[..., 0].max() < self.H and points_int[..., 1].max() < self.W and points_int[..., 2].max() < self.D

        means3D_int = ((means3D.detach() - nyu_pc_min) / self.grid_size).to(torch.int)

        # assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        # radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)

        # GaussianFormer 旧逻辑，现已停用；因此 self.radii_min 只是占位参数。
        # radii = radii.clamp(min=self.radii_min)

        # assert radii.min() >= 1
        # cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        assert means3D_int.min() >= 0
        assert means3D_int[:, 0].max() < self.H
        assert means3D_int[:, 1].max() < self.W
        assert means3D_int[:, 2].max() < self.D
        # 当前实际半径只由 Gaussian scale、scale_multiplier 和 grid_size 决定。
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        if opas.shape[-1] == 1:
            opas = opas.squeeze(-1)

        # Invoke C++/CUDA rasterization routine
        logits, bin_logits, density = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        return logits, bin_logits, density # n, c; n, c; n
