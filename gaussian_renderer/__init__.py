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


# ✅ means3D 和 means2D 的含义与区别
# 名称	表示含义	数据空间	维度	用途
# means3D	每个高斯的三维位置（中心）	世界坐标系	Nx3	真实的 3D 高斯位置
# means2D	每个高斯在图像上的投影位置（中心）	屏幕/像素坐标	Nx2 或 Nx3	渲染时用于 rasterization 的位置
# means3D (世界坐标)
#    ↓ 通过 view/proj 矩阵
# means2D (屏幕像素中心)
#    ↓ 作为光栅化起点
# Rasterization & 渲染输出
# means2D 实际上是把 means3D 投影（Projection）到相机视图后的坐标，但这一步在 GaussianRasterizer 内部是必须显示提供的，因为它可以被训练用于反向传播优化。



import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 🧠 含义：
    #     创建一个和 pc.get_xyz（高斯3D位置）同形状、同数据类型的张量，初始化为 0。
    #     requires_grad=True 表示该张量会参与 梯度计算。
    # 📦 _xyz 的结构：
    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     类型：[N, 3] 的张量
    #     含义：包含 N 个高斯点，每个点的 (x, y, z) 坐标
    # 🚛“我用 means2D 做一个假的中转仓库，只为了帮我把梯度寄回来。真正的地址是 means3D，但不能直接寄，所以我中转一下。”
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 如果不是特例的 "PanopticSports" 数据集，就使用标准的相机设置（即 MiniCam 类型）。
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        
        # ✅ 将相机的水平视场角（FoVx）和垂直视场角（FoVy）转换为其一半的正切值：
        # 🧠 背景知识：视场角和透视投影
        #     在透视投影中，视场角（FoV, Field of View）定义了相机观察的“张角”。越大，视野越广。
        #     FoVx：水平视场角（弧度）
        #     FoVy：垂直视场角（弧度）
        # 🎯 为什么需要 tanfovx 和 tanfovy？
        #     在 GaussianRasterizer 中，这两个值被用来：
        #     将高斯点从 3D 空间投影到屏幕坐标（screen space）；
        #     决定点在图像中的屏幕大小（受视角影响）；
        #     保证渲染时不同分辨率或视角下投影尺寸一致。
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # 设置 Gaussian 光栅化器的关键参数：
        #     image_height/width	渲染图像的分辨率
        #     tanfovx/y	水平/垂直视场角的缩放因子
        #     bg	背景色（如白色或黑色）
        #     scale_modifier	缩放调节器（控制屏幕空间中的点大小）
        #     viewmatrix	世界→相机的变换矩阵
        #     projmatrix	投影矩阵（相机→屏幕）
        #     sh_degree	当前球谐函数的阶数
        #     campos	相机位置
        #     prefiltered	是否启用预滤波（关）
        #     debug	是否调试模式
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        
    # 🧠 GaussianRasterizer 是干什么的？
        # 它是整个 3D高斯 Splatting 渲染核心，作用如下：
        # ①	将每个高斯基元从 3D 投影到屏幕空间（使用投影矩阵）
        # ②	按高斯协方差计算屏幕上的半径大小
        # ③	对高斯进行 rasterization（光栅化），融合其颜色、不透明度、深度
        # ④	合成最终图像，支持梯度传播
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    # 这三行代码是准备高斯体渲染（Gaussian Splatting）所需的关键输入，它们分别指定了屏幕空间位置、透明度和颜色特征。
    # ✅ means2D = screenspace_points
    #     作用：设置每个高斯在屏幕空间（即图像平面）上的位置。
    #     含义：screenspace_points 是一个与 pc.get_xyz 同形状的张量，初始为 0，但由于其启用了 requires_grad=True，它可以用于后续计算 视平面梯度。
    #     这个变量通常在光栅化中用于记录 2D 投影坐标的位置，并用于 梯度回传（用于训练）。
    #     背景：虽然这里是 0，但实际渲染中 rasterizer 会内部更新为每个高斯真实的屏幕坐标。
    # ✅ opacity = pc._opacity  作用：获取每个高斯当前的 原始不透明度参数，尚未经过 sigmoid 激活。
    # ✅ shs = pc.get_features  作用：提取每个高斯的 球谐系数（Spherical Harmonics） 颜色特征。
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        
        # 用 Python 在前处理阶段计算高斯的协方差矩阵
        # 返回一个 (N, 3, 3) 的张量，表示每个高斯的协方差矩阵；
            # 它是通过：
            # 先将 _scaling 应用 exp（确保正数）；
            # 然后与 _rotation 构建一个变换矩阵；
            # 最后通过 𝐿⋅𝐿𝑇L⋅L T计算协方差；
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    # ✅ 它是一个形如 [True, False, True, True, ...] 的布尔向量，表示哪些高斯启用了变形（deformation）模块：
    #     长度与高斯数相同；
    #     值为 True 的点才会被送入变形网络；
    #     后续用于筛选 deformation 输入。
    deformation_point = pc._deformation_table

    # coarse 阶段：
    # 表示训练的早期阶段，重点在于结构初始化，不考虑时间变化。
    # 所以直接使用原始（静态）参数：
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs

    
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)

    # 变正值    
    # self.scaling_activation = torch.exp
    # 将缩放尺度 scales_final（通常为网络输出的原始值）变换为正值，用于构造高斯协方差矩阵。
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)


    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None

    # 初始化变量，准备后续赋值。这个变量最终会存储每个高斯的 RGB 颜色值。
    colors_precomp = None

    # 判断是否使用外部强制设置的颜色（override_color），如果没有，就用球谐系数 SH 计算颜色。
    # 实现了高斯颜色的确定
    if override_color is None:
        if pipe.convert_SHs_python:
            
            # 将球谐系数 reshape 成 (num_points, 3, num_coeffs)。
            # 示例：如果有 10,000 个高斯、3 通道、16 个球谐系数 → shape 是 [10000, 3, 16]
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)

            # 计算每个高斯中心指向相机的向量（相机视线方向）。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))

            # 对该方向向量进行单位化，作为方向输入给 SH 函数。
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

            # 用方向和 SH 系数计算出 RGB 值：
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)

            # SH 求和结果可能在 [-0.5, +0.5] 的范围（默认以 0 为中值），所以加 0.5；
            # 再做一个下限裁剪，防止负数，确保 RGB ≥ 0。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()

    # 调用高斯渲染器（rasterizer）生成最终图像 的核心步骤
    # 输出：
    #     rendered_image：最终 RGB 图像；
    #     radii：每个高斯点在屏幕上的投影半径（判断显著性）；
    #     depth：深度图（Z 值），用于遮挡或可视化。
    # 参数	说明
        # means3D_final	形变后的高斯中心坐标（[N, 3]）
        # means2D	高斯中心在屏幕上的投影点坐标（[N, 2]）
        # shs 或 colors_precomp	颜色信息（球谐系数 或 预计算 RGB）
        # opacities	每个高斯的透明度 α（[N, 1]）
        # scales	各向异性的高斯缩放系数（[N, 3]）
        # rotations	高斯朝向的四元数旋转（[N, 4]）
        # cov3D_precomp	可选的协方差矩阵，提前计算好提高效率
            # for each gaussian i:
            #     1. 将 mean3D[i] 投影成 mean2D[i]
            #     2. 用 scale + rotation 构造 2D 高斯协方差（shape on screen）
            #     3. 按照 mean2D[i]、cov[i]，渲染出它在图像上的影响区域（高斯罩）
            #     4. 用颜色 × 不透明度（shs 或 RGB），合成 RGB 图像（累加、前向透明度合成）
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,  # 真正影响渲染的高斯位置，来自 pc._xyz
        means2D = means2D,   # 只是为了保留梯度信息的 dummy tensor
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}

