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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    # 如果 dataset.white_background 为 True，表示该数据集采用白背景，则设置 bg_color = [1, 1, 1]，即 RGB 的白色；
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()

                             
    # 从当前 scene 中获取用于 视频渲染 的相机列表（video_cams），用于后续动态场景可视化或评估阶段的图像渲染。
    # 也就是说，它是一个封装了所有用于视频回放（或渲染评估）的相机序列的数据集对象，包含如下内容：
                             
    # 每一帧相机的：
    # 内参（FoV、分辨率等）
    # 外参（R, T）
    # 时间戳 t
    # 所对应图像路径
    # 投影矩阵、相机中心等派生信息
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 是否使用默认随机采样
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:

        viewpoint_stack = scene.getTrainCameras()
        # viewpoint_stack = [
        #     Camera(...),  # 第1个视角
        #     Camera(...),  # 第2个视角
        #     ...
        #     Camera(...),  # 第N个视角
        # ]
        
        # Camera(
        #     colmap_id=0,
        #     R=array([[1.0, 0.0, 0.0], 
        #              [0.0, 1.0, 0.0], 
        #              [0.0, 0.0, 1.0]]),              # 旋转矩阵
        #     T=array([0.0, 0.0, -5.0]),                # 平移向量
        #     FoVx=0.857,                               # 水平方向视场角（弧度）
        #     FoVy=0.642,                               # 垂直方向视场角（弧度）
        #     image=tensor(3×H×W),                      # 原始图像（RGB）
        #     gt_alpha_mask=tensor(1×H×W),              # alpha蒙版或 None
        #     image_name='r_001.png',                   # 图像文件名
        #     uid=0,                                    # 唯一ID
        #     time=0.123,                               # 当前相机对应帧的时间戳
        #     mask=None,                                # 可选的 mask
        #     depth=None,                               # 可选的深度图
        #     trans=array([0.0, 0.0, 0.0]),             # 相机平移
        #     scale=1.0,                                # 场景缩放比例
        #     world_view_transform=tensor(4×4),         # 相机变换矩阵 (世界到相机)
        #     projection_matrix=tensor(4×4),            # 投影矩阵
        #     full_proj_transform=tensor(4×4),          # 投影 * 视图矩阵
        #     camera_center=tensor(3,)                  # 相机中心位置
        # )

        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
                             
    # 从整个 viewpoint_stack 数据集中，抽取所有相机在某一个特定时间戳（timestamp）下对应的视角数据，即构造一个跨不同视角的“同一时间帧”的 Camera 列表。
    # 它的核心思路是：对于每个相机视角，取其第 timestamp 帧的数据。
        # 举例：假设 frame_length=60，表示每个视角有 60 帧；
        # 输入 timestamp=0，则你会选出索引 [0, 60, 120, 180, ...]；
        # 输入 timestamp=10，则索引为 [10, 70, 130, 190, ...]。
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 

    # | GUI (`custom_cam`)     | `render_training_image()` 渲染图 |
    # | ---------------------- | ----------------------------- |
    # | 你绕着人自由旋转，看到侧脸、背面、动态模糊等 | 固定 5 个正面角度，每隔 10 帧或 300 帧采样一次 |
    # | 可以调节视角时间戳（快速浏览时间）      | 渲染的是固定时间戳（如 timestamp=0）      |
    # 🎨 3. 图像细节 / 渲染质量差别？
    # ✅ 几乎没有差别（同样是调用 render(...) 函数）
    count = 0
    for iteration in range(first_iter, final_iter+1):    
        # ✅ 它的核心作用
        # 不是用于训练！
        # 而是：将高斯模型渲染成图像供外部 GUI（如可视化工具）实时预览使用。
        # | 项目        | GUI 渲染分支（你问的那行）                        | 模型训练分支（viewpoint\_stack）                  |
        # | --------- | -------------------------------------- | ----------------------------------------- |
        # | 数据来源      | 来自外部图形界面传来的相机 `custom_cam`             | 从数据集加载的训练相机（`viewpoint_stack`）            |
        # | 调用位置      | `while network_gui.conn != None` 这一段循环 | 主训练循环 `for iteration in ...` 中            |
        # | 调用目的      | 实时渲染当前相机图像，**发送回 GUI 端可视化**            | 用于计算 loss、反向传播、参数更新                       |
        # | 是否参与 loss | ❌ 不参与，纯推理                              | ✅ 是训练主数据来源                                |
        # | 渲染频率      | 每次 GUI 发来请求时调用                         | 每轮迭代训练时使用                                 |
        # | 输出图像      | 仅用于 GUI 显示，通常为 `uint8` 格式              | 用于与 GT 计算 L1/SSIM/Lpips loss，`float32` 张量 |
        # 🔍 为什么还要用这个 custom_cam？
        #     这是为了支持 交互式训练可视化，例如：
        #     用户拖动 GUI 中视角，实时看到当前模型渲染效果
        #     评估当前模型在任意相机下的可视化表现
        #     可远程可视化高斯渲染状态，查看训练进展
        #     这种机制就是 HUGS/FourDGS 等项目为什么能边训练边显示实时 3D 渲染的基础。
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                
                # 假设 network_gui.receive() 返回一个长度为 6 的元组，格式如下：
                # custom_cam = ...                  # 用于在 GUI 中显示或测试的相机视角（可能是实时移动相机）
                # do_training = ...                # 是否继续训练模型
                # pipe.convert_SHs_python = ...    # 控制管线是否用 Python 实现 SH（球谐光照）转换
                # pipe.compute_cov3D_python = ...  # 控制管线是否用 Python 实现协方差计算
                # keep_alive = ...                 # 保持 GUI 连接活跃
                # scaling_modifer = ...           # 缩放倍数（可调整模型显示大小）
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)

                    ##参数	含义
                    # custom_cam	当前的相机视角，类型是 MiniCam，由 GUI 传入，包含视锥体参数、投影矩阵、视图矩阵等
                    # gaussians	当前高斯模型（GaussianModel），里面包括了所有点的：位置、缩放、旋转、透明度、球谐系数等可学习参数
                    # pipe	渲染管线配置对象，控制是否使用 Python SH 转换、是否使用 Python 的协方差计算等
                    # background	背景颜色，通常为 [1,1,1] 或 [0,0,0] 的 CUDA Tensor
                    # scaling_modifer	缩放调节器，用于动态控制高斯的屏幕空间大小影响
                    # stage	当前阶段，"coarse" 或 "fine"，影响是否使用变形网络
                    # cam_type=scene.dataset_type	当前相机/数据集类型（例如 PanopticSports、Colmap、Blender），决定相机设置方式

                    # 这个 render() 函数会返回一个字典，包含渲染输出的各个分量，例如：
                    # {
                    #     "render": rendered_image,             # ⬅ 最终渲染图像（torch.Tensor）【我们就是取这个】
                    #     "viewspace_points": screenspace_points,  # 用于可视化/优化的屏幕空间坐标（有梯度）
                    #     "visibility_filter": radii > 0,       # 哪些点参与了投影、可见
                    #     "radii": radii,                       # 每个高斯点在屏幕上的半径（单位像素）
                    #     "depth": depth                        # 渲染的深度图（每个像素对应的深度）
                    # }
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]
                    
                    # 将网络渲染输出的 PyTorch 图像张量，转换成 [H, W, 3] 的 uint8 NumPy 图像，并打包成字节流以供外部使用。
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                
                # 这是用于 图形界面 GUI 实时预览或远程可视化（如 GUI 接收的 camera 参数），用于实时观察高斯渲染效果。
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每 1000 次迭代提升一次当前使用的 SH（球谐）系数阶数（逐步使用更复杂的光照建模）
            # 例如：            
            # 第 0-1000 步：只用 SH-0（即 RGB 常数）
            # 第 1000-2000 步：用 SH-1（线性系数）
            # 直到最大阶数 max_sh_degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                # viewpoint_cams 是一个 list，长度为 batch_size，元素是摄像头参数字典（包含位姿、内参、图像等）
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []
            while idx < batch_size :    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            # 变量名	内容
            #     images	渲染图像
            #     gt_images	Ground Truth 图像
            #     radii_list	每个高斯点投影到屏幕的半径
            #     visibility_filter_list	视锥裁剪掩码
            #     viewspace_point_tensor_list	用于 densify 的屏幕坐标误差量
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)

        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            # tv_loss 指的是 Temporal Smoothness Regularization Loss，
            # 也就是 时间平滑正则项。它是 FreeGaussians、4D-Gaussian Splatting、FreeTimeGS 等方法中，为了约束高斯在时间维度上的平滑变化而引入的一种正则损失，用来避免模型在学习动态场景时产生不连续、不自然的变化。
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        # render_training_image(...) 就是为了保存渲染图像结果到硬盘，方便你 不用 GUI 也能看到训练效果。你可以在训练过程中自动保存的文件夹里找到它们。
                        # render_training_image 渲染的是 训练集 / 测试集中的固定相机视角（Camera 对象），
                        # 而 GUI 使用的 custom_cam 是 外部控制的实时相机视角，
                        # 所以二者的 相机位姿（pose）、FoV、时间戳等可能完全不同。
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                # 更新当前可见高斯点在屏幕空间的最大半径（用作稀疏剔除依据）。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 系统记录每个高斯点的 屏幕梯度大小，用于之后判断该点是否需要复制或细化。
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                # 3. 动态设置 densify & pruning 阈值
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    # 在 fine 阶段，使用 线性插值的方式随着 iteration 逐步降低阈值 → 更加保守地 densify/prune
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                

                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 当前轮数达到可 densify 训练阶段，当前高斯点数量不超过 36w
                    # 将高梯度、opacity 合理的高斯点 复制，用于 finer 层次的建模。
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 高斯点数 > 20w（避免提前清除有效结构） 5. 条件触发 Prune（剔除稀疏/冗余点）
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    # 点的主动生长（grow） 
                    # 调用 grow 函数（不是复制已有点，而是新建点）：
                    # 参数 (5, 5)：表示在每张视图中添加 5×5 个新点（类似均匀网格初始化或噪声采样）；                    
                    # 这些点用于补充建模不足的区域，尤其在初始阶段建模不全时使用；
                    # 与 densify()（复制已有高斯点）不同，grow() 是完全新增点。
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    # 透明度重置（reset_opacity） 
                    # 每隔一定轮次（如 3000 iter），对所有点的 opacity 重新初始化；
                    # 这样可以“唤醒”那些原本已经趋于静态或不可见的点，避免模型陷入局部最优；
                    gaussians.reset_opacity()
                    
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
                
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    # coarse
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    # fine
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
