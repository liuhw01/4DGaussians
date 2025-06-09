import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        # ✏️ 举个例子
        #     假设某个点的位置是 (𝑥=1.2,𝑦=0.3，𝑧=2.8)，时间戳是 𝑡=0.5。 它会被送入：
        #     grid_feature = self.grid(torch.tensor([[1.2, 0.3, 2.8]]), torch.tensor([[0.5]]))
        #         在 grid 内部，这个位置和时间会被：
        #         投影到六个平面上（如 xy, yz, zx, xt, yt, zt）
        #         对每个平面，进行 2D bilinear interpolation，采样出一个特征向量
        #         所有平面特征拼接 → 得到该点在该时刻的 grid 特征向量（如维度是 64）
        #         这一步类似于将空间-时间中的任意一点，转换为高维特征向量，并作为 deform MLP 的输入。
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    # HexPlaneField 提供了一种高效、可插值、可训练的时空特征场，作为动态高斯变形的“感知输入源”。 它让模型能理解“我现在这个点，在这个时间，应该如何变化”。
    # 🚀 它相比 MLP 的优势
    #     特性	        MLP-only 模型	   加上 HexPlaneField
    #     局部感知能力	差	               强（查周围网格）
    #     参数可控性	    全局统一	           局部分布式、可学习
    #     插值能力	    差（不连续）	       强（平滑插值）
    #     时空一致性	    依赖训练调节	       天然支持
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            #直接将空间位置 + 时间作为输入拼接进 MLP：
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            # 调用 HexPlaneField 网格结构，输入是：
            # HexPlaneField 是一种 多平面因式分解的网格结构，通常用于表达 3D 空间 + 1D 时间的高维特征，目标是用少量参数捕捉复杂空间关系。
            # 📐 1.1 输入：空间位置 + 时间 (x, y, z, t)
            # 🧱 1.2 内部结构：
            #     它将高维特征场分解为若干个 平面投影（2D planes），比如：
            #         空间投影：
            #             (x, y)
            #             (x, z)
            #             (y, z)
            #         时空投影：
            #             (x, t)
            #             (y, t)
            #             (z, t)
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx


    # 🔁 基于当前帧时间 t，对每个高斯点的五个属性进行动态调整和变形，从而建模动态场景。
    # 输入的每个变量都已经过 PE（positional encoding）编码：
    #     输入变量	含义
    #     rays_pts_emb	点的位置（空间坐标 [x,y,z] 的PE）
    #     scales_emb	点的尺度（高斯半径）的PE
    #     rotations_emb	点的旋转（四元数）的PE
    #     opacity_emb	点的不透明度（sigmoid前值）
    #     shs_emb	点的 SH（球谐系数）
    #     time_emb	当前帧时间 t 的PE
    #     time_feature	对 time_emb 的映射结果（通过 timenet）
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        
        # 调用 query_time 函数，从 grid（HexPlane）中查表，并送入 MLP 得到最终特征：
        # 举例：
        # in：    
        #     query_time(
        #             rays_pts_emb = PE([[-1, 0, 1], [0.5, 0.5, 0.5]]),  # size [2, pos_pe_dim]
        #             time_emb = PE([0.2, 0.2])                         # size [2, time_pe_dim]
        #         )
        # out：
        #     grid(...) 会查到每个点在 4D 网格中的特征，经过 MLP 投影后，返回每个点的 hidden 向量，形如：
        #     hidden = [[0.12, -0.3, ..., 0.5], [0.21, -0.7, ..., 0.9]]  # shape = [2, W]
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)

        # ② 生成 mask（控制哪些点发生变化）
        if self.args.static_mlp:
            mask = self.static_mlp(hidden) # mask ∈ [0,1]
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3]) # 查询 DenseGrid 得到 mask
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)  # 默认全部变形
            
        # breakpoint()
        # ③ 点坐标变形（位置变换）
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]  # 不变形
        else:
            # 如果开启形变，则位置 pts = 原位置 * mask + dx，相当于加权偏移。
            # self.pos_deform 是一个小的 MLP，输入是 hidden（由 HexPlane 提取的特征）
            # 它的输出是 每个点的位移向量 dx ∈ ℝ³，用于对 3D 坐标 (x, y, z) 进行形变
            # 即：网络学会了对每个点施加一个“运动方向”或偏移量
                # 📌 举例：
                # 假设有 N = 1024 个高斯点，那么：
                # 输入：hidden.shape = [1024, 256]
                # 输出：dx.shape = [1024, 3]
            dx = self.pos_deform(hidden)
            
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            # 当 mask[i] ≈ 1：点将获得完整的偏移 dx[i]，表示其被认为是动态的
            # 当 mask[i] ≈ 0：该点不会被偏移，表示保持静态
            # rays_pts_emb[:,:3] 是输入点的原始 3D 坐标 xyz，后面是sin+cos等位置编码
            pts = rays_pts_emb[:,:3]*mask + dx

        # ⑦ SH系数变形（颜色）
        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds

        # 旋转
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        # 透明度
        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do

        # 球谐扰动
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs

    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list


# #🧠 核心作用
# deform_network = timenet + deformation_net
# 它主要分为两个子网络：
# timenet：将时间编码为特征向量（非线性处理时间）
# deformation_net：基于位置、时间等信息预测每个高斯点的变形量（Δposition, Δscale, Δrotation 等）

# 参数名	说明
# net_width	deformation MLP 的宽度
# defor_depth	deformation MLP 的深度
# timebase_pe, posbase_pe	时间 / 位置的 Positional Encoding 维度
# scale_rotation_pe, opacity_pe	对应属性的 Positional Encoding 维度
# grid_pe	是否对 Grid 特征进一步编码
# timenet_output	timenet 输出时间向量的维度
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        
        # 将时间的正余弦编码（times_ch = 2 * timebase_pe + 1）输入，并输出时间特征 t_feat，用于动态变形。
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))

        # 构建主干形变网络 Deformation，输入为位置 / 时间编码等，输出为多分支（dx、ds、dr、do、dshs）。
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)

        # self.pos_poc 是一个 position positional encoding（位置位置编码）频率表
        # posbase_pe 是用户配置的 位置频率编码维数（L）
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)

        # poc_fre() 是一种正余弦编码（Fourier embedding），例如 sin(2⁰·x), cos(2⁰·x), sin(2¹·x)...
        # 编码目的是提升模型对频率变化（如时间变化、空间位置）的拟合能力。
        # 举例：若 posbase_pe = 4，则
        # self.pos_poc = tensor([1., 2., 4., 8.])
        # poc_fre(point, pos_poc) 对 point 的每一维（如 x）进行如下编码：
        # [x, sin(x*1), cos(x*1), sin(x*2), cos(x*2), sin(x*4), cos(x*4), sin(x*8), cos(x*8)]
        # 共 1（原始）+ 2×4（频率编码）= 9 维
        # 对 x, y, z 进行拼接后得到一个 3×9=273×9=27 维向量
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)

        
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)

        #Deformation 网络接收时空嵌入后输出形变结果
            # Δx（位置偏移）
            # Δs（尺度偏移）
            # Δr（旋转偏移）
            # Δα（透明度偏移）
            # Δshs（SH 偏移）
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

# 原始数据+sin+cos[位置编码]
def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
