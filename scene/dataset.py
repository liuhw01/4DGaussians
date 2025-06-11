from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                # 尝试从 self.dataset[index] 中解析 (image, w2c, time) 三元组：
                #     image: 图像 Tensor
                #     w2c: 世界到相机的 R（旋转）+ T（平移）
                #     time: 该帧的时间戳
                image, w2c, time = self.dataset[index]
                R,T = w2c
                
                   #      视野 (FoVx)
                   #     <--------->
                   #      __________
                   #     /          \
                   #    /            \
                   #   /              \    ↑
                   #  /                \   | y方向的视野（FoVy）
                   # ------------------    ↓
                   #       摄像平面
                # FoVx：水平方向的视场角（Field of View in x-axis）
                # FoVy：垂直方向的视场角（Field of View in y-axis）
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
                
            # Camera(
            #     R=...,                 # 旋转矩阵（3x3）
            #     T=...,                 # 平移向量（3,）
            #     FoVx=..., FoVy=...,   # 水平 / 垂直视场角（由 focal 计算）
            #     image=...,             # 图像 Tensor
            #     time=...,              # 时间戳
            #     mask=...,              # 可选遮罩
            #     gt_alpha_mask=None,    # 透明度 mask（可选）
            #     image_name=str(index),
            #     uid=index
            # )
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
