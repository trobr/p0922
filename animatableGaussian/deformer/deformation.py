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

# from scene.grid import HashHexPlane

from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw

def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.

    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.

    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)

    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)

    return norm_q3

def apply_rotation(q1, q2):
    """
    Applies a rotation to a quaternion.

    Parameters:
    q1 (Tensor): The original quaternion.
    q2 (Tensor): The rotation quaternion to be applied.

    Returns:
    Tensor: The resulting quaternion after applying the rotation.
    """
    # Extract components for readability
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Compute the product of the two quaternions
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Combine the components into a new quaternion tensor
    q3 = torch.tensor([w3, x3, y3, z3])

    # Normalize the resulting quaternion
    q3_normalized = q3 / torch.norm(q3)

    return q3_normalized

# 路径 A：
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width if hasattr(args, 'net_width') else 64
        timebase_pe = args.timebase_pe if hasattr(args, 'timebase_pe') else 4
        defor_depth = args.defor_depth if hasattr(args, 'defor_depth') else 0
        posbase_pe = args.posebase_pe if hasattr(args, 'posebase_pe') else 10
        scale_rotation_pe = args.scale_rotation_pe if hasattr(args, 'scale_rotation_pe') else 2
        opacity_pe = args.opacity_pe if hasattr(args, 'opacity_pe') else 2
        timenet_width = args.timenet_width if hasattr(args, 'timenet_width') else 64
        timenet_output = args.timenet_output if hasattr(args, 'timenet_output') else 32
        grid_pe = args.grid_pe if hasattr(args, 'grid_pe') else 0
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.time_poc = self.time_poc.to(device)
        self.pos_poc = self.pos_poc.to(device)
        self.rotation_scaling_poc = self.rotation_scaling_poc.to(device)
        self.opacity_poc = self.opacity_poc.to(device)
        self.apply(initialize_weights)
        self.shs_deform = nn.Sequential(
            nn.Linear(108, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(292, 256),  # skip 后：256 + 36 = 292
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),  # 新增的一层隐藏层
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10)
        ).cuda()
        # print(self)

    # def save_deform_weights(self, model_path, iteration):
    #     out_weights_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}")
    #     os.makedirs(out_weights_path, exist_ok=True)
    #     torch.save(self.deformation_net.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    # def load_deform_weights(self, model_path, iteration=-1):
    #     from utils.system_utils import searchForMaxIteration
    #     if iteration == -1:
    #         loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
    #     else:
    #         loaded_iter = iteration
    #     weights_path = os.path.join(model_path, 'deform.pth')# "point_cloud/iteration_{}".format(loaded_iter)
    #     self.deformation_net.load_state_dict(torch.load(weights_path))

    def forward(self, point, scales=None, rotations=None,pose=None,iteration=None,total_iteration=None):
        return self.forward_dynamic(point, scales, rotations,pose,iteration,total_iteration)

    @property
    def get_aabb(self):
        # 路径A不使用网格，返回None
        return None

    @property
    def get_empty_ratio(self):
        # 路径A不使用empty_voxel，返回0.0
        return 0.0

    def forward_static(self, points):
        # 路径A下没有静态网格分支，直接返回输入
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, pose=None,iteration=None,total_iteration=None):
        #point_emb = poc_fre(point,self.pos_poc)
        device = point.device
        pose = pose.to(device)
        # point_emb0=torch.cat([pose.unsqueeze(0).repeat(point.shape[0], 1) , point], dim=-1)
        # point_emb = nerf_positional_encoding(point_emb0)
        # 位置编码

        # TODO(keye): 这里get_embedder链路可以要优化，不要每次都重新创建
        pos_emb0 = get_embedder(iteration, multires=6, kick_in_iter=0.1 * total_iteration, full_band_iter=total_iteration)[0](point)
        #point_emb = torch.cat([pose.unsqueeze(0).repeat(point.shape[0], 1), pos_emb0], dim=-1)

        # 拼接姿态
        if pose.shape[0] == 1:
            point_emb = torch.cat([pose.repeat(point.shape[0], 1), pos_emb0], dim=-1)
        else:
            point_emb = torch.cat([pose.unsqueeze(0).repeat(point.shape[0], 1), pos_emb0], dim=-1)

        # 暂时"复用"了 self.deformation_net.shs_deform 这串层当做通用校正 MLP（并在 i==4 时做一次 skip，把 pos_emb0 再拼进去）。
        for i in range(len(self.shs_deform)):
            if i==4:
                point_emb = torch.cat([point_emb, pos_emb0], dim=-1)
            point_emb = self.shs_deform[i](point_emb)
        offset = point_emb
        
        means3D = point + offset[..., :3]
        scales = scales + offset[..., 3:6]
        # 适度约束 log-scales 范围，避免数值爆炸（可按需要调整上下界）
        # scales = torch.clamp(scales, min=math.log(1e-6), max=math.log(1.0))
        
        # === 修复四元数处理路径（不使用增益系数）===
        delta_rot = offset[..., 6:]
        q1 = delta_rot.clone()
        q1[:, 0] = 1.0  # w分量设为1
        
        # 归一化q1，确保是单位四元数
        q1 = F.normalize(q1, p=2, dim=1)
        
        # 确保输入旋转q2也是归一化的
        q2 = F.normalize(rotations, p=2, dim=1)
        
        # 四元数乘积，并在前后都做归一化
        q1_wxyz = quat_wxyz_to_xyzw(q1)  # 转换为roma格式
        q2_wxyz = quat_wxyz_to_xyzw(q2)
        q_result_wxyz = quat_product(q1_wxyz, q2_wxyz)  # roma四元数乘积
        q_result = quat_xyzw_to_wxyz(q_result_wxyz)  # 转换回来
        
        # 最终归一化输出四元数
        rotations = F.normalize(q_result, p=2, dim=1)
        
        return means3D, scales, rotations, offset

    def get_mlp_parameters(self):
        # 仅返回路径A中实际存在的MLP参数
        return list(self.shs_deform.parameters()) + list(self.timenet.parameters())

    def get_grid_parameters(self):
        # 路径A没有grid参数
        return []

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)#将input_data、正弦值和余弦值在最后一个维度上拼接起来
    return input_data_emb

def nerf_positional_encoding(input_data, L=10):
    device = input_data.device
    freq_bands = (2.0 ** torch.arange(L, device=device).float()) * torch.pi
    input_data_emb = (input_data.unsqueeze(-1) * freq_bands).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_encoded = torch.cat([input_data_sin, input_data_cos], dim=-1)
    return input_data_encoded

# TODO(keye): 这里可以kernel优化
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']#xyz
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']#5
        N_freqs = self.kwargs['num_freqs']#6

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)#2^0-2^5

        # get hann window weights
        kick_in_iter = torch.tensor(self.kwargs['kick_in_iter'],
                                    dtype=torch.float32)
        t = torch.clamp(self.kwargs['iteration'] - kick_in_iter, min=0.)
        N = self.kwargs['full_band_iter'] - kick_in_iter
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(iteration,multires,kick_in_iter=0, full_band_iter=50000, is_identity=0):
    if is_identity == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iteration': iteration,
        'kick_in_iter': kick_in_iter,
        'full_band_iter': full_band_iter,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim