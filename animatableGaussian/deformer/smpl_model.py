import torch.nn as nn
from animatableGaussian.deformer.encoder.position_encoder import SHEncoder, DisplacementEncoder
from animatableGaussian.deformer.encoder.time_encoder import AOEncoder
from animatableGaussian.deformer.deformation import deform_network  # 新增导入
import torch
import pickle
import os
import numpy as np
from animatableGaussian.deformer.lbs import lbs
# TODO(keye): 这里可以kernel优化
from simple_knn._C import distCUDA2


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class SMPLModel(nn.Module):
    def __init__(self, model_path, max_sh_degree=0, max_freq=4, gender="male", num_repeat=15, num_players=1, use_point_color=False, use_point_displacement=False, enable_ambient_occlusion=False, use_deform_network=True):
        super().__init__()

        self.num_players = num_players
        self.enable_ambient_occlusion = enable_ambient_occlusion
        self.use_point_displacement = use_point_displacement
        self.use_point_color = use_point_color

        smpl_path = os.path.join(
            model_path, 'SMPL_{}'.format(gender.upper()))
        v_template = np.loadtxt(os.path.join(
            smpl_path, 'v_template.txt'))
        weights = np.loadtxt(os.path.join(
            smpl_path, 'weights.txt'))
        kintree_table = np.loadtxt(os.path.join(
            smpl_path, 'kintree_table.txt'))
        J = np.loadtxt(os.path.join(
            smpl_path, 'joints.txt'))
        self.register_buffer('v_template', torch.Tensor(
            v_template)[None, ...].repeat(
                [self.num_players, 1, 1]))
        dist2 = torch.clamp_min(
            distCUDA2(self.v_template[0].cuda()), 0.0000001)[..., None].repeat([num_repeat, 3])
        self.v_template = self.v_template.repeat([1, num_repeat, 1])
        self.v_template += (torch.rand_like(self.v_template) - 0.5) * \
            dist2.cpu() * 20
        dist2 /= num_repeat
        self.weights = nn.Parameter(
            torch.Tensor(weights).repeat([num_repeat, 1]))
        self.parents = kintree_table[0].astype(np.int64)
        self.parents[0] = -1

        self.J = nn.Parameter(torch.Tensor(
            J)[None, ...].repeat([self.num_players, 1, 1]))

        minmax = [self.v_template[0].min(
            dim=0).values * 1.05,  self.v_template[0].max(dim=0).values * 1.05]
        self.register_buffer('normalized_vertices',
                             (self.v_template - minmax[0]) / (minmax[1] - minmax[0]))

        if use_point_displacement:
            self.displacements = nn.Parameter(
                torch.zeros_like(self.v_template))
        else:
            self.displacementEncoder = DisplacementEncoder(
                encoder="hash", num_players=num_players)

        n = self.v_template.shape[1] * num_players

        if use_point_color:
            self.shs_dc = nn.Parameter(torch.zeros(
                [n, 1, 3]))
            self.shs_rest = nn.Parameter(torch.zeros(
                [n, (max_sh_degree + 1) ** 2 - 1, 3]))
        else:
            self.shEncoder = SHEncoder(max_sh_degree=max_sh_degree,
                                       encoder="hash", num_players=num_players)
        self.opacity = nn.Parameter(inverse_sigmoid(
            0.2 * torch.ones((n, 1), dtype=torch.float)))
        self.scales = nn.Parameter(
            torch.log(torch.sqrt(dist2)).repeat([num_players, 1]))
        rotations = torch.zeros([n, 4])
        rotations[:, 0] = 1
        self.rotations = nn.Parameter(rotations)

        if enable_ambient_occlusion:
            self.aoEncoder = AOEncoder(
                encoder="hash", max_freq=max_freq, num_players=num_players)
        self.register_buffer("aos", torch.ones_like(self.opacity))

        # 新增 deform_network 实例
        if use_deform_network:
            # 创建一个简单的 args 对象用于初始化 deform_network
            class DeformArgs:
                def __init__(self):
                    self.net_width = 64
                    self.timebase_pe = 4
                    self.defor_depth = 6
                    self.posebase_pe = 10
                    self.scale_rotation_pe = 2
                    self.opacity_pe = 2
                    self.timenet_width = 64
                    self.timenet_output = 32
                    self.grid_pe = 0
                    
            deform_args = DeformArgs()
            self.deform_network = deform_network(deform_args)
            self.use_deform_network = True
        else:
            self.use_deform_network = False

    def configure_optimizers(self, training_args):
        l = [
            {'params': [self.weights],
                'lr': training_args.weights_lr, "name": "weights"},
            {'params': [self.J], 'lr': training_args.joint_lr, "name": "J"},
            {'params': [self.opacity],
                'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scales],
                'lr': training_args.scaling_lr, "name": "scales"},
            {'params': [self.rotations],
                'lr': training_args.rotation_lr, "name": "rotations"}
        ]

        if self.enable_ambient_occlusion:
            l.append({'params': self.aoEncoder.parameters(),
                      'lr': training_args.ao_lr, "name": "aoEncoder"})
        if self.use_point_displacement:
            l.append({'params': [self.displacements],
                      'lr': training_args.displacement_lr, "name": "displacements"})
        else:
            l.append({'params': self.displacementEncoder.parameters(),
                      'lr': training_args.displacement_encoder_lr, "name": "displacementEncoder"})
        if self.use_point_color:
            l.append({'params': [self.shs_dc],
                      'lr': training_args.shs_lr, "name": "shs"})
            l.append({'params': [self.shs_rest],
                      'lr': training_args.shs_lr/20.0, "name": "shs"})
        else:
            l.append({'params': self.shEncoder.parameters(),
                      'lr': training_args.sh_encoder_lr, "name": "shEncoder"})
        
        # 新增：如果使用 deform_network，添加其参数到优化器
        if self.use_deform_network:
            l.append({'params': self.deform_network.get_mlp_parameters(),
                      'lr': training_args.deform_lr if hasattr(training_args, 'deform_lr') else 1e-4, "name": "deform_network_mlp"})
            l.append({'params': self.deform_network.get_grid_parameters(),
                      'lr': training_args.deform_grid_lr if hasattr(training_args, 'deform_grid_lr') else 1e-3, "name": "deform_network_grid"})

        return torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def forward(self, body_pose, global_orient, transl, time, iteration, total_iteration, is_use_ao=False):
        """
        Returns:
            vertices (torch.Tensor[N, 3]) : 
            opacity (torch.Tensor[N, 1]) : 
            scales (torch.Tensor[N, 3]) : 
            rotations (torch.Tensor[N, 4]) : 
            shs (torch.Tensor[N, (max_sh_degree + 1) ** 2, 3]) : 
            aos (torch.Tensor[N, 1]) : 
            transforms (torch.Tensor[N, 3]) : 
        """
        full_body_pose = torch.cat(
            [global_orient[:, None, :], body_pose], dim=1)

        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            shs = self.shEncoder(self.normalized_vertices)
        if self.enable_ambient_occlusion:
            aos = self.aoEncoder(self.normalized_vertices, time)
        else:
            aos = self.aos
        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            v_displaced = self.v_template + \
                self.displacementEncoder(self.normalized_vertices)
        

        # 新增：使用 deform_network 进行变形
        if self.use_deform_network:
            # 准备输入参数
            points = v_displaced.reshape([-1, 3])  # 展平为 [N, 3]
            # 修复：传入 log-scales（不再提前 exp）
            scales = self.scales.reshape([-1, 3])
            rotations = torch.nn.functional.normalize(self.rotations.reshape([-1, 4]))  # 归一化四元数
            
            # 准备姿态向量：将 full_body_pose 展平
            pose = full_body_pose.reshape(-1)  # 展平姿态参数
            
            # 变形前的数据统计
            points_orig_mean = points.mean(dim=0)
            points_orig_std = points.std()
            scales_orig_actual = torch.exp(self.scales.reshape([-1, 3]))
            scales_orig_mean = scales_orig_actual.mean(dim=0)
            scales_orig_std = scales_orig_actual.std()
            rotations_orig_mean = rotations.mean(dim=0)
            
            # 调用 deform_network.forward
            deformed_points, deformed_scales, deformed_rotations, offset = self.deform_network.forward(
                point=points,
                scales=scales,
                rotations=rotations,
                pose=pose,
                iteration=iteration,
                total_iteration=total_iteration
            )
            
            # # 变形后的数据统计
            # points_deformed_mean = deformed_points.mean(dim=0)
            # points_deformed_std = deformed_points.std()
            # scales_deformed_mean = deformed_scales.mean(dim=0)
            # scales_deformed_std = deformed_scales.std()
            # rotations_deformed_mean = deformed_rotations.mean(dim=0)
            
            # # 计算变形量
            # points_delta = (deformed_points - points).abs()
            # scales_delta = (deformed_scales - scales).abs()
            # rotations_delta = (deformed_rotations - rotations).abs()
            
            # # 监控输出（每100次迭代输出一次，避免日志过多）
            # if iteration % 100 == 0 or iteration < 10:
            #     print(f"\n=== Deformation Monitor (iter {iteration}/{total_iteration}) ===")
            #     print(f"Points - Original mean: {points_orig_mean.detach().cpu().numpy()}")
            #     print(f"Points - Deformed mean: {points_deformed_mean.detach().cpu().numpy()}")
            #     print(f"Points - Delta mean: {points_delta.mean().item():.6f}, max: {points_delta.max().item():.6f}")
            #     print(f"Points - Std change: {points_orig_std.item():.6f} -> {points_deformed_std.item():.6f}")
                
            #     print(f"Scales - Original mean: {scales_orig_mean.detach().cpu().numpy()}")
            #     print(f"Scales - Deformed mean: {scales_deformed_mean.detach().cpu().numpy()}")
            #     print(f"Scales - Delta mean: {scales_delta.mean().item():.6f}, max: {scales_delta.max().item():.6f}")
            #     print(f"Scales - Std change: {scales_orig_std.item():.6f} -> {scales_deformed_std.item():.6f}")
                
            #     print(f"Rotations - Original mean: {rotations_orig_mean.detach().cpu().numpy()}")
            #     print(f"Rotations - Deformed mean: {rotations_deformed_mean.detach().cpu().numpy()}")
            #     print(f"Rotations - Delta mean: {rotations_delta.mean().item():.6f}, max: {rotations_delta.max().item():.6f}")
                
            #     if offset is not None:
            #         print(f"Offset - mean: {offset.mean().item():.6f}, std: {offset.std().item():.6f}")
            #     else:
            #         print("Offset - None")
            #     print("=" * 60)
            
            # # 检查是否有异常值
            # if torch.isnan(deformed_points).any() or torch.isinf(deformed_points).any():
            #     print(f"WARNING: NaN or Inf detected in deformed_points at iteration {iteration}")
            # if torch.isnan(deformed_scales).any() or torch.isinf(deformed_scales).any():
            #     print(f"WARNING: NaN or Inf detected in deformed_scales at iteration {iteration}")
            # if torch.isnan(deformed_rotations).any() or torch.isinf(deformed_rotations).any():
            #     print(f"WARNING: NaN or Inf detected in deformed_rotations at iteration {iteration}")
            
            # # 检查变形是否过大（可能导致渲染问题）
            # if points_delta.max() > 1.0:  # 如果最大位移超过1.0单位
            #     print(f"WARNING: Large point displacement detected at iteration {iteration}: max delta = {points_delta.max().item():.6f}")
            # if scales_delta.max() > 0.5:  # 如果最大尺度变化超过0.5
            #     print(f"WARNING: Large scale change detected at iteration {iteration}: max delta = {scales_delta.max().item():.6f}")
            
            # 使用变形后的参数
            v_displaced = deformed_points.reshape(v_displaced.shape)  # 恢复原始形状
            # 修复：对返回的 log-scales 做 exp，得到实际尺度
            scales_out = torch.exp(deformed_scales).reshape(self.scales.shape)
            rotations_out = deformed_rotations.reshape(self.rotations.shape)  # 已经是归一化四元数
            
            # 可选：添加一个开关来临时禁用变形（用于对比实验）
            use_deformation_result = True  # 设为 False 可以禁用变形结果，用于对比
            if not use_deformation_result:
                print("WARNING: Using original parameters instead of deformed ones for comparison")
                scales_out = torch.exp(self.scales)
                rotations_out = torch.nn.functional.normalize(self.rotations)
                v_displaced = v_displaced  # 保持原始位置
                
        else:
            # 原始逻辑：不使用 deform_network
            scales_out = torch.exp(self.scales)
            rotations_out = torch.nn.functional.normalize(self.rotations)

        T = lbs(full_body_pose, transl, self.J, self.parents, self.weights)

        return v_displaced.reshape([-1, 3]), torch.sigmoid(self.opacity), scales_out, rotations_out, shs, aos, T[:, :, :3, :].reshape([-1, 3, 4])
