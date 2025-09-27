from PIL import Image
import torch
import torch.nn as nn
import hydra
import time
import numpy as np
import pytorch_lightning as pl
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import os
from torch.cuda.amp import custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from animatableGaussian.utils import ssim, l1_loss
import inspect
# 新增: AIAP损失函数导入
import torch.nn.functional as F
# 顶部：新增导入
from animatableGaussian.gaussian_density import GaussianDensityController


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


class NeRFModel(pl.LightningModule):
    def __init__(self, opt):
        super(NeRFModel, self).__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(opt.deformer)
        self.training_args = opt.training_args
        self.sh_degree = opt.max_sh_degree
        self.lambda_dssim = opt.lambda_dssim
        # 新增: GaussianDensityController 配置
        self.density_controller = GaussianDensityController(
            eps_scale=getattr(opt, 'eps_scale', 0.0125),
            use_scale_quantile=getattr(opt, 'use_scale_quantile', False),
            quantile_threshold=getattr(opt, 'quantile_threshold', 0.9),
            T_split=getattr(opt, 'T_split', 500),
            perturbation_factor=getattr(opt, 'perturbation_factor', 0.25),
            max_points=getattr(opt, 'max_points', 50000),
        )
        # 新增：densify 冻结步数（从 opt 读取，未配置时默认 200）
        self.densify_freeze_iters = getattr(opt, 'densify_freeze_iters', 200)
        self.lambda_isopos = getattr(opt, 'lambda_isopos', 0.0)  # 位置等距权重
        self.lambda_isocov = getattr(opt, 'lambda_isocov', 0.0)  # 协方差等距权重
        self.aiap_k = getattr(opt, 'aiap_k', 6)  # KNN邻居数
        
        # 新增: AIAP内存控制参数
        self.aiap_max_points = getattr(opt, 'aiap_max_points', 8000)  # 最大参与AIAP的点数
        self.aiap_chunk_size = getattr(opt, 'aiap_chunk_size', 2000)  # 分块大小
        
        self.evaluator = Evaluator()
        if not os.path.exists("val"):
            os.makedirs("val")
        if not os.path.exists("test"):
            os.makedirs("test")

        self.epoch_start_time = None
        self.epoch_times = []
    
    def on_fit_start(self):
        self.model.deform_network.set_total_iteration(self._get_total_training_steps())
    
    def on_test_start(self):
        self.model.deform_network.set_total_iteration(1000)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.perf_counter()
    
    def on_train_epoch_end(self):
        elapsed = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        # self.log("epoch_time_sec", elapsed, prog_bar=True)

    def on_train_end(self):
        mean_epoch = np.mean(self.epoch_times)
        std_epoch = np.std(self.epoch_times)

        print("="*40)
        print(f"Total training time: {sum(self.epoch_times):.2f} sec")
        print(f"Epoch time mean ± std: {mean_epoch:.2f} ± {std_epoch:.2f} sec")
        print("="*40)

    @torch.no_grad()
    def build_knn_idx(self, x_can: torch.Tensor, k: int):
        """
        在 canonical 集合上建 KNN 邻接索引，纯 PyTorch 实现，支持分块和下采样。
        x_can: (N, D)
        返回 nn_ix: (N_sub, k+1), indices: 下采样索引映射
        """
        N, D = x_can.shape
        device = x_can.device
        
        # 下采样控制：如果点数超过 aiap_max_points，随机采样
        if N > self.aiap_max_points:
            indices = torch.randperm(N, device=device)[:self.aiap_max_points]
            x_can_sub = x_can[indices]
        else:
            indices = torch.arange(N, device=device)
            x_can_sub = x_can
        
        N_sub = x_can_sub.shape[0]
        
        # 分块计算 KNN 以控制峰值显存
        if N_sub <= self.aiap_chunk_size:
            # 小规模直接计算
            dist_matrix = torch.cdist(x_can_sub, x_can_sub)  # (N_sub, N_sub)
            _, nn_ix = torch.topk(dist_matrix, k + 1, dim=1, largest=False, sorted=True)
        else:
            # 大规模分块计算
            nn_ix = torch.zeros(N_sub, k + 1, dtype=torch.long, device=device)
            
            for start_idx in range(0, N_sub, self.aiap_chunk_size):
                end_idx = min(start_idx + self.aiap_chunk_size, N_sub)
                chunk = x_can_sub[start_idx:end_idx]
                
                # 对当前块计算与整个下采样集合的距离
                dist_chunk = torch.cdist(chunk, x_can_sub)  # (chunk_size, N_sub)
                _, chunk_nn_ix = torch.topk(dist_chunk, k + 1, dim=1, largest=False, sorted=True)
                nn_ix[start_idx:end_idx] = chunk_nn_ix
        
        return nn_ix, indices  # 返回索引和下采样映射

    def aiap_loss(self, x_can: torch.Tensor, x_def: torch.Tensor, nn_ix: torch.Tensor = None, k: int = 5):
        """
        AIAP: 保持 canonical 与 deformed 空间中，局部邻域的成对距离一致。
        支持分块计算和自动下采样以控制显存，纯 PyTorch 实现。
        x_can, x_def: (N, D)，形状必须相同
        nn_ix: (N_sub, K) 的近邻索引，可传入缓存；若为 None 则内部构建
        k: 近邻数（不含自身）
        返回: 标量 L1 损失
        """
        if x_can.shape != x_def.shape:
            raise ValueError("x_can and x_def must have the same shape")

        N = x_can.shape[0]
        device = x_can.device

        # 若未给索引，则构建带下采样的 KNN
        if nn_ix is None:
            nn_ix, indices = self.build_knn_idx(x_can, k)  # (N_sub, k+1)
            # 对应地下采样 deformed 点
            x_can_sub = x_can[indices]
            x_def_sub = x_def[indices]
        else:
            # 使用传入的索引，假设已经是下采样后的
            x_can_sub = x_can
            x_def_sub = x_def
            indices = torch.arange(x_can_sub.shape[0], device=device)

        N_sub = x_can_sub.shape[0]
        
        # 自适配：检测第 1 个是否是"自身"（距离为0），是的话就去掉它
        has_self = (nn_ix[:, 0] == torch.arange(N_sub, device=device)).all()
        if has_self:
            nn_ix_use = nn_ix[:, 1 : k + 1]  # 去掉自身，保留 k 个邻居
        else:
            nn_ix_use = nn_ix[:, :k]

        # 分块计算距离以控制显存
        chunk_size = min(self.aiap_chunk_size, N_sub)
        total_loss = 0.0
        num_chunks = 0
        
        for start_idx in range(0, N_sub, chunk_size):
            end_idx = min(start_idx + chunk_size, N_sub)
            
            # 当前块的点和邻居索引
            x_can_chunk = x_can_sub[start_idx:end_idx]  # (chunk_size, D)
            x_def_chunk = x_def_sub[start_idx:end_idx]  # (chunk_size, D)
            nn_ix_chunk = nn_ix_use[start_idx:end_idx]  # (chunk_size, k)
            
            # 使用 no_grad 计算距离以节省显存
            with torch.no_grad():
                # 计算 canonical 空间中当前块到其邻居的距离
                neighbors_can = x_can_sub[nn_ix_chunk]  # (chunk_size, k, D)
                d_can = torch.norm(x_can_chunk.unsqueeze(1) - neighbors_can, dim=2)  # (chunk_size, k)
                
                # 计算 deformed 空间中当前块到其邻居的距离
                neighbors_def = x_def_sub[nn_ix_chunk]  # (chunk_size, k, D)
                d_def = torch.norm(x_def_chunk.unsqueeze(1) - neighbors_def, dim=2)  # (chunk_size, k)
            
            # 计算损失（重新启用梯度）
            chunk_loss = F.l1_loss(d_can, d_def)
            total_loss += chunk_loss
            num_chunks += 1

        # 返回平均损失
        if num_chunks > 0:
            return total_loss / num_chunks
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def forward(self, camera_params, model_param, time, iteration, total_iteration, render_point=False, train=True, return_aux_info=False):
        torch.cuda.nvtx.range_push("f-forwad")
        torch.cuda.nvtx.range_push("f-pre")
        is_use_ao = (not train) or self.current_epoch > 3
        torch.cuda.nvtx.range_pop()

        # 根据被实例化的deformer.forward签名，决定是否传入iteration/total_iteration
        
        torch.cuda.nvtx.range_push("f-model")
        model_kwargs = dict(time=time, is_use_ao=is_use_ao, iteration=iteration, total_iteration=total_iteration, **model_param)

        verts, opacity, scales, rotations, shs, aos, transforms = self.model(**model_kwargs)
        torch.cuda.nvtx.range_pop()
        
        # 新增: 如果需要辅助信息（用于AIAP损失计算），则获取canonical数据
        torch.cuda.nvtx.range_push("f-aux")
        aux_info = {}
        if return_aux_info and train and hasattr(self.model, 'v_template'):
            # 获取canonical顶点和尺度
            canonical_verts = self.model.v_template.reshape([-1, 3]).detach()
            canonical_scales = torch.exp(self.model.scales).detach()  # 原始log-scales转换为实际尺度
            
            # 获取协方差矩阵（用于L_isocov计算）
            def get_covariance_matrix(scales, rotations):
                """从尺度和旋转计算协方差矩阵"""
                # scales: (N, 3), rotations: (N, 4) 四元数
                N = scales.shape[0]
                
                # 四元数转旋转矩阵
                q = F.normalize(rotations, dim=1)  # 确保四元数归一化
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                
                # 构建旋转矩阵 R (N, 3, 3)
                R = torch.zeros(N, 3, 3, device=scales.device, dtype=scales.dtype)
                R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
                R[:, 0, 1] = 2 * (x*y - w*z)
                R[:, 0, 2] = 2 * (x*z + w*y)
                R[:, 1, 0] = 2 * (x*y + w*z)
                R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
                R[:, 1, 2] = 2 * (y*z - w*x)
                R[:, 2, 0] = 2 * (x*z - w*y)
                R[:, 2, 1] = 2 * (y*z + w*x)
                R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
                
                # 尺度矩阵 S (N, 3, 3)
                S = torch.diag_embed(scales)  # (N, 3, 3)
                
                # 协方差矩阵 C = R * S * S^T * R^T = R * S^2 * R^T
                S_squared = torch.diag_embed(scales**2)  # (N, 3, 3)
                C = torch.bmm(torch.bmm(R, S_squared), R.transpose(-2, -1))  # (N, 3, 3)
                
                # 将协方差矩阵展平为向量以便计算距离
                return C.reshape(N, -1)  # (N, 9)
            
            canonical_cov = get_covariance_matrix(canonical_scales, 
                                                F.normalize(self.model.rotations.detach(), dim=1))
            deformed_cov = get_covariance_matrix(scales, F.normalize(rotations, dim=1))
            
            aux_info = {
                'canonical_verts': canonical_verts,
                'deformed_verts': verts,
                'canonical_cov': canonical_cov,
                'deformed_cov': deformed_cov
            }
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("f-render")
        means2D = torch.zeros_like(
            verts, dtype=verts.dtype, requires_grad=True, device=verts.device)
        try:
            means2D.retain_grad()
        except:
            pass
        raster_settings = GaussianRasterizationSettings(
            sh_degree=self.sh_degree,
            prefiltered=False,
            debug=False, **camera_params
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        cov3D_precomp = None
        if render_point:
            colors_precomp = torch.rand_like(scales)
            scales /= 10
            opacity *= 100
            shs = None
        else:
            colors_precomp = None
        image, radii = rasterizer(
            means3D=verts,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            aos=aos,
            transforms=transforms,
            cov3D_precomp=cov3D_precomp)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_pop()
        if return_aux_info:
            return image, aux_info
        return image

    def _get_total_training_steps(self):
        # 优先使用 max_steps（若设置且有效）
        if hasattr(self.trainer, "max_steps") and self.trainer.max_steps and self.trainer.max_steps > 0:
            return int(self.trainer.max_steps)
        # 其次尝试 estimated_stepping_batches
        est = getattr(self.trainer, "estimated_stepping_batches", None)
        if est is not None:
            return int(est)
        # 回退到 max_epochs * num_training_batches（若可用）
        max_epochs = getattr(self.trainer, "max_epochs", None)
        num_batches = getattr(self.trainer, "num_training_batches", None)
        if max_epochs is not None and num_batches is not None:
            try:
                return int(max_epochs * num_batches)
            except Exception:
                pass
        # 最差回退，避免除零等问题
        return 1

    def training_step(self, batch, batch_idx):
        torch.cuda.nvtx.range_push("training_step")
        torch.cuda.nvtx.range_push("t-pre")
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        iteration = int(self.global_step)
        total_iteration = self._get_total_training_steps()
        torch.cuda.nvtx.range_pop()
    
        # 新增: 获取渲染图像和辅助信息
        torch.cuda.nvtx.range_push("t-forward")
        image, aux_info = self(camera_params, model_param, batch["time"], iteration, total_iteration, return_aux_info=True)
        gt_image = batch["gt"]
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("t-l1-loss")
        # 基础损失
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.lambda_dssim) * Ll1 + \
            self.lambda_dssim * (1.0 - ssim(image, gt_image))
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("t-aiap-loss")
        # 新增: AIAP损失计算
        if aux_info and self.lambda_isopos > 0 or self.lambda_isocov > 0:
            try:
                # 位置等距损失 L_isopos
                if self.lambda_isopos > 0:
                    loss_isopos = self.aiap_loss(
                        aux_info['canonical_verts'], 
                        aux_info['deformed_verts'], 
                        k=self.aiap_k
                    )
                    loss += self.lambda_isopos * loss_isopos
                    # self.log('train_loss_isopos', loss_isopos, prog_bar=False)
                
                # 协方差等距损失 L_isocov  
                if self.lambda_isocov > 0:
                    loss_isocov = self.aiap_loss(
                        aux_info['canonical_cov'], 
                        aux_info['deformed_cov'], 
                        k=self.aiap_k
                    )
                    loss += self.lambda_isocov * loss_isocov
                    # self.log('train_loss_isocov', loss_isocov, prog_bar=False)
                    
            except Exception as e:
                print(f"Warning: AIAP loss computation failed: {e}")
                # 如果AIAP计算失败，继续训练但不添加该损失项
                pass
        torch.cuda.nvtx.range_pop()
        
        # self.log('train_loss', loss, prog_bar=True)
        torch.cuda.nvtx.range_pop()
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        iteration = int(self.global_step)
        total_iteration = self._get_total_training_steps()
        rgb = self(camera_params, model_param, batch["time"], iteration, total_iteration)
        rgb_gt = batch["gt"]
        image = torch.cat((rgb, rgb_gt), dim=2)
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"val/{self.current_epoch}.png")

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        # 直接设定一个足够大的总迭代，避免访问训练数据
        total_iteration = 1000
        iteration = total_iteration
        rgb = self(camera_params, model_param, batch["time"], iteration, total_iteration, train=False)
        rgb_gt = batch["gt"]
        losses = {
            # add some extra loss here
            **self.evaluator(rgb[None], rgb_gt[None]),
            "rgb_loss": (rgb - rgb_gt).square().mean(),
        }
        image = rgb
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}.png")
        image = rgb_gt
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}_gt.png")

        for k, v in losses.items():
            self.log(f"test/{k}", v, on_epoch=True, batch_size=1)
        return {}

    def on_test_epoch_end(self):
        # 只在测试结束时做一次基准对比（与Lightning最终汇总一致）
        cb = getattr(self.trainer, "callback_metrics", {})
        keys = ["test/lpips", "test/psnr", "test/rgb_loss", "test/ssim"]

        # 将最终聚合指标取成 float
        final_metrics = {}
        for k in keys:
            v = cb.get(k, None)
            if v is None:
                continue
            try:
                final_metrics[k] = float(v.detach().cpu().item())
            except Exception:
                try:
                    final_metrics[k] = float(v)
                except Exception:
                    pass

        # 基准值与方向（按你提供的要求）
        baselines = {
            "lpips": 0.026445978079915047,          # 越小越好
            "psnr": 29.054935482177734,             # 越大越好
            "rgb_loss": 0.0012658877210059762,      # 越小越好
            "ssim": 0.9714913491134644,             # 越大越好
        }
        direction = {
            "lpips": "lower_better",
            "psnr": "higher_better",
            "rgb_loss": "lower_better",
            "ssim": "higher_better",
        }

        # 输出与基准对比
        print("──────────────── Baseline Comparison (final only) ────────────────")
        for k_disp in ["lpips", "psnr", "rgb_loss", "ssim"]:
            k = f"test/{k_disp}"
            if k not in final_metrics:
                continue
            cur = final_metrics[k]
            base = baselines[k_disp]
            diff = cur - base
            better = (direction[k_disp] == "lower_better" and diff < 0) or \
                     (direction[k_disp] == "higher_better" and diff > 0)
            tie = abs(diff) < 1e-12
            pct_text = f"{(abs(diff)/abs(base)*100.0):.2f}%" if abs(base) > 0 else "N/A"
            direction_text = "越小越好" if direction[k_disp] == "lower_better" else "越大越好"

            if tie:
                verdict = "持平"
                change_text = "变化 0.000000 (0.00%)"
            else:
                verdict = "更好" if better else "更差"
                change_text = f"{'改善' if better else '变差'} {abs(diff):.6f} ({pct_text})"

            print(f"{k:>16s}: 当前={cur:.6f}, 基准={base:.6f}，{direction_text}，结论：{verdict}，{change_text}")
        print("──────────────────────────────────────────────────────────────────")

    def configure_optimizers(self):
        return self.model.configure_optimizers(self.training_args)

    def on_after_backward(self):
        """在反向传播后周期性执行 split_with_scale 操作"""
        # 只在训练阶段启用
        if not self.training:
            return

        iteration = int(self.global_step)

        # 新增：前期冻结 densify
        if iteration < getattr(self, 'densify_freeze_iters', 0):
            # 可选：按 T_split 打印一次冻结提示
            if iteration % self.density_controller.T_split == 0:
                print(f"[Densify Frozen] iter={iteration}, freeze_until={self.densify_freeze_iters}")
            return

        # 兼容 Lightning 返回多优化器的情况
        opt = self.optimizers()
        optimizer = opt[0] if isinstance(opt, (list, tuple)) else opt

        with torch.no_grad():
            stats = self.density_controller.split_with_scale(self.model, optimizer, iteration)

        # 记录统计
        # if stats.get("split_count", 0) > 0:
        #     # 说明：这里转成 float 可以避免 Lightning 的类型转换提示（可选）
        #     self.log('split_count', float(stats["split_count"]), prog_bar=False)
        #     self.log('total_points', float(stats["total_points"]), prog_bar=True)
        #     if "threshold" in stats:
        #         self.log('split_threshold', float(stats["threshold"]), prog_bar=False)

    def _resize_smpl_to_target_points(self, target_N: int):
        """将底层 SMPLModel 的点相关参数/缓冲区调整为 target_N 的形状，以便严格加载 checkpoint。"""
        m = self.model
        # 设备与精度沿用现有张量
        dev = None
        dt = None
        if hasattr(m, "opacity"):
            dev = m.opacity.device
            dt = m.opacity.dtype
        else:
            # 回退：从任一参数获取设备
            try:
                p = next(m.parameters())
                dev, dt = p.device, p.dtype
            except StopIteration:
                dev, dt = torch.device("cpu"), torch.float32

        def _reset_param(name, shape):
            if hasattr(m, name):
                obj = getattr(m, name)
                if isinstance(obj, nn.Parameter):
                    new_p = nn.Parameter(torch.zeros(*shape, device=obj.device, dtype=obj.dtype))
                    setattr(m, name, new_p)
                elif torch.is_tensor(obj):
                    # buffer：就地替换 data
                    new_t = torch.zeros(*shape, device=obj.device, dtype=obj.dtype)
                    obj.data = new_t
                else:
                    # 未注册为 parameter/buffer：直接设为 Tensor
                    setattr(m, name, torch.zeros(*shape, device=dev, dtype=dt))

        # 推导维度
        # weights 的第二维保持不变（通常是 24）
        if hasattr(m, "weights"):
            Jdim = m.weights.shape[1]
        else:
            Jdim = 24  # 合理默认

        # 调整参数
        _reset_param("opacity", (target_N, 1))
        _reset_param("scales", (target_N, 3))
        _reset_param("rotations", (target_N, 4))
        _reset_param("weights", (target_N, Jdim))

        # 调整 buffer
        _reset_param("v_template", (1, target_N, 3))
        _reset_param("normalized_vertices", (1, target_N, 3))
        _reset_param("aos", (target_N, 1))

    def load_state_dict(self, state_dict, strict: bool = True):
        """在严格加载前，根据 checkpoint 中的点数调整底层 SMPLModel 的形状，解决 densify 导致的形状不匹配。"""
        # 从 checkpoint 的 state_dict 中探测目标 N
        target_N = None
        # 优先从低开销张量推断
        if "model.opacity" in state_dict:
            target_N = state_dict["model.opacity"].shape[0]
        elif "model.scales" in state_dict:
            target_N = state_dict["model.scales"].shape[0]
        elif "model.rotations" in state_dict:
            target_N = state_dict["model.rotations"].shape[0]
        elif "model.v_template" in state_dict:
            # v_template: [1, N, 3]
            target_N = state_dict["model.v_template"].shape[1]

        if target_N is not None and hasattr(self, "model") and hasattr(self.model, "opacity"):
            try:
                current_N = self.model.opacity.shape[0]
            except Exception:
                current_N = None
            if current_N is None or current_N != target_N:
                # 预调整底层 SMPLModel 的相关参数/缓冲区形状
                self._resize_smpl_to_target_points(target_N)

        # 再执行标准加载
        return super().load_state_dict(state_dict, strict=strict)
        # 新增：densify 冻结步数（从 opt 读取，未配置时默认 200）
        self.densify_freeze_iters = getattr(opt, 'densify_freeze_iters', 200)
        
        # 记录统计
        if stats.get("split_count", 0) > 0:
            self.log('split_count', stats["split_count"], prog_bar=False)
            self.log('total_points', stats["total_points"], prog_bar=True)
            if "threshold" in stats:
                self.log('split_threshold', stats["threshold"], prog_bar=False)
