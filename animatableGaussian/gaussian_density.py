import torch
from typing import Optional, Dict, Any
import logging
import torch.nn as nn  # 新增：用于判断 nn.Parameter


class GaussianDensityController:
    """
    实现 GaussianBody 的 split_with_scale 策略：
    - s_eff = max(exp(sx), exp(sy), exp(sz)) 为触发依据
    - 支持固定阈值 eps_scale 或分位数阈值 quantile_threshold
    - 每隔 T_split 步检查，触发的点替换为两个尺度减半的新点，位置可按 s_eff 微扰
    - 同步更新模型参数与优化器状态
    """

    def __init__(
        self,
        eps_scale: float = 0.0125,
        use_scale_quantile: bool = False,
        quantile_threshold: float = 0.9,
        T_split: int = 500,
        perturbation_factor: float = 0.25,
        max_points: int = 50000,
        logger: Optional[logging.Logger] = None
    ):
        self.eps_scale = float(eps_scale)
        self.use_scale_quantile = bool(use_scale_quantile)
        self.quantile_threshold = float(quantile_threshold)
        self.T_split = int(T_split)
        self.perturbation_factor = float(perturbation_factor)
        self.max_points = int(max_points)
        self.logger = logger or logging.getLogger(__name__)
        self.split_ops = 0
        self.total_splits = 0

    # -------- 参数访问器 --------
    def get_xyz(self, model) -> torch.Tensor:
        if not hasattr(model, "v_template"):
            raise ValueError("Model does not have v_template")
        return model.v_template.reshape([-1, 3])

    def get_scales_log(self, model) -> torch.Tensor:
        return model.scales  # (N, 3) in log-space

    def get_rot(self, model) -> torch.Tensor:
        return model.rotations  # (N, 4)

    def get_opa(self, model) -> torch.Tensor:
        return model.opacity  # (N, 1)

    def get_sh_dc(self, model) -> Optional[torch.Tensor]:
        return getattr(model, "shs_dc", None)

    def get_sh_rest(self, model) -> Optional[torch.Tensor]:
        return getattr(model, "shs_rest", None)

    def get_weights(self, model) -> Optional[torch.Tensor]:
        return getattr(model, "weights", None)

    def point_count(self, model) -> int:
        if hasattr(model, "opacity"):
            return int(model.opacity.shape[0])
        if hasattr(model, "scales"):
            return int(model.scales.shape[0])
        return 0

    # -------- 主流程 --------
    @torch.no_grad()
    def split_with_scale(self, model, optimizer, iteration: int) -> Dict[str, Any]:
        if iteration % self.T_split != 0:
            return {"split_count": 0, "new_points": 0, "total_points": self.point_count(model)}

        scales_log = self.get_scales_log(model)           # (N, 3)
        scales = torch.exp(scales_log)                    # (N, 3)
        s_eff = scales.max(dim=-1)[0]                     # (N,)

        threshold = torch.quantile(s_eff, self.quantile_threshold) if self.use_scale_quantile \
            else torch.as_tensor(self.eps_scale, device=s_eff.device, dtype=s_eff.dtype)

        split_mask = s_eff > threshold
        M = int(split_mask.sum().item())
        if M == 0:
            return {"split_count": 0, "new_points": 0, "total_points": self.point_count(model)}

        cur = self.point_count(model)
        if cur + M > self.max_points:
            # 限额处理
            allow = self.max_points - cur
            if allow <= 0:
                self.logger.warning(f"[split_with_scale] point cap reached ({cur}), skip")
                return {"split_count": 0, "new_points": 0, "total_points": cur}
            idx = torch.where(split_mask)[0]
            choose = idx[torch.randperm(idx.numel(), device=idx.device)[:allow]]
            split_mask = torch.zeros_like(split_mask, dtype=torch.bool)
            split_mask[choose] = True
            M = allow

        stats = self._perform_split(model, optimizer, split_mask)
        self.split_ops += 1
        self.total_splits += M
        new_total = self.point_count(model)
        self.logger.info(f"[split_with_scale] iter={iteration} split={M} thr={float(threshold):.6f} count: {cur}->{new_total}")
        return {
            "split_count": M,
            "new_points": M,
            "total_points": new_total,
            "threshold": float(threshold),
        }

    @torch.no_grad()
    def _perform_split(self, model, optimizer, split_mask: torch.Tensor) -> Dict[str, Any]:
        xyz_all = self.get_xyz(model)                # (N,3)
        scales_log_all = self.get_scales_log(model)  # (N,3)
        rot_all = self.get_rot(model)                # (N,4)
        opa_all = self.get_opa(model)                # (N,1)
        sh_dc_all = self.get_sh_dc(model)            # (N,1,3)?
        sh_rest_all = self.get_sh_rest(model)        # (N,*,3)?
        w_all = self.get_weights(model)              # (N,J)?

        # 切片被分裂项
        xyz = xyz_all[split_mask]
        scales_log = scales_log_all[split_mask]
        rot = rot_all[split_mask]
        opa = opa_all[split_mask]
        sh_dc = sh_dc_all[split_mask] if sh_dc_all is not None else None
        sh_rest = sh_rest_all[split_mask] if sh_rest_all is not None else None
        w = w_all[split_mask] if w_all is not None else None
        M = int(xyz.shape[0])

        # 位置扰动与尺度更新
        if self.perturbation_factor > 0:
            s_eff = torch.exp(scales_log).max(dim=-1, keepdim=True)[0]  # (M,1)
            noise = torch.randn_like(xyz) * (self.perturbation_factor * s_eff)
            xyz1 = xyz + noise
            xyz2 = xyz - noise
        else:
            xyz1 = xyz.clone()
            xyz2 = xyz.clone()

        # 尺度减半（log-space 减 ln 2）
        half = float(torch.log(torch.tensor(2.0, device=scales_log.device, dtype=scales_log.dtype)))
        scales_new = scales_log - half
        rot_new = rot.clone()
        opa_new = opa.clone()
        sh_dc_new = sh_dc.clone() if sh_dc is not None else None
        sh_rest_new = sh_rest.clone() if sh_rest is not None else None
        w_new = w.clone() if w is not None else None

        # 合并两个新点
        add_xyz = torch.cat([xyz1, xyz2], dim=0)          # (2M,3)
        add_scales = torch.cat([scales_new, scales_new], 0)
        add_rot = torch.cat([rot_new, rot_new], 0)
        add_opa = torch.cat([opa_new, opa_new], 0)
        add_sh_dc = torch.cat([sh_dc_new, sh_dc_new], 0) if sh_dc_new is not None else None
        add_sh_rest = torch.cat([sh_rest_new, sh_rest_new], 0) if sh_rest_new is not None else None
        add_w = torch.cat([w_new, w_new], 0) if w_new is not None else None

        # 计算保留项索引（未分裂）
        keep_mask = ~split_mask
        keep_indices = torch.where(keep_mask)[0]

        # 更新 v_template（buffer，不在优化器中）
        vt = model.v_template.reshape([-1, 3])
        vt_keep = vt[keep_mask]
        vt_updated = torch.cat([vt_keep, add_xyz], dim=0)
        model.v_template.data = vt_updated.reshape(model.v_template.shape[0], -1, 3)

        # 新增：同步 normalized_vertices（与 v_template 强相关，下一次 forward 要用）
        with torch.no_grad():
            # 复用 SMPLModel 初始化时的归一化方式：以第一个 player 的范围归一化
            vt0 = model.v_template[0]  # (N', 3)
            vmin = vt0.min(dim=0).values * 1.05
            vmax = vt0.max(dim=0).values * 1.05
            denom = torch.clamp(vmax - vmin, min=1e-8)
            normalized = (model.v_template - vmin) / denom  # (B, N', 3)
            if hasattr(model, "normalized_vertices"):
                model.normalized_vertices.data = normalized
            else:
                # 若未注册过，注册为 buffer
                model.register_buffer('normalized_vertices', normalized)

        # 更新其余参数（这些是 nn.Parameter，优化器会跟踪）
        model.scales.data = torch.cat([model.scales[keep_mask], add_scales], dim=0)
        model.rotations.data = torch.cat([model.rotations[keep_mask], add_rot], dim=0)
        model.opacity.data = torch.cat([model.opacity[keep_mask], add_opa], dim=0)
        if sh_dc_all is not None:
            model.shs_dc.data = torch.cat([model.shs_dc[keep_mask], add_sh_dc], dim=0)
        if sh_rest_all is not None:
            model.shs_rest.data = torch.cat([model.shs_rest[keep_mask], add_sh_rest], dim=0)
        if w_all is not None:
            model.weights.data = torch.cat([model.weights[keep_mask], add_w], dim=0)

        # 新增：如果存在 aos（在未启用 AOEncoder 时使用），同步到新尺寸
        if hasattr(model, "aos") and isinstance(model.aos, torch.Tensor):
            model.aos.data = torch.ones_like(model.opacity)

        # 同步优化器 state（对每个受影响的参数）
        self._update_optimizer_states(model, optimizer, keep_indices, num_new=2 * M)
        return {"split": M}

    @torch.no_grad()
    def _update_optimizer_states(self, model, optimizer, keep_indices: torch.Tensor, num_new: int):
        """
        将受影响参数的优化器 state 从旧形状 (N, ...) 更新到新形状 (kept + num_new, ...)
        规则：
          new_state = cat(old_state[keep_indices], zeros(num_new, ...), dim=0)
        失败时对该参数重置优化器状态，避免训练中断。
        同时需要同步参数的梯度张量 p.grad 到新形状，否则会在 optimizer.step 时与 exp_avg 等状态产生形状不匹配。
        """
        if optimizer is None:
            return

        # 兼容：Lightning 可能返回列表
        opt_list = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]

        # 仅跟踪在优化器中的 nn.Parameter（且第 0 维随点数变化的参数）
        tracked = {}
        for attr in ("opacity", "scales", "rotations", "weights", "shs_dc", "shs_rest", "displacements"):
            if hasattr(model, attr):
                p = getattr(model, attr)
                if isinstance(p, nn.Parameter):
                    tracked[id(p)] = p

        # 新增：先对受影响参数的梯度做重映射，确保与后续 state 的尺寸一致
        def remap_state_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                return t
            kept = t.index_select(0, keep_indices)
            tail_shape = list(t.shape)
            tail_shape[0] = num_new
            zeros = torch.zeros(tail_shape, dtype=t.dtype, device=t.device)
            return torch.cat([kept, zeros], dim=0)

        for p in tracked.values():
            if p.grad is not None and torch.is_tensor(p.grad):
                try:
                    p.grad = remap_state_tensor(p.grad)
                except Exception as e:
                    # 梯度重映射失败时，置空梯度，避免本次 step 冲突
                    p.grad = None
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Reset grad for param due to remap failure: {e}")

        for opt in opt_list:
            for group in opt.param_groups:
                for p in group["params"]:
                    if id(p) not in tracked:
                        continue
                    state = opt.state.get(p, None)
                    if not state:
                        continue
                    for k, v in list(state.items()):
                        if torch.is_tensor(v) and v.dim() > 0:
                            try:
                                state[k] = remap_state_tensor(v)
                            except Exception as e:
                                # 若映射失败，回退到安全策略：重置该参数的优化器状态，并清空梯度
                                if p in opt.state:
                                    del opt.state[p]
                                if p.grad is not None:
                                    p.grad = None
                                if hasattr(self, "logger"):
                                    self.logger.warning(f"Reset optimizer state for param due to remap failure: {e}")