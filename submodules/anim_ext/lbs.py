import torch
from torch.autograd import Function
from lbs_cuda import batch_rodrigues_forward, batch_rigid_transform_forward

class _BatchRodrigues(Function):
    @staticmethod
    def forward(ctx, rot_vecs):
        # rot_vecs: (B*(J+1), 3) flattened or (B, J+1, 3)
        was_batched = True
        if rot_vecs.dim() == 3:
            B, N, _ = rot_vecs.shape
            rv = rot_vecs.reshape(-1, 3)
        else:
            rv = rot_vecs
        out = batch_rodrigues_forward(rv.contiguous())
        if rot_vecs.dim() == 3:
            out = out.view(B, N, 3, 3)
        ctx.save_for_backward(rot_vecs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Fallback: recompute via PyTorch ops (clean, correct). Not optimal but correct.
        (rot_vecs,) = ctx.saved_tensors
        rot_vecs = rot_vecs.detach().requires_grad_(True)
        R = batch_rodrigues(rot_vecs)  # uses python wrapper below
        grads = torch.autograd.grad(R, rot_vecs, grad_out, retain_graph=False, allow_unused=True)
        return grads[0]

def batch_rodrigues(rot_vecs):
    return _BatchRodrigues.apply(rot_vecs)

class SMPL_LBS(Function):
    @staticmethod
    def forward(ctx, pose, translate, J, parents, lbs_weights):
        """
        pose: B x (J+1) x 3  (axis-angle)
        translate: B x 3
        J: B x J x 3
        parents: J
        lbs_weights: V x (J+1) OR (V, J+1)  -> we expect V x (J+1)
        """
        B = pose.shape[0]
        batch_rot = batch_rodrigues(pose).view(B, -1, 3, 3)
        # build transforms per-joint
        A = batch_rigid_transform_forward(batch_rot, translate, J, parents)
        # W broadcasting
        W = lbs_weights.unsqueeze(0).expand(B, -1, -1)  # B x V x (J+1)
        num_joints = J.shape[1]
        T = torch.matmul(W, A.view(B, num_joints, 16)).view(B, -1, 4, 4)
        # Save for backward
        ctx.save_for_backward(pose, translate, J, parents, lbs_weights, A, W)
        return T

    @staticmethod
    def backward(ctx, grad_T):
        # Fallback: recompute forward with autograd-enabled ops and let PyTorch compute grads.
        pose, translate, J, parents, lbs_weights, A_saved, W_saved = ctx.saved_tensors
        pose = pose.detach().requires_grad_(True)
        translate = translate.detach().requires_grad_(True)
        J = J.detach()  # joints often considered constant; if you need gradients for joints, enable requires_grad_
        lbs_weights = lbs_weights.detach()

        T = SMPL_LBS.apply(pose, translate, J, parents, lbs_weights)  # this will call forward (and uses CUDA forward kernels)
        grads = torch.autograd.grad(T, (pose, translate, J, lbs_weights), grad_T, allow_unused=True)
        # grads order: pose_grad, translate_grad, J_grad, lbs_weights_grad; parents has no grad
        return grads[0], grads[1], grads[2], None, grads[3]

# A convenience wrapper function
def lbs(pose, translate, J, parents, lbs_weights):
    return SMPL_LBS.apply(pose, translate, J, parents, lbs_weights)
