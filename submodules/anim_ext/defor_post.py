# deform_post_autograd.py
import torch
from torch.autograd import Function
import defor_cuda  # compiled extension module name (from setup)

class DeformPostFunction(Function):
    @staticmethod
    def forward(ctx, point, offset, scales, rotations):
        # point: (N,3), offset: (N,10), scales: (N,3), rotations: (N,4)
        assert point.is_cuda and offset.is_cuda and scales.is_cuda and rotations.is_cuda
        N = point.shape[0]
        means3D = torch.empty((N,3), device=point.device, dtype=point.dtype)
        scales_out = torch.empty((N,3), device=point.device, dtype=point.dtype)
        rotations_out = torch.empty((N,4), device=point.device, dtype=point.dtype)

        # launch CUDA forward
        defor_cuda.launch_forward(point.contiguous(), offset.contiguous(),
                                        scales.contiguous(), rotations.contiguous(),
                                        means3D, scales_out, rotations_out)
        # save for backward
        ctx.save_for_backward(point, offset, scales, rotations)
        return means3D, scales_out, rotations_out

    @staticmethod
    def backward(ctx, grad_means3D, grad_scales_out, grad_rot_out):
        point, offset, scales, rotations = ctx.saved_tensors
        N = point.shape[0]

        # allocate grads
        grad_point = torch.zeros_like(point)
        grad_offset = torch.zeros_like(offset)
        grad_scales_in = torch.zeros_like(scales)
        grad_rot_in = torch.zeros_like(rotations)

        # ensure contiguous
        defor_cuda.launch_backward(
            grad_means3D.contiguous(), grad_scales_out.contiguous(), grad_rot_out.contiguous(),
            point.contiguous(), offset.contiguous(), scales.contiguous(), rotations.contiguous(),
            grad_point, grad_offset, grad_scales_in, grad_rot_in
        )
        # return in the same order as inputs to forward
        return grad_point, grad_offset, grad_scales_in, grad_rot_in

# convenience wrapper
def deform_post(point, offset, scales, rotations):
    return DeformPostFunction.apply(point, offset, scales, rotations)