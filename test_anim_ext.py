import torch
import posenc_cuda


from unittest import TestCase
from tqdm import tqdm

from animatableGaussian.deformer.deformation import get_embedder

from submodules.anim_ext.embedder import EmbedderModule


class RefEmbedder(torch.nn.Module):
    def __init__(self, iteration, multires,kick_in_iter=0, full_band_iter=50000):
        super().__init__()
        self.embedder = get_embedder(iteration, multires, kick_in_iter, full_band_iter)[0]
    
    def forward(self, x):
        return self.embedder(x)


class TestPosEnc(TestCase):
    def test_forward(self):
        kick_in_iter = 171
        total_iter = 1413

        for k in tqdm(range(1, 200000, 5000)):
            a = torch.randn(k, 3, requires_grad=True).cuda()
            b = a.clone().detach().requires_grad_(True)
            for i in tqdm(range(200, 1000, 2)):
                out = posenc_cuda.forward(a, i, kick_in_iter, total_iter)
                ref = RefEmbedder(i, 6, kick_in_iter, total_iter)(b)
                s = ref.sum()
                s.backward()
                print('grad', b.grad)

                import pdb; pdb.set_trace()
                if not torch.allclose(out, ref, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"mismatch at iter {i} k {k}")

class TestEmbedderModule(TestCase):
    def test_forward_backward(self):
        kick_in_iter = 171
        total_iter = 1413


        for k in tqdm(range(1, 200000, 5000)):
            a = torch.randn(k, 3, device='cuda', requires_grad=True)
            b = a.clone().detach().requires_grad_(True)
            for i in tqdm(range(200, 1000, 2)):
                out = EmbedderModule(3, kick_in_iter, total_iter)(a, i)
                so = out.sum()
                so.backward()

                ref = RefEmbedder(i, 6, kick_in_iter, total_iter)(b)
                s = ref.sum()
                s.backward()
                if not torch.allclose(a.grad, b.grad, atol=1e-2, rtol=1e-2):
                    import pdb; pdb.set_trace()
                    raise ValueError(f"mismatch at iter {i} k {k}")
