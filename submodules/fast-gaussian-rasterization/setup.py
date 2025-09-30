from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


import os

# os.environ['CC'] = 'ccache gcc'
# os.environ['CXX'] = 'ccache g++'
# os.environ['NVCC'] = 'nvcc'  # nvcc 自带缓存机制，如果希望用 sccache 可以替换

setup(
    name='fast_rasterizer',
    ext_modules=[
        CUDAExtension(
            name='fast_rasterizer',
            sources=[
                "src/backward.cu",
                'src/fast_rasterizer_autograd.cpp',
                'src/forward.cu',
                'src/rasterization_api.cu',
            ],
            include_dirs= ["include"],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '--use_fast_math',
                    '-maxrregcount=64',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True),}
)