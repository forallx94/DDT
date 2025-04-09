from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dcn_cuda_forward',
    ext_modules=[
        CUDAExtension('dcn_cuda_forward', ['./forward.cu',],
        extra_compile_args={'cxx': [], 'nvcc': [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--use_fast_math",
            "-O3",
        ]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='dcn_cuda_backward',
    ext_modules=[
        CUDAExtension('dcn_cuda_backward', ['./backward.cu',],
        extra_compile_args={'cxx': [], 'nvcc': [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--use_fast_math",
            "-O3",
        ]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


# setup(
#     name='mycuda',
#     ext_modules=[
#         CUDAExtension('mycuda', ['./backward.cu',],
#                       extra_compile_args={'cxx': [], 'nvcc': [
#                           "-O3",
#                           "-DCUDA_HAS_FP16=1",
#                           "-D__CUDA_NO_HALF_OPERATORS__",
#                           "-D__CUDA_NO_HALF_CONVERSIONS__",
#                           "-D__CUDA_NO_HALF2_OPERATORS__",
#                       ]}
#                       ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )