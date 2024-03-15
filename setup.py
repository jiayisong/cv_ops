import torch
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        raise EnvironmentError('CUDA is required to compile!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='cv_ops',
        packages=find_packages(),
        include_package_data=True,
        package_data={'cv_ops': ['*/*.so']},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='cv_ops.bbox3d',
                sources=[
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='grid_sample_deterministic_cuda',
                module='cv_ops.grid_sample_deterministic',
                sources=[
                    'src/grid_sample_deterministic.cpp',
                    'src/grid_sample_deterministic_cuda.cu',
                ]
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        version='1.0'
    )
