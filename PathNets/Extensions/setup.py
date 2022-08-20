from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(name = "PathNetOptimizer", 
    ext_modules = [
        CppExtension("PathNetOptimizer", ["Combinatorial.cxx"]),
    ],
    cmdclass = {"build_ext" : BuildExtension}
)


