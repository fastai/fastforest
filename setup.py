# python setup.py build_ext --inplace
# See <foo>.html for annotated python with C code
from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize
from glob import glob

extensions = [
    Extension(
        'fastforest',
        glob('*.pyx'),
        extra_compile_args=["-std=c++11"])
]

setup(
    name='fastforest',
    ext_modules=cythonize(extensions),
    zip_safe=False,
    include_dirs=[np.get_include()]
)

