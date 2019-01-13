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
        extra_link_args=['-fopenmp', '-Wl,-rpath,/usr/local/Cellar/gcc/8.2.0/lib/gcc/8'],
        extra_compile_args=["-std=c++11", '-fopenmp', '-Wno-cpp'])
]

setup(
    name='fastforest',
    ext_modules=cythonize(extensions), #, gdb_debug=True),
    zip_safe=False,
    include_dirs=[np.get_include()]
)

