# fast-forest

A forest that is fast. Works on OS X and linux so far.

## Building

Requires `cmake`. To install and run a test:

```bash
conda install pybind11 # use conda not pip
cmake .
make fastforest
python test.py
```

NB: the makefile is created by cmake, so don't edit it. Instead, edit CMakeLists.txt and then rerun "`cmake .`".

## Building for PyPi publishing

To use the mechanism for publishing to pypi:

```bash
conda install pybind11 # use conda not pip
python setup.py build_ext --inplace
python test.py
```

