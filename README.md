# fast-forest

A forest that is fast. Requires `cmake`. To install and run a test:

```bash
conda install pybind11
cmake .
make fastfastory
python test.py
```

NB: the makefile is created by cmake, so don't edit it. Instead, edit CMakeLists.txt and then rerun "`cmake .`".

Works on OS X and linux so far.
