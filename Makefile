all: fastforest.cpp setup.py test.py
	./test.py

fastforest.cpp: cfastforest.cpp cfastforest.hpp fastforest.pyx
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf build *.o *.so

