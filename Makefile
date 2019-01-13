all: fastforest.cpp setup.py test.py cfastforest.hpp
	./test.py

fastforest.cpp: cfastforest.cpp cfastforest.hpp fastforest.pyx setup.py
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf build *.o *.so fastforest.cpp

