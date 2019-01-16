# distutils: language = c++
# cython: language_level=3
# distutils: sources = cfastforest.cpp

import pandas as pd, numpy as np
cimport numpy as np
from cython cimport view
from libcpp cimport bool
np.import_array()

cdef extern from "cfastforest.hpp":
    FastForest* trainFF(float *x_, float *y, int nrows, int ncols)
    cdef cppclass Node:
        Node *left
        Node *right
        bool isLeft
        int start, n, bestPred
        float cutoff, value, gini
    cdef cppclass FastForest:
        FastTree** trees
        int nrows, ncols
        float* predict(float* rows, int nrows, int ncols)
    cdef cppclass FastTree:
        int nrows, ncols
        Node* root
        float predict(float* arr)

cdef makenode(Node *ptr):
    res = PyNode()
    res.ptr = ptr
    return res

cdef class PyNode:
    cdef Node *ptr
    def __cinit__(self): self.ptr = NULL
    @property
    def left(self): return makenode(self.ptr.left)
    @property
    def right(self): return makenode(self.ptr.right)
    @property
    def isLeft(self): return self.ptr.isLeft
    @property
    def start(self): return self.ptr.start
    @property
    def n(self): return self.ptr.n
    @property
    def bestPred(self): return self.ptr.bestPred
    @property
    def cutoff(self): return self.ptr.cutoff
    @property
    def value(self): return self.ptr.value
    @property
    def gini(self): return self.ptr.gini

cdef class PyFastTree:
    cdef FastTree *ptr
    def __cinit__(self): self.ptr = NULL
    @property
    def root(self): return makenode(self.ptr.root)
    cpdef predict(self, float[:] row): return self.ptr.predict(&row[0])

cdef class PyFastForest:
    cdef FastForest *ptr
    def __cinit__(self, x, y):
        nrows, ncols = x.shape
        x = np.ascontiguousarray(x).astype(np.float32)
        cdef float [:,:] xv = x
        y = np.ascontiguousarray(y).astype(np.float32)
        cdef float [:] yv = y
        self.ptr = trainFF(&xv[0,0], &yv[0], nrows, ncols)

    def get_tree(self,i):
        res = PyFastTree()
        res.ptr = self.ptr.trees[i]
        return res

cpdef train(x, y):
    print(x.shape)
    res = PyFastForest(x, y)
    return res

