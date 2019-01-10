# distutils: language = c++

import pandas as pd, numpy as np
cimport numpy as np

cdef extern from "cfastforest.cpp":
    int train_ff(float *x_, float *y, int r, int c)

#cdef void train_ff(np.ndarray[float, ndim=2, mode="c"] a): train_ff(&a[0,0], a.shape[0], a.shape[1])

def split_vals(a,n): return a[:n], a[n:]

cpdef foo():
    df_raw = pd.read_pickle('tmp/bulldozers-raw')
    df_trn = pd.read_pickle('tmp/df_trn')
    y_trn = np.load(open('tmp/y_trn', 'rb'))

    n_valid = 12000
    n_trn = len(df_trn)-n_valid
    X_train, X_valid = split_vals(df_trn, n_trn)
    y_train, y_valid = split_vals(y_trn, n_trn)
    raw_train, raw_valid = split_vals(df_raw, n_trn)

    x_sub = X_train[['YearMade', 'MachineHoursCurrentMeter']]
    cdef np.ndarray[float, ndim=2, mode="c"] x = np.ascontiguousarray(x_sub.astype(np.float32).values)
    cdef np.ndarray[float, ndim=1] y = np.ascontiguousarray(y_train).astype(np.float32)
    cdef int r = train_ff(&x[0,0], &y[0], x.shape[0], x.shape[1])

    print(x[:5,:5])
    print(y_train[:5])
    print(r)

