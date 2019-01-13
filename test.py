#!/usr/bin/env python
from fastforest import *
import numpy as np

"""
X = np.vstack([
    np.linspace(0,1,100),
    np.linspace(0,1,100),
    np.random.random(100)]).T
y = np.array([0.]*50 + [1.]*50)
train(X,y)
exit()
"""

def split_vals(a,n): return a[:n], a[n:]

def show(df, name, node):
    bp = df.columns[node.bestPred]
    print(f'{name} n {node.n} v {node.value} g {node.gini} bp {bp} c {node.cutoff}')

def test_it(df, y, ff):
    x = df.values
    #print(x[:5])
    print(x.mean(0))
    print(y.mean())
    #exit()
    print("---")
    tree = ff.get_tree(0)
    root = tree.root
    show(df, 'root', root)
    show(df, 'l   ', root.left)
    show(df, 'r   ', root.right)
    print("---")

    xs = x[np.array([0,90])]
    #print('xs', xs)
    print('ys', y[0], y[90])
    print('x0', tree.predict(xs[0]))
    """
    cdef float [:] rowv2 = xs[1]
    #print('x90', x[90], tree.Predict(&rowv2[0]))
    print('x90', tree.Predict(&rowv2[0]))
    print("---")
    cdef float [:,:] rowsv = xs
    cdef view.array rr = <float[:2]> ff.Predict(&rowsv[0,0], 2, c)
    res = np.asarray(rr)
    print("====")
    print(res)
    print("====")
    """

df_raw = pd.read_pickle('tmp/bulldozers-raw')
df_trn = pd.read_pickle('tmp/df_trn')
y_trn = np.load(open('tmp/y_trn', 'rb'))

n_valid = 160000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)

cols = ['YearMade', 'MachineHoursCurrentMeter', 'UsageBand', 'ProductSize', 'Enclosure', 'saleYear',
       'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type',
       'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls'
       ]
#x_sub = X_valid[['YearMade', 'MachineHoursCurrentMeter']]
x_sub = X_valid
#print(X_valid.columns)
ff = train(x_sub.values, y_valid)
test_it(x_sub, y_valid, ff)

