#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

extern int train_ff(float *x_, float *y, int r, int c);

class Node {
public:
    int start, n, bestPred;
    struct Node* left, *right;
    bool isLeft;
    float cutoff;
    float value, gini;

    //Node(int start, int n, Node* parent, bool isLeft);
    bool IsTerminal();
};

class FastTree;

class FastForest {
public:
    const int NTREE = 300, MIN_NODE = 8;
    const double PROP_TRAIN = 0.8, PROP_OOB = 0.5;

    float* X;
    float* y;

    FastTree* trees;
    int N,C;

    FastForest(float* X_, float* y_, int n, int c);
    //~FastForest();
    void build();
    //double* predict(array<array<float>^>^ rows);
};

class FastTree {
public:
    const int MAXN = 160, CUTOFF_DIVISOR = 100;

    FastTree(FastForest* parent);
    FastForest* parent_;
    int c_, n_;
    float* acts_;  // rows subset used to train tree
    float** preds_;  // rows subset used to train tree
    Node* root_;
};

