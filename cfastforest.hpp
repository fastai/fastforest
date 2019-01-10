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

class GcTree {
public:
    //GcForest* parent_;
    int c_, n_;
    float* acts_;  // rows subset used to train tree
    float** preds_;  // rows subset used to train tree
    Node* root_;
    float *lT, *cutvals, *tmp;
    int *lC, *cutidxs;
};

