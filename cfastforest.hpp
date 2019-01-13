#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <numeric>
#include <stack>
#include <assert.h>
#include <tgmath.h>
#include <algorithm>

using namespace std;

class Node {
public:
    struct Node *left, *right;
    bool isLeft;
    int start, n, bestPred;
    float cutoff, value, gini;

    Node(int start, int n, Node *parent, bool isLeft);
    bool IsTerminal();
};

class FastTree;

class FastForest {
public:
    int NTREE = 10, MIN_NODE = 5;
    const float PROP_TRAIN = 0.8, PROP_OOB = 0.5;

    float *X,*y;
    int n,c;

    FastTree** trees;

    FastForest(float* X_, float* y_, int n_, int c_);
    void Build();
    float* Predict(float* rows, int n, int c);
};

class FastTree {
public:
    const int MAXN = 160, CUTOFF_DIVISOR = 10;
    FastTree(FastForest* parent);

    FastForest* parent;
    default_random_engine* rng;
    int c, n;
    float* y;  // rows subset used to train tree
    float** X;  // rows subset used to train tree
    int* idxs;
    Node* root;
    float *lT, *lT2, *cutvals;
    int *lC, *cutidxs;

    void ClearStorage_();
    void Reset(int nc);
    void CreateIdxsAndOob_();
    void Shuffle();
    void BuildNodes_();
    void CheckCutoffs(int start, int n, int nc);
    void BestCutoff_(Node* node);
    bool AllSame_(Node* node);
    static float WgtGini_(float cActs, float cActs2, float cCount, float totActs, float totActs2, float totCount);
    static float Gini_(float numAct, float numAct2, float n);
    int Shuffle_(Node* node);
    float Predict(float* arr);
};

FastForest* train_ff(float *x_, float *y, int r, int c);

