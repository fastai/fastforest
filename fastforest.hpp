#pragma once

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
#include <Eigen/Dense>

using namespace std;
using Mat = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec = Eigen::ArrayXf;

/** This class represents both a decision node and a leaf (when bestPred==-1) in a decision tree. */
class Node {
public:
    struct Node *left, *right;
    int start;    // observation index into tree's data rows where this node's samples start
    int nrows;    // number of rows beyond start index of tree's data rows associated with this node
    int cutcol;   // which column/variable/feature to test if decision node; -1 indicates leaf
    float cutval; // split value for cutcol
    float value;  // prediction value (set even for internal decision nodes)
    float gini;   // uncertainty/impurity of this node

    Node(int start, int nrows, Node *parent);
    bool isTerminal();
};

class FastTree;

class FastForest {
public:
    int NTREE = 5, MIN_NODE = 25;
    const float sampleFraction = 0.8; // fraction of rows to extract from training set to train each tree
    const float PROP_OOB = 0.5;

    Mat X; // all training feature vectors, one row per observation
    Vec y; // all training target values, one per observation
    int nrows, ncols;

    FastTree **trees;

    FastForest(Mat X, Vec y);
    void build();
    FastTree *getTree(int i);
    Vec predict(Mat X);
};

struct CandidateInfo {
    float leftTarget;    // sum of all target values for observations where X[cutcol] < cutval
    float leftSqrTarget; // sum of square of target values to left of cutval
    int leftCount;       // how many observations fall to the left of cutval
    int cutcol;          // which feature/column this candidate tests
    float cutval;        // which split value this candidate tests

    CandidateInfo() { leftSqrTarget = leftTarget = leftCount = 0; }
};

class FastTree {
public:
    const int MAXN = 160, CUTOFF_DIVISOR = 10;

    FastForest* parent;
    default_random_engine* rng; // single random num generator used by code building this tree
    int nrows, ncols;
    float *y;  // subset size nrows of forest's X rows used to train this tree
    float **X; // subset size nrows of forest's y rows used to train this tree (array of ptrs to float)
    int *idxs; // nrows indexes into forest's X/y training observations
    Node *root;

    FastTree(FastForest* parent);
    float predict(Vec X);
    void shuffle();

protected:
    void createIdxsAndOob(float *Xall, float *yall);
    void buildNodes();
    void bestCutoff(Node *node);
    void checkCutoffs(int start, int n, CandidateInfo *candInfo, int ncandidates);
    bool allSame(Node *node);
    static float wgtGini(float leftTarget, float leftSqrTarget, float leftCount,
                         float sumTarget, float sumSqrTarget, float totCount);
    int partition(Node *node);
};

FastForest *trainFF(Mat X, Vec y);

template <typename T> double stdev(T b, T e);
float loss_(float sumTarget, float sumSqrTarget, float n);

