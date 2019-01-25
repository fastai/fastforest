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
    bool isLeft;
    int start;    // observation index into tree's data rows where this node's samples start
    int nrows;    // observation index into tree's data rows where this node's samples stop (inclusive)
    int bestPred; // which column/variable/feature to test if decision node; -1 indicates leaf
    float cutoff;
    float value;
    float gini;

    Node(int start, int nrows, Node *parent, bool isLeft);
    bool isTerminal();
};

class FastTree;

class FastForest {
public:
    int NTREE = 5, MIN_NODE = 25;
    const float PROP_TRAIN = 0.8, PROP_OOB = 0.5;

    Mat X;
    Vec y;
    int nrows, ncols;

    FastTree** trees;

    FastForest(Mat X, Vec y);
    void build();
    FastTree* getTree(int i);
    Vec predict(Mat X);
};

struct CandidateInfo {
    float leftTarget, leftSqrTarget, cutval;
    int leftCount, cutcol;

    // might be tiny bit slower to init with ctor vs iterating over array but
    // we don't need reset() function this way.
    CandidateInfo() { leftSqrTarget = leftTarget = leftCount = 0; }
};

class FastTree {
public:
    const int MAXN = 160, CUTOFF_DIVISOR = 10;
    FastTree(FastForest* parent);

    FastForest* parent;
    default_random_engine* rng;
    int nrows, ncols;
    float* y;   // rows subset used to train tree
    float** X;  // rows subset used to train tree
    int* idxs;
    Node* root;

    void createIdxsAndOob_();
    void shuffle();
    void buildNodes_();
    void checkCutoffs(int start, int n, CandidateInfo *candInfo, int ncandidates);
    void bestCutoff_(Node *node);
    bool allSame_(Node *node);
    static float wgtGini_(float leftTarget, float leftSqrTarget, float leftCount, float sumTarget, float sumSqrTarget, float totCount);
    int partition_(Node *node);
    float predict(Vec X);
};

FastForest* trainFF(Mat X, Vec y);

template <typename T> double stdev(T b, T e);
float loss_(float sumTarget, float sumSqrTarget, float n);

