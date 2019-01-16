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
    int start, nrows, bestPred;
    float cutoff, value, gini;

    Node(int start, int nrows, Node *parent, bool isLeft);
    bool isTerminal();
};

class FastTree;

class FastForest {
public:
    int NTREE = 10, MIN_NODE = 5;
    const float PROP_TRAIN = 0.8, PROP_OOB = 0.5;

    float *X,*y;
    int nrows, ncols;

    FastTree** trees;

    FastForest(float *X, float* y, int nrows, int ncols);
    void build();
    float* predict(float *X, int nrows, int ncols);
};

struct CandidateInfo {
	float leftTarget, leftSqrTarget, cutvals;
	int leftCount, cutidxs;
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
	//_mm_free(leftSqrTarget); _mm_free(leftTarget); _mm_free(leftCount); _mm_free(cutvals); _mm_free(cutidxs);
//    float *leftTarget, *leftSqrTarget, *cutvals;
//    int *leftCount, *cutidxs;
	CandidateInfo *candInfo;

    void clearStorage_();
    void reset(int ncandidates);
    void createIdxsAndOob_();
    void shuffle();
    void buildNodes_();
    void checkCutoffs(int start, int n, int ncandidates);
    void bestCutoff_(Node *node);
    bool allSame_(Node *node);
    static float wgtGini_(float leftTarget, float leftSqrTarget, float leftCount, float sumTarget, float sumSqrTarget, float totCount);
    int shuffle_(Node *node);
    float predict(float const *X);
};

FastForest* trainFF(float *X, float *y, int nrows, int ncols);

template <typename T> double stdev(T b, T e);
float gini_(float sumTarget, float sumSqrTarget, float n);