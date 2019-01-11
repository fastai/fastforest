#include "cfastforest.hpp"

using namespace std;

int train_ff(float *x_, float *y, int r, int c) {
    Node* n = new Node();
    n->start = 3; 
    printf("%d", n->start);

    FastForest* ff = new FastForest(x_, y, r, c);
    float (*x)[c] = (float (*)[c])ff->X;
    x[3][1] = 99999;
    printf("%f", x[3][1]);
    //t->root_ = n;
    //printf("%d", t->root_->start);
 
    random_device rd;
    default_random_engine e1(rd());
    uniform_int_distribution<int> uniform_dist(1, r);
    return uniform_dist(e1);
}

FastForest::FastForest(float* X_, float* y_, int n, int c) {
    C = c; N = n; X = X_; y = y_;
}
void FastForest::build() {
    trees = new FastTree(this);
}

FastTree::FastTree(FastForest* parent) {
    parent_ = parent;
    //CreateIdxsAndOob_();  // NB: Only happens once per top-level tree
    c_ = parent->C;

    int usedN = n_ > MAXN ? MAXN : n_;
    int nc = usedN * c_ / CUTOFF_DIVISOR;
    if (nc < 4) nc = 4;
    //BuildNodes_();
    //ClearStorage_();
}

