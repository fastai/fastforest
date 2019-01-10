#include "cfastforest.hpp"

using namespace std;

int train_ff(float *x_, float *y, int r, int c) {
    float (*x)[c] = (float (*)[c])x_;
    x[3][1] = 99999.0f;
    Node* n = new Node();
    n->start = 3; 
    printf("%d", n->start);

    GcTree* t = new GcTree();
    t->root_ = n;
    printf("%d", t->root_->start);
 
    random_device rd;
    default_random_engine e1(rd());
    uniform_int_distribution<int> uniform_dist(1, r);
    return uniform_dist(e1);
}

