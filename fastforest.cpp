#include "fastforest.hpp"
#include <mm_malloc.h>
#include <iostream>

FastForest* trainFF(Mat X, Vec y) {
    auto ff = new FastForest(X, y);
    ff->build();
    return ff;
}

Node::Node(int start, int nrows, Node* parent) {
    bestPred = -1;
    value=gini=cutoff=0.0;
    this->start = start;
    this->nrows = nrows;
    if (parent == nullptr) return;
}

bool Node::isTerminal() { return bestPred == -1; }

FastForest::FastForest(Mat X, Vec y) {
    this->nrows = X.rows(); this->ncols = X.cols(); this->X = X; this->y = y;
}

void FastForest::build() {
    // TODO: Multiclass classification
    trees = new FastTree*[NTREE];
    #pragma omp parallel for
    for (int i=0; i<NTREE; i++) {
        trees[i] = new FastTree(this);
    }
}

Vec FastForest::predict(Mat X) {
    int nrows = X.rows();
    int ncols = X.cols();
    //float* Xd = X.data();
    Vec res(nrows);
    for (int i=0; i<nrows; i++) res[i]=0;

    //#pragma omp parallel for
    for (int i = 0; i < NTREE; i++) {
        auto tree = trees[i];
        for (int j = 0; j < nrows; j++) {
            auto predict = tree->predict(X.row(j));
            if (predict<1) printf("i %d j %d\n", i, j);
            //printf("- %f ",predict);
            //#pragma omp atomic
            res(j) += predict;
        }
    }

    res /= NTREE;
    return res;
}

FastTree::FastTree(FastForest* parent) {
    this->parent = parent;
    this->ncols = parent->ncols;

    random_device rd;
    rng = new default_random_engine(rd());

    createIdxsAndOob_();  // NB: Only happens once per top-level tree

    int usedN = nrows > MAXN ? MAXN : nrows; // TODO: is this just min(ncols, MAXN)?
    int ncandidates = usedN * sqrt(ncols) / CUTOFF_DIVISOR;
    if (ncandidates < 4) ncandidates = 4;
    //printf("nc %d\n", ncandidates);

        buildNodes_();
}

void FastTree::createIdxsAndOob_() {
    // TODO: OOB
    nrows = 0;
    auto r = new float[parent->nrows];
    uniform_real_distribution<float> rowgen(0.0f, 1.0f);

    for (auto i = 0; i < parent->nrows; i++) {
        r[i] = rowgen(*rng);
        if (r[i] < parent->PROP_TRAIN) nrows++;
    }

    X = new float*[nrows];
    y = new float[nrows];
    idxs = new int[nrows];
    float* Xd = parent->X.data();
    float* yd = parent->y.data();
    auto cur = 0;
    for (auto i = 0; i < parent->nrows; i++) {
        if (r[i] < parent->PROP_TRAIN)
        {
            X[cur] = &(Xd[i*ncols]);
            y[cur] = yd[i];
            idxs[cur] = i;
            cur++;
        }
    }
    // test this worked

        shuffle();
}

void FastTree::shuffle() {
    uniform_real_distribution<float> rowgen(0.0f, 1.0f);

    for (int i = nrows; i > 1; i--) {
        // Pick random element to swap.
        auto j = (int)(rowgen(*rng) * i); // 0 <= j <= i-1
        // Swap
        float* tmp = X[j];
        X[j] = X[i - 1];
        X[i - 1] = tmp;

        float tmp2 = y[j];
        y[j] = y[i - 1];
        y[i - 1] = tmp2;
    }
}

void FastTree::buildNodes_() {
    stack<Node*> s;
    // TODO: clean up Node memory on destruction
    root = new Node(0, nrows, nullptr);
    s.push(root);

    int i=0;
    while (!s.empty()) {
        i++;
        auto node = s.top();
        s.pop();

        bestCutoff_(node);
        if (node->isTerminal()) {
            //printf("---T: i %d v %f\n", i, node->value);
            continue;
        }

        int leftn = shuffle_(node);
        int rightn = node->nrows - leftn;
        if (leftn == 0 || rightn == 0)
            printf("i %d l %d r %d\n", i, leftn, rightn);

        node->left = new Node(node->start, leftn, node);
        node->right = new Node(node->start+leftn, rightn, node);
        s.push(node->right);
        s.push(node->left);
    }
}

void FastTree::bestCutoff_(Node *node) {
    int n = node->nrows;
    auto start = node->start;
    float sumTarget=0, sumSqrTarget=0;

    if (n < parent->MIN_NODE || allSame_(node)) {
        for (int i = start; i < start+n; i++) sumTarget += y[i];
        node->value = sumTarget/n;
        if (node->value<1)
            printf("*** n %d bp %d v %f sumTarget %f\n", n, node->bestPred, node->value, sumTarget);
        return;
    }

    // First: set up the cutoffs[]
    int usedN = n > MAXN ? MAXN : n;
    int ncandidates = usedN * sqrt(ncols) / CUTOFF_DIVISOR;
    if (ncandidates < 4) ncandidates = 4;

    CandidateInfo candInfo[ncandidates]; // Only works in GCC I think but that's ok as we rely on OpenMP in GCC also

    uniform_int_distribution<int> colgen(0, ncols-1);
    uniform_int_distribution<int> splitgen(start, start+n-1);
    // TODO: Don't add cutoffs that aren't unique on both val and predidx
    for (int i = 0; i < ncandidates; i++) {
        auto colidx = colgen(*rng);
        candInfo[i].cutcol = colidx;
        candInfo[i].cutval = X[splitgen(*rng)][colidx];
    }

    //printf("a start %d n %d usedN %d\n", start, n, usedN);
    int usedStart = start;
    if (n>usedN) {
        uniform_int_distribution<int> startgen(0, n-usedN-1);
        usedStart += startgen(*rng);
    }
    checkCutoffs(usedStart, usedN, candInfo, ncandidates);

    for (int r = usedStart; r < usedStart + usedN; r++) {
        sumTarget += y[r]; sumSqrTarget += y[r]*y[r];
    }
    //printf("n %d start %d pn %d ncandidates %d sumTarget %d usedN %d MAXN %d c %d\n", n, start, parent->n, ncandidates, sumTarget, usedN, MAXN, c);
    //for (int i=0; i<ncandidates; i++) printf("lt %f cutval %f lc %d cutcol %d\n", candInfo[i].leftTarget, candInfo[i].cutval, candInfo[i].leftCount, candInfo[i].cutcol);

    // Finally: See which cutoff has best information gain
    float crit = node->gini = wgtGini_(0, 0, 0, sumTarget, sumSqrTarget, usedN);

    int bestidx = -1;
    for (int i = 0; i < ncandidates; i++) {
        int l=candInfo[i].leftCount; int r=usedN-candInfo[i].leftCount;
        int min_size = max(int(usedN*0.05), parent->MIN_NODE/3);
        if (l<min_size || r<min_size) {
            //printf("XXX l %d r %d ci %d cu %f \n", l, r, cutcol[i], cutval[i]);
            continue;
        }
        auto g = wgtGini_(candInfo[i].leftTarget, candInfo[i].leftSqrTarget, l, sumTarget, sumSqrTarget, usedN);
        //printf("   i %d g %f crit %f bestidx %d l %d r %d lt %f\n", i, g, crit, bestidx, l, r, candInfo[i].leftTarget);
        if (g <= crit) continue;
        //printf("***i %d g %f crit %f bestidx %d l %d r %d lt %f\n", i, g, crit, bestidx, l, r, candInfo[i].leftTarget);
        crit = g;
        bestidx = i;
    }

    if (bestidx >= 0) { // Did we make an improvement?
        node->bestPred = candInfo[bestidx].cutcol;
        node->cutoff = candInfo[bestidx].cutval;
    }

    // Only approximate if sampled:
    node->value = sumTarget / usedN;
    if (node->value<1)
        printf("*** n %d bp %d v %f sumTarget %f\n", n, node->bestPred, node->value, sumTarget);
    //printf("lt %f lc %d n %d crit %f gini %f bestidx %d bestPred %d cutoff %f value %f\n",
            //candInfo[bestidx].leftTarget, candInfo[bestidx].leftCount, usedN, crit, node->gini, bestidx, node->bestPred, node->cutoff, node->value);
}

// TODO: is pred actually x (feature value)?
void do_cutoffchk(int ncandidates, float const * pred, float target, CandidateInfo *candInfo) {
    //#pragma ivdep
    for (int i = 0; i < ncandidates; i++) {
        if (pred[candInfo[i].cutcol] >= candInfo[i].cutval) continue;
        candInfo[i].leftTarget += target;
        candInfo[i].leftSqrTarget += target*target;
        candInfo[i].leftCount++;
    }
}

void FastTree::checkCutoffs(int start, int n, CandidateInfo *candInfo, int ncandidates) {
    //printf("b start %d n %d\n", start, n);
    for (int r = start; r < start + n; r++) {
        do_cutoffchk(ncandidates, X[r], y[r], candInfo);
    }
}

bool FastTree::allSame_(Node *node) {
    int n = node->nrows;
    if (n > MAXN) n = MAXN;
    int start = node->start;
    int first = y[start]; // TODO: this has to be float

    for (int i = start + 1; i < start + n; i++)
        if (y[i] != first) return false;

    return true;
}

float FastTree::wgtGini_(float leftTarget, float leftSqrTarget, float leftCount, float sumTarget, float sumSqrTarget, float totCount) {
    float l = loss_(leftTarget, leftSqrTarget, leftCount);
    float r = loss_(sumTarget - leftTarget, sumSqrTarget - leftSqrTarget, totCount - leftCount);
    float lprop = leftCount/totCount;
    float result = 0.0;

    // avoids NaNs by checking size of segment
    if (lprop > 0) result += l*lprop;
    if (lprop < 1) result += r*(1 - lprop);
    return result;
}

int FastTree::shuffle_(Node *node) {
    int start = node->start;
    int n = node->nrows, p = node->bestPred;
    float cutoff = node->cutoff;

    int end = start + n;
    int i;
    for (i = start; i < end; i++) {
        float* pred = X[i];
        if (pred[p] < cutoff) continue;

        int e = end-1;
        float* tmp = X[e];
        X[e] = pred;
        X[i] = tmp;

        float tmp2 = y[e];
        y[e] = y[i];
        y[i] = tmp2;
        i--;
        end--;
    }
    return i-start;
}

float FastTree::predict(Vec X) {
    Node* node = root;
    const int FAILSAFE = 1000; // TODO: remove when done debugging
        int i=0;                   // TODO: remove when done debugging
    while ( node->bestPred >= 0 && i<FAILSAFE ) {
        node = (X(node->bestPred) < node->cutoff) ? node->left : node->right;
        i++;
    }
    if (i>FAILSAFE) printf("************---\n");
    return node->value;
}

FastTree* FastForest::getTree(int i) {
    return trees[i];
}

// U t i l i t y  F u n c t i o n s

template<typename T>
double stdev(T b, T e) {
    auto d = distance(b, e);
    double sum = 0.0, sqr = 0.0;
    for (T it=b; it!=e; ++it) { sum += *it; sqr += (*it) * (*it); }
    return sqrt((sqr - (sum*sum) / d) / (d-1));
}

/*
 * Compute loss as negative of standard deviation analogous to alternate
 * formulation of stddev: s = sqrt((sum(x^2) - n(x_bar)^2)/(n-1)).
 */
float loss_(float sumTarget, float sumSqrTarget, float n) {
    return -sqrt((sumSqrTarget - (sumTarget*sumTarget)/n) / (n-1));
}

