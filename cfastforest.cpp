#include <mm_malloc.h>
#include "cfastforest.hpp"

using namespace std;

FastForest* trainFF(float *X, float *y, int nrows, int ncols) {
    auto ff = new FastForest(X, y, nrows, ncols);
	ff->build();
    return ff;
}

Node::Node(int start, int n, Node* parent, bool isLeft) {
    bestPred = -1;
    value=gini=cutoff=0.0;
    this->start = start;
    this->n = n;
    this->isLeft = isLeft;
    if (parent == nullptr) return;

    if (isLeft) parent->left = this;
    else parent->right = this;
}

bool Node::isTerminal() { return bestPred == -1; }

FastForest::FastForest(float* X, float* y, int nrows, int ncols) {
	this->nrows = nrows; this->ncols = ncols; this->X = X; this->y = y;
}

void FastForest::build() {
    // TODO: Multiple trees
    // TODO: Multiclass classification
    trees = new FastTree*[NTREE];
    #pragma omp parallel for
    for (int i=0; i<NTREE; i++) {
        trees[i] = new FastTree(this);
    }
}

float* FastForest::predict(float *X, int nrows, int ncols) {
    auto res = new float[nrows];
    for (int i=0; i<nrows; i++) res[i]=0;

    //#pragma omp parallel for  
    for (int i = 0; i < NTREE; i++) {
        auto tree = trees[i];
        for (int j = 0; j < nrows; j++) {
            auto predict = tree->predict(&X[j * ncols]);
            if (predict<1) printf("i %d j %d\n", i, j);
            //printf("- %f ",predict);
            //#pragma omp atomic
            res[j] += predict;
        }
        //printf("\nrows");
    }

    for (int j = 0; j < nrows; j++) {res[j] /= NTREE;}
    return res; 
}


void FastTree::clearStorage_() {
    _mm_free(leftSqrTarget); _mm_free(leftTarget); _mm_free(leftCount); _mm_free(cutvals); _mm_free(cutidxs);
}

void FastTree::reset(int ncandidates) {
    for (int i = 0; i < ncandidates; i++) { leftSqrTarget[i]=leftTarget[i]=leftCount[i] = 0; }
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
    //printf("nc %d\n",nc);
    leftTarget = (float*)_mm_malloc(ncandidates * sizeof(float), 64);
    leftSqrTarget = (float*)_mm_malloc(ncandidates * sizeof(float), 64);
    cutvals = (float*)_mm_malloc(ncandidates * sizeof(float), 64);
    leftCount = (int*)_mm_malloc(ncandidates * sizeof(int), 64);
    cutidxs = (int*)_mm_malloc(ncandidates * sizeof(int), 64);

	buildNodes_();
	clearStorage_();
}

void FastTree::createIdxsAndOob_() {
    // TODO: OOB
    nrows = 0;
    auto r = new float[parent->nrows];
    uniform_real_distribution<float> gen(0.0f, 1.0f);

    for (auto i = 0; i < parent->nrows; i++) {
        r[i] = gen(*rng);
        if (r[i] < parent->PROP_TRAIN) nrows++;
    }

    X = new float*[nrows];
    y = new float[nrows];
    idxs = new int[nrows];
    auto cur = 0;
    for (auto i = 0; i < parent->nrows; i++) {
        if (r[i] < parent->PROP_TRAIN)
        {
            X[cur] = &(parent->X[i*ncols]);
            y[cur] = parent->y[i];
            idxs[cur] = i;
            cur++;
        }
    }
    // test this worked

	shuffle();
}

void FastTree::shuffle() {
    uniform_real_distribution<float> gen(0.0f, 1.0f);

    for (int i = nrows; i > 1; i--) {
        // Pick random element to swap.
        auto j = (int)(gen(*rng) * i); // 0 <= j <= i-1
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
    root = new Node(0, nrows, nullptr, true);
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
        int rightn = node->n - leftn;
        if (leftn == 0 || rightn == 0)
            printf("i %d l %d r %d\n", i, leftn, rightn); 

        s.push(new Node(node->start+leftn, rightn, node, false));
        s.push(new Node(node->start, leftn, node, true));
    }
}

void FastTree::bestCutoff_(Node *node) {
    int n = node->n;
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
	reset(ncandidates);

    uniform_int_distribution<int> gen2(start, start+n-1);
    uniform_int_distribution<int> gen3(0, ncols-1);
    // TODO: Don't add cutoffs that aren't unique on both val and predidx
    for (int i = 0; i < ncandidates; i++) {
        auto predidx = gen3(*rng);
        cutidxs[i] = predidx;
        cutvals[i] = X[gen2(*rng)][predidx];
    }

    //printf("a start %d n %d usedN %d\n", start, n, usedN);
    int usedStart = start;
    if (n>usedN) {
        uniform_int_distribution<int> gen4(0, n-usedN-1);
        usedStart += gen4(*rng);
    }
	checkCutoffs(usedStart, usedN, ncandidates);

    for (int r = usedStart; r < usedStart + usedN; r++) {
        sumTarget += y[r]; sumSqrTarget += y[r]*y[r];
    }
    //printf("n %d start %d pn %d ncandidates %d sumTarget %d usedN %d MAXN %d c %d\n", n, start, parent->n, ncandidates, sumTarget, usedN, MAXN, c);
    //for (int i=0; i<ncandidates; i++) printf("lt %f cutvals %f lc %d cutidxs %d\n", leftTarget[i], cutvals[i], leftCount[i], cutidxs[i]);

    // Finally: See which cutoff has best information gain
    float crit = node->gini = wgtGini_(0, 0, 0, sumTarget, sumSqrTarget, usedN);

    int bestidx = -1;
    for (int i = 0; i < ncandidates; i++) {
        int l=leftCount[i]; int r=usedN-leftCount[i];
        int min_size = max(int(usedN*0.05), parent->MIN_NODE/3);
        if (l<min_size || r<min_size) {
            //printf("XXX l %d r %d ci %d cu %f \n", l, r, cutidxs[i], cutvals[i]);
            continue;
        }
        auto g = wgtGini_(leftTarget[i], leftSqrTarget[i], l, sumTarget, sumSqrTarget, usedN);
        //printf("   i %d g %f crit %f bestidx %d l %d r %d lt %f\n", i, g, crit, bestidx, l, r, leftTarget[i]);
        if (g <= crit) continue;
        //printf("***i %d g %f crit %f bestidx %d l %d r %d lt %f\n", i, g, crit, bestidx, l, r, leftTarget[i]);
        crit = g; 
        bestidx = i;
    }

    if (bestidx >= 0) { // Did we make an improvement?
        node->bestPred = cutidxs[bestidx];
        node->cutoff = cutvals[bestidx];
    }

    // Only approximate if sampled:
    node->value = sumTarget / usedN;
    if (node->value<1)
        printf("*** n %d bp %d v %f sumTarget %f\n", n, node->bestPred, node->value, sumTarget);
    //printf("lt %f lc %d n %d crit %f gini %f bestidx %d bestPred %d cutoff %f value %f\n",
            //leftTarget[bestidx], leftCount[bestidx], usedN, crit, node->gini, bestidx, node->bestPred, node->cutoff, node->value);
}

void do_cutoffchk(int ncandidates, float const * pred, float act, int const *cutidxs, float const *cutvals, float* leftTarget, float* leftSqrTarget, int* leftCount) {
    //#pragma ivdep
    for (int i = 0; i < ncandidates; i++) {
        if (pred[cutidxs[i]] >= cutvals[i]) continue;
        leftTarget[i] += act;
        leftSqrTarget[i] += act*act;
        leftCount[i]++;
    }
}

void FastTree::checkCutoffs(int start, int n, int ncandidates) {
    //printf("b start %d n %d\n", start, n);
    for (int r = start; r < start + n; r++) {
        do_cutoffchk(ncandidates, X[r], y[r], cutidxs, cutvals, leftTarget, leftSqrTarget, leftCount);
    }
}

bool FastTree::allSame_(Node *node) {
    int n = node->n;
    if (n > MAXN) n = MAXN;
    int start = node->start;
    int first = y[start]; // TODO: this has to be float

    for (int i = start + 1; i < start + n; i++)
        if (y[i] != first) return false;

    return true;
}

float FastTree::wgtGini_(float leftTarget, float leftSqrTarget, float leftCount, float sumTarget, float sumSqrTarget, float totCount) {
    float l = gini_(leftTarget, leftSqrTarget, leftCount);
    float r = gini_(sumTarget - leftTarget, sumSqrTarget - leftSqrTarget, totCount - leftCount);
    float lProp = leftCount/totCount;
    float result = 0.0;

    // avoids NaNs by checking size of segment
    if (lProp > 0) result += l*lProp;
    if (lProp < 1) result += r*(1 - lProp);
    return result;
}

int FastTree::shuffle_(Node *node) {
    int start = node->start;
    int n = node->n, p = node->bestPred;
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

float FastTree::predict(float const *X) {
    Node* node = root;
    int i=0;
    while (node->bestPred >= 0 && i++<10000) {
        node = (X[node->bestPred] < node->cutoff) ? node->left : node->right;
    }
    if (i>1000) printf("************---\n");
    return node->value;
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
float gini_(float sumTarget, float sumSqrTarget, float n) {
	return -sqrt((sumSqrTarget - (sumTarget*sumTarget)/n) / (n-1));
}
