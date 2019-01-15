#include "cfastforest.hpp"

using namespace std;

FastForest* train_ff(float *x_, float *y, int r, int c) {
    FastForest* ff = new FastForest(x_, y, r, c);
	ff->build();
    return ff;
}

template<typename T>
double stdev(T b, T e) {
    auto d = distance(b, e);
    double sum = 0.0, sqr = 0.0;
    for (T it=b; it!=e; ++it) { sum += *it; sqr += (*it) * (*it); }
    return sqrt((sqr - (sum*sum) / d) / (d-1));
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

bool Node::IsTerminal() { return bestPred == -1; }

FastForest::FastForest(float* X_, float* y_, int n_, int c_) {
    c = c_; n = n_; X = X_; y = y_;
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

float* FastForest::predict(float *rows, int n, int c) {
    auto res = new float[n];
    for (int i=0; i<n; i++) res[i]=0;

    //#pragma omp parallel for  
    for (int i = 0; i < NTREE; i++) {
        auto tree = trees[i];
        for (int j = 0; j < n; j++) {
            auto predict = tree->predict(&rows[j * c]);
            if (predict<1) printf("i %d j %d\n", i, j);
            //printf("- %f ",predict);
            //#pragma omp atomic
            res[j] += predict;
        }
        //printf("\n");
    }

    for (int j = 0; j < n; j++) {res[j] /= NTREE;}
    return res; 
}


void FastTree::clearStorage_() {
    _mm_free(lT2); _mm_free(lT); _mm_free(lC); _mm_free(cutvals); _mm_free(cutidxs);
}

void FastTree::reset(int nc) {
    for (int i = 0; i < nc; i++) { lT2[i]=lT[i]=lC[i] = 0; }
}

FastTree::FastTree(FastForest* parent) {
    this->parent = parent;
    c = parent->c;
 
    random_device rd;
    rng = new default_random_engine(rd());

	createIdxsAndOob_();  // NB: Only happens once per top-level tree

    int usedN = n > MAXN ? MAXN : n;
    int nc = usedN * sqrt(c) / CUTOFF_DIVISOR;
    if (nc < 4) nc = 4;
    //printf("nc %d\n",nc);
    lT = (float*)_mm_malloc(nc * sizeof(float), 64);
    lT2 = (float*)_mm_malloc(nc * sizeof(float), 64);
    cutvals = (float*)_mm_malloc(nc * sizeof(float), 64);
    lC = (int*)_mm_malloc(nc * sizeof(int), 64);
    cutidxs = (int*)_mm_malloc(nc * sizeof(int), 64);

	buildNodes_();
	clearStorage_();
}

void FastTree::createIdxsAndOob_() {
    // TODO: OOB
    n = 0;
    auto r = new float[parent->n];
    uniform_real_distribution<float> gen(0.0f, 1.0f);

    for (auto i = 0; i < parent->n; i++) {
        r[i] = gen(*rng);
        if (r[i] < parent->PROP_TRAIN) n++;
    }

    X = new float*[n];
    y = new float[n];
    idxs = new int[n];
    auto cur = 0;
    for (auto i = 0; i < parent->n; i++) {
        if (r[i] < parent->PROP_TRAIN)
        {
            X[cur] = &(parent->X[i*c]);
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

    for (int i = n; i > 1; i--) {
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
    root = new Node(0, n, nullptr, true);
    s.push(root);

    int i=0;
    while (!s.empty()) {
        i++;
        auto node = s.top();
        s.pop();

		bestCutoff_(node);
        if (node->IsTerminal()) {
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
    float nact=0, nact2=0;

    if (n < parent->MIN_NODE || allSame_(node)) {
        for (int i = start; i < start+n; i++) nact += y[i];
        node->value = nact/n;
        if (node->value<1)
            printf("*** n %d bp %d v %f nact %f\n", n, node->bestPred, node->value, nact);
        return;
    }

    // First: set up the cutoffs[]
    int usedN = n > MAXN ? MAXN : n;
    int nc = usedN * sqrt(c) / CUTOFF_DIVISOR;
    if (nc < 4) nc = 4;
	reset(nc);

    uniform_int_distribution<int> gen2(start, start+n-1);
    uniform_int_distribution<int> gen3(0, c-1);
    // TODO: Don't add cutoffs that aren't unique on both val and predidx
    for (int i = 0; i < nc; i++) {
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
	checkCutoffs(usedStart, usedN, nc);

    for (int r = usedStart; r < usedStart + usedN; r++) {
        nact += y[r]; nact2 += y[r]*y[r];
    }
    //printf("n %d start %d pn %d nc %d nact %d usedN %d MAXN %d c %d\n", n, start, parent->n, nc, nact, usedN, MAXN, c);
    //for (int i=0; i<nc; i++) printf("lt %f cutvals %f lc %d cutidxs %d\n", lT[i], cutvals[i], lC[i], cutidxs[i]);

    // Finally: See which cutoff has best information gain
    float crit = node->gini = wgtGini_(0, 0, 0, nact, nact2, usedN);

    int bestidx = -1;
    for (int i = 0; i < nc; i++) {
        int l=lC[i]; int r=usedN-lC[i];
        int min_size = max(int(usedN*0.05), parent->MIN_NODE/3);
        if (l<min_size || r<min_size) {
            //printf("XXX l %d r %d ci %d cu %f \n", l, r, cutidxs[i], cutvals[i]);
            continue;
        }
        auto g = wgtGini_(lT[i], lT2[i], l, nact, nact2, usedN);
        //printf("   i %d g %f crit %f bestidx %d l %d r %d lt %f\n", i, g, crit, bestidx, l, r, lT[i]);
        if (g <= crit) continue;
        //printf("***i %d g %f crit %f bestidx %d l %d r %d lt %f\n", i, g, crit, bestidx, l, r, lT[i]);
        crit = g; 
        bestidx = i;
    }

    if (bestidx >= 0) { // Did we make an improvement?
        node->bestPred = cutidxs[bestidx];
        node->cutoff = cutvals[bestidx];
    }

    // Only approximate if sampled:
    node->value = nact / usedN;
    if (node->value<1)
        printf("*** n %d bp %d v %f nact %f\n", n, node->bestPred, node->value, nact);
    //printf("lt %f lc %d n %d crit %f gini %f bestidx %d bestPred %d cutoff %f value %f\n",
            //lT[bestidx], lC[bestidx], usedN, crit, node->gini, bestidx, node->bestPred, node->cutoff, node->value);
}

void do_cutoffchk(int nc, float* pred, float act, int* cutidxs, float* cutvals, float* lT, float* lT2, int* lC) {
    //#pragma ivdep
    for (int i = 0; i < nc; i++) {
        if (pred[cutidxs[i]] >= cutvals[i]) continue;
        lT[i] += act;
        lT2[i] += act*act;
        lC[i]++;
    }
}

void FastTree::checkCutoffs(int start, int n, int nc) {
    //printf("b start %d n %d\n", start, n);
    for (int r = start; r < start + n; r++) {
        do_cutoffchk(nc, X[r], y[r], cutidxs, cutvals, lT, lT2, lC);
    }
}

bool FastTree::allSame_(Node *node) {
    int n = node->n;
    if (n > MAXN) n = MAXN;
    int start = node->start;
    int first = y[start];

    for (int i = start + 1; i < start + n; i++)
        if (y[i] != first) return false;

    return true;
}

float FastTree::wgtGini_(float cActs, float cActs2, float cCount, float totActs, float totActs2, float totCount) {
    float l = gini_(cActs, cActs2, cCount);
    float r = gini_(totActs - cActs, totActs2 - cActs2, totCount - cCount);
    float lProp = cCount/totCount;
    float res = 0.0; 

    // avoids NaNs by checking size of segment
    if (lProp > 0) res += l*lProp;
    if (lProp < 1) res += r*(1 - lProp);
    return res;
}

float FastTree::gini_(float numAct, float numAct2, float n) {
    return -sqrt((numAct2 - (numAct*numAct)/n) / (n-1));
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

float FastTree::predict(float *arr) {
    Node* node = root;
    int i=0;
    while (node->bestPred >= 0 && i++<10000) {
        node = (arr[node->bestPred] < node->cutoff) ? node->left : node->right;
    }
    if (i>1000) printf("************---\n");
    return node->value;
}
 
