#include "tnc.h"
#include "tron.h"
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

struct Xw {
    int index;
    double value;
};

typedef int (*compfn)(const void *, const void *);
int comparedouble(struct Xw *a, struct Xw *b) {

    if(a->value < b->value)
        return -1;
    if(a->value > b->value)
        return 1;
    if(a->index < b->index)
        return -1;
    if(a->index > b->index)
        return 1;
    return 0;
}

static void print_string_l2r_l2(const char *s) {

	fputs(s,stdout);
	fflush(stdout);
}

l2r_l2_primal_fun::l2r_l2_primal_fun(const problem *prob, double weight) {

    set_print_string(print_string_l2r_l2);
    info("C_r = %f ", weight);
    this->prob = prob;
    int l = prob->l;
    start = prob->start;
    count = prob->count;

    int i;
    Xw_sorted = new Xw[l];
    for(i = 0; i < l; i++) {

        Xw_sorted[i].index = i;
    }
    
    z = new double[l];
    I = new int[l];
    alpha = new double[l];
    beta = new double[l];
    z_r = new double[l];
    Xs = new double[l];
    cache = new int*[l];
    for(i = 0; i < l; i++) {

        cache[i] = new int[1]();
    }

    this->C_e = prob->C_e;
    this->C_r = weight;
}

l2r_l2_primal_fun::~l2r_l2_primal_fun() {

    info("rank C in input is: %f\n", this->C_r);
    info("I'm quiting with total violation: %f\n", totalViolation);
    delete[] Xw_sorted;
    delete[] I;
    delete[] z;
    delete[] alpha;
    delete[] beta;
    delete[] z_r;
    delete[] Xs;

    int i;
    for(i = 0; i < prob->l; i++) {
        delete[] cache[i];
    }
    delete[] cache;
}

double l2r_l2_primal_fun::fun(double *w) {

    int i, j;
    double f = 0, f_delta = 0;
    double *y = prob->y;
    int w_size = get_nr_variable();

    Xv(w, z);
    memcpy(z_r, z, (size_t)prob->l * sizeof(double));

    for(i = 0; i < w_size; i++) {

        f += w[i] * w[i];
    }
    f /= 2.0;

    for(j = 0; j < prob->l; j++) {

        z[j] = y[j] * z[j];
        double d = 1 - z[j];
        if(d > 0) {

            f += C_e[j] * d * d;
        }
    }

    for(i = start[0]; i < start[1]; i++) {

        z_r[i] -= 0.5;
    }
    for(i = start[1]; i < start[2]; i++) {

        z_r[i] += 0.5;
    }
    for(i = start[2]; i < start[3]; i++) {

        z_r[i] -= 0.5;
    }
    for(i = start[3]; i < prob->l; i++) {

        z_r[i] += 0.5;
    }

    totalViolation = 0;
    for(i = start[0]; i < start[1]; i++) {
        for(j = start[1]; j < start[2]; j++) {

            double d = z_r[j] - z_r[i];
            if(d > 0) {

                f_delta += d * d;
                totalViolation++;
            }
        }
    }

    for(i = start[2]; i < start[3]; i++) {
        for(j = start[3]; j < prob->l; j++) {

            double d = z_r[j] - z_r[i];
            if(d > 0) {

                f_delta += d * d;
                totalViolation++;
            }
        }
    }

    if(totalViolation > 0) {
        
        f += (1/totalViolation) * C_r * f_delta;
    }
    info("total Violation: %f\n", totalViolation);
    return(f);
}

void l2r_l2_primal_fun::pairGrad(double *w, int i, int j, double *g) {

    double *y = prob->y;
    int w_size = get_nr_variable();

    sizeI = 0;
    for(i = 0; i < prob->l; i++) {

        if(z[i] < 1) {
            z[sizeI] = C_e[i] * y[i] * (z[i] - 1);
            I[sizeI] = i;
            sizeI++;
        }
    }

    subXTv(z, g);

    Xv(w, z_r);
    for(i = start[0]; i < start[1]; i++) {

        z_r[i] -= 0.5;
        Xw_sorted[i].value = z_r[i];
    }
    for(i = start[1]; i < start[2]; i++) {

        z_r[i] += 0.5;
        Xw_sorted[i].value = z_r[i];
    }
    for(i = start[2]; i < start[3]; i++) {

        z_r[i] -= 0.5;
        Xw_sorted[i].value = z_r[i];
    }
    for(i = start[3]; i < prob->l; i++) {

        z_r[i] += 0.5;
        Xw_sorted[i].value = z_r[i];
    }

    for(i = 0; i < 4; i++) {
        qsort((void *)&Xw_sorted[start[i]],
                (size_t)count[i],
                sizeof(struct Xw),
                (compfn)comparedouble);
    }

    findab(alpha, beta, Xw_sorted, z_r, cache);
    totalViolation = 0;
    for(i = 0; i < prob->l; i++) {

        totalViolation += alpha[i];
    }
    totalViolation /= 2.0;

    double *g_delta = new double[prob->l]();
    CXT_alpha_zr_beta(z_r, beta, g_delta);
    
    if(totalViolation > 0) {

        for(i = 0; i < w_size; i++) {

            g[i] = w[i] + 2 * (g[i] + (1/totalViolation) * C_r * g_delta[i]);
        }
    } else {

        for(i = 0; i < w_size; i++) {

            g[i] = w[i] + 2 * g[i];
        }
    }

    delete[] g_delta;
}

void l2r_l2_primal_fun::CXT_alpha_zr_beta(
        const double *z_rr, const double *beta1, double *Xg) {

    // Xg are assumed to be zeros.
    int i;
    feature_node **x = prob->x;

    if(totalViolation < 1) return;

    for(i = 0; i < prob->l; i++) {

        if(alpha[i] == 0) continue;
        feature_node *xx = x[i];
        while(xx->index != -1) {

            Xg[xx->index-1] += (alpha[i] * z_rr[i] - beta[i]) * xx->value;
            xx++;
        }
    }
}

void l2r_l2_primal_fun::Hv(double *g, double *Hs) {

    int i, j;
    int w_size = get_nr_variable();
    double *wa = new double[sizeI];

    subXv(g, wa);
    for(i = 0; i < sizeI; i++) {

        wa[i] = C_e[I[i]] * wa[i];
    }
    subXTv(wa, Hs);

    Xv(g, Xs);
    double *Hs_delta = new double[w_size]();
    double *gamma = new double[prob->l]();
    feature_node **x = prob->x;
    for(i = 0; i < prob->l; i++) {
        for(j = 0; j < cache[i][0]; j++) {

            gamma[i] += Xs[cache[i][j+1]];
        }
    }
    for(i = 0; i < prob->l; i++) {

        if(totalViolation < 1) break;
        if(alpha[i] == 0) continue;
        feature_node *xx = x[i];
        while(xx->index != -1) {
            Hs_delta[xx->index-1] += (alpha[i] * Xs[i] - gamma[i]) *xx->value;
            xx++;
        }
    }

    if(totalViolation > 0) {

        for(i = 0; i < w_size; i++) {

            Hs[i] = g[i] + 2 * (Hs[i] + (1/totalViolation) * C_r * Hs_delta[i]);
        }
    } else {

        for(i = 0; i < w_size; i++) {

            Hs[i] = g[i] + 2 * Hs[i];
        }
    }


    delete[] wa;
    delete[] Hs_delta;
    delete[] gamma;
}


void l2r_l2_primal_fun::findab(double* al, double* be,
        struct Xw* Xw_sorted, const double* orig, int **table) {

    int l = prob->l;
    int i, j, k;
    for(i = 0; i < l; i++) {

        delete[] table[i];
    }


    for(i = 0; i < start[1]; i++) {// high positive

        al[i] = 0;
        be[i] = 0;
        for(j = start[2]-1; j > start[1]-1; j--) { // low positive

            if(Xw_sorted[j].value <= orig[i]) {// original high > largest low
                break;
            }
            al[i]++;
            be[i] += Xw_sorted[j].value;
        }
        table[i] = new int[(int)al[i]+1]();
        table[i][0] = (int)al[i];
        for(k = 1; k < al[i]+1; k++) {
            table[i][k] = Xw_sorted[start[2]-k].index;
        }
    }

    for(i = start[1]; i < start[2]; i++) { //low positive

        al[i] = 0;
        be[i] = 0;
        for(j = start[0]; j < start[1]; j++) { // high positive

            if(Xw_sorted[j].value >= orig[i]) { // original low < smallest high
                break;
            }
            al[i]++;
            be[i] += Xw_sorted[j].value;
        }
        table[i] = new int[(int)al[i]+1]();
        table[i][0] = (int)al[i];
        for(k = 1; k < al[i]+1; k++) {
            table[i][k] = Xw_sorted[start[0]+k-1].index;
        }
    }

    for(i = start[2]; i < start[3]; i++) {// high negative

        al[i] = 0;
        be[i] = 0;
        for(j = l-1; j > start[3]-1; j--) {// low negative

            if(Xw_sorted[j].value <= orig[i]) { //original high > largest low
                break;
            }
            al[i]++;
            be[i] += Xw_sorted[j].value;
        }
        table[i] = new int[(int)al[i]+1]();
        table[i][0] = (int)al[i];
        for(k = 1; k < al[i]+1; k++) {
            table[i][k] = Xw_sorted[l-k].index;
        }
    }

    for(i = start[3]; i < l; i++) {// low negative

        al[i] = 0;
        be[i] = 0;
        for(j = start[2]; j < start[3]; j++) { //high negative

            if(Xw_sorted[j].value >= orig[i]) {//original low < smallest high
                break;
            }
            al[i]++;
            be[i] += Xw_sorted[j].value;
        }
        table[i] = new int[(int)al[i]+1]();
        table[i][0] = (int)al[i];
        for(k = 1; k < al[i]+1; k++) {
            table[i][k] = Xw_sorted[start[2]+k-1].index;
        }
    }
}

int l2r_l2_primal_fun::get_nr_variable(void) {

    return prob->n;
}

void l2r_l2_primal_fun::Xv(double *v, double *Xv) {

    int i; 
    feature_node **x = prob->x;

    for(i = 0; i < prob->l; i++) {

        feature_node *s = x[i];
        Xv[i] = 0;

        while(s->index != -1) {

            Xv[i] += v[s->index-1] * s->value;
            s++;
        }
    }
}

void l2r_l2_primal_fun::XTv(double *v, double *XTv) {

    int i;
    int w_size = get_nr_variable();
    feature_node **x = prob->x;
    for(i = 0; i < w_size; i++) {

        XTv[i] = 0;
    }

    for(i = 0; i < prob->l; i++) {
        
        feature_node *s = x[i];
        while(s->index != -1) {
            
            XTv[s->index-1] += v[i] * s->value;
            s++;
        }
    }
}

void l2r_l2_primal_fun::subXTv(double *v, double *XTv) {

    int i;
    int w_size = get_nr_variable();
    feature_node **x = prob->x;

    for(i = 0; i < w_size; i++) {
        XTv[i] = 0;
    }

    for(i = 0; i < sizeI; i++) {

        feature_node *s = x[I[i]];
        while(s->index != -1) {

            XTv[s->index-1] += v[i] * s->value;
            s++;
        }
    }
}
    
void l2r_l2_primal_fun::subXv(double *v, double *Xv) {

    int i;
    feature_node **x = prob->x;

    for(i = 0; i < sizeI; i++) {

        feature_node *s = x[I[i]];
        Xv[i] = 0;
        while(s->index != -1) {
            
            Xv[i] += v[s->index-1] * s->value;
            s++;
        }
    }
}

void l2r_l2_primal_fun::set_print_string(
        void (*print_string) (const char* buf)) {

    tron_print_string = print_string;
}

void l2r_l2_primal_fun::info(const char *fmt, ...) {

    char buf[BUFSIZ];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    (*tron_print_string)(buf);
}
