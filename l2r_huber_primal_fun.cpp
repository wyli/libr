#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cstring>
#include "tron.h"

void *l2r_huber_ptr;

struct Xw {
    int index;
    double value;
};

static void print_string_l2r_huber(const char *s) {
    fputs(s, stdout);
    fflush(stdout);
}

typedef int (*compnode)(const void *, const void *);
int compare_nodes(struct Xw *a, struct Xw *b) {

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

l2r_huber_primal_fun::l2r_huber_primal_fun(const problem *prob, double C_r) {

    int i;
    this->prob = prob;
    set_print_string(print_string_l2r_huber);
    info("C_r = %f", C_r);

    int l = prob->l;
    start = prob->start;
    count = prob->count;

    z = new double[l];
    z_r = new double[l];
    I = new int[l];

    alpha = new double[l];
    beta = new double[l];
    theta = new double[l];

    xs = new double[l];
    cache = new int*[l];
    for(i = 0; i < l; i++) {
        
        cache[i] = new int[1]();
    }
    xw_sorted = new Xw[l];
    for(i = 0; i < l; i++) {

        xw_sorted[i].index = i;
    }

    this->C_e = prob->C_e;
    this->C_r = C_r;
}

l2r_huber_primal_fun::~l2r_huber_primal_fun() {
    
    info("rank C in input is %f\n", this->C_r);
    info("I'm quitting with total violation: %f\n", totalViolation);
    delete[] z;
    delete[] z_r;
    delete[] I;
    delete[] alpha;
    delete[] beta;
    delete[] theta;
    delete[] xs;
    delete[] xw_sorted;

    int i;
    for(i = 0; i < prob->l; i++) {

        delete[] cache[i];
    }
    delete[] cache;
}

double l2r_huber_primal_fun::fun(double *w) {

    int i, j;
    double f = 0, f_r = 0;
    double *y = prob->y;
    int w_size = get_nr_variable();
    
    Xv(w, z);
    memcpy(z_r, z, (size_t)prob->l * sizeof(double));
    // l2r
    for(i = 0; i < w_size; i++) {

        f += w[i] * w[i];
    }
    f /= 2.0;

    // l2 loss
    for(i = 0; i < prob->l; i++) {

        z[i] = y[i] * z[i];
        double d = 1 - z[i];
        if(d > 0) {

            f += C_e[i] * d * d;
        }
    }

    // rank loss
    shiftXW(z_r);
    totalViolation = 0;
    for(i = start[0]; i < start[1]; i++) {
        for(j = start[1]; j < start[2]; j++) {

            double er = z_r[j] - z_r[i];
            if(er <= 0) { 

                continue;
            } else if(er <= 1) {

                totalViolation++;
                f_r += C_r * er * er;
            } else {

                totalViolation++;
                f_r += 2 * er - 1;
            }
        }
    }

    if(totalViolation > 0) {

        f += (1/totalViolation) * C_r * f_r;
    }

    return f;
}

void l2r_huber_primal_fun::grad(double *w, double *g) {

    int i;
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
    shiftXW(z_r);
    for(i = 0; i < prob->l; i++) {

        xw_sorted[i].value = z_r[i];
    }

    for(i = 0; i < 4; i++) {

        qsort((void *)&xw_sorted[start[i]],
                (size_t)count[i],
                sizeof(struct Xw),
                (compnode)compare_nodes);
    }

    findab(alpha, beta, theta, xw_sorted, z_r, cache);

    double violation_1 = 0;
    double violation_2 = 0;
    for(i = 0; i < prob->l; i++) {

        violation_1 += alpha[i];
        if(theta[i] > 0) {
            violation_2 += theta[i];
        } else {
            violation_2 += theta[i] * -1.0;
        }
    }
    totalViolation = (violation_1 + violation_2)/2.0;
    info("normal violation: %.0f ", violation_1/2);
    info("worse violation: %.0f ", violation_2/2);
    info("total: %.0f\n", totalViolation);

    double *g_delta = new double[prob->l]();
    feature_node **x = prob->x;
    for(i = 0; i < prob->l; i++) {

        feature_node *xx = x[i];

        if(alpha[i] > 0) {

            while(xx->index != -1) {

                g_delta[xx->index-1] += 
                    (alpha[i] * z_r[i] - -beta[i]) * xx->value;
                xx++;
            }
        } else if(theta[i] >= 1 || theta[i] <= -1) {

            while(xx->index != -1) {

                g_delta[xx->index-1] +=
                    2.0 * (theta[i] * xx->value) - 1;
                xx++;
            }
        }
    }

    // combine together
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

void l2r_huber_primal_fun::Hv(double *g, double *Hs) {

    int i, j;
    int w_size = get_nr_variable();
    double *wa = new double[sizeI];

    subXv(g, wa);
    for(i = 0; i < sizeI; i++) {

        wa[i] = C_e[I[i]] * wa[i];
    }
    subXTv(wa, Hs);


    Xv(g, xs);
    double *Hs_delta = new double[w_size]();
    double *gamma = new double[prob->l]();
    feature_node **x = prob->x;
    for(i = 0; i < prob->l; i++) {
        for(j = 0; j < cache[i][0]; j++) {

            gamma[i] += xs[cache[i][j+1]];
        }
    }

    for(i = 0; i < prob->l; i++) {

        feature_node *xx = x[i];
        while(xx->index != -1) {

            Hs_delta[xx->index-1] += (alpha[i] * xs[i] - gamma[i]) * xx->value;
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
}

void l2r_huber_primal_fun::findab(double* al, double* be, double* ta,
        struct Xw* Xw_sorted, const double* orig, int **table) {

    int l = prob->l;
    int i, j, k;
    for(i = 0; i < l; i++) {

        delete[] table[i];
    }


    for(i = 0; i < start[1]; i++) {// high positive

        al[i] = 0;
        be[i] = 0;
        ta[i] = 0;
        for(j = start[2]-1; j > start[1]-1; j--) { // low positive

            double loss = Xw_sorted[j].value - orig[i];
            if(loss < 0) {// original high > largest low
                break;
            } else if(loss > 1) {
                ta[i]--;
            } else {
                al[i]++;
                be[i] += Xw_sorted[j].value;
            }
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
        ta[i] = 0;
        for(j = start[0]; j < start[1]; j++) { // high positive

            double loss = orig[i] - Xw_sorted[j].value;
            if(loss < 0) { // original low < smallest high
                break;
            } else if(loss > 1) {
                ta[i]++;
            } else {
                al[i]++;
                be[i] += Xw_sorted[j].value;
            }
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
        ta[i] = 0;
        for(j = l-1; j > start[3]-1; j--) {// low negative

            double loss = Xw_sorted[j].value - orig[i];
            if(loss < 0) { //original high > largest low
                break;
            } else if(loss > 1) {
                ta[i]--;
            } else {
                al[i]++;
                be[i] += Xw_sorted[j].value;
            }
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
        ta[i] = 0;
        for(j = start[2]; j < start[3]; j++) { //high negative

            double loss = orig[i] - Xw_sorted[j].value;
            if(loss < 0) {//original low < smallest high
                break;
            } else if(loss > 1) {
                ta[i]++;
            } else {
                al[i]++;
                be[i] += Xw_sorted[j].value;
            }
        }
        table[i] = new int[(int)al[i]+1]();
        table[i][0] = (int)al[i];
        for(k = 1; k < al[i]+1; k++) {
            table[i][k] = Xw_sorted[start[2]+k-1].index;
        }
    }
}

void l2r_huber_primal_fun::shiftXW(double *orig) {

    int i = 0;

    for(i = start[0]; i < start[1]; i++) {
        
        orig[i] -= 0.5;
    }

    for(i = start[1]; i < start[2]; i++) {

        orig[i] += 0.5;
    }

    for(i = start[2]; i < start[3]; i++) {

        orig[i] -= 0.5;
    }

    for(i = start[3]; i< prob->l; i++) {

        orig[i] += 0.5;
    }
}

void l2r_huber_primal_fun::Xv(double *v, double *Xv) {

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

void l2r_huber_primal_fun::XTv(double *v, double *XTv) {

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

void l2r_huber_primal_fun::subXv(double *v, double *Xv) {

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

void l2r_huber_primal_fun::subXTv(double *v, double *XTv) {

    int i;
    int w_size = get_nr_variable();
    feature_node **x = prob->x;

    for(i =0; i < w_size; i++) {

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

int l2r_huber_primal_fun::get_nr_variable(void) {

    return prob->n;
}

void l2r_huber_primal_fun::set_print_string(
        void (*print_string) (const char* buf)) {

    tron_print_string = print_string;
}

void l2r_huber_primal_fun::info(const char *fmt, ...) {

    char buf[BUFSIZ];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    (*tron_print_string)(buf);
}
