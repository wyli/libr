#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cstring>
#include "tron.h"

static void print_string_l2r_huber(const char *s) {

    fputs(s, stdout);
    fflush(stdout);
}

l2r_huber_primal_fun::l2r_huber_primal_fun(const problem *prob, double C_r) {

    this->prob = prob;
    set_print_string(print_string_l2r_huber);
    info("C_r = %f\n", C_r);

    start = prob->start;
    count = prob->count;

    this->C_e = prob->C_e;
    this->C_r = C_r;

    int l = prob->l;
    z = new double[l]();
}

l2r_huber_primal_fun::~l2r_huber_primal_fun() {

    delete[] z;
}

double l2r_huber_primal_fun::fun(double *w) {

    int i, j;
    double l = prob->l;
    double f = 0, f_c = 0, f_r = 0;
    double *y = prob->y;
    int w_size = get_nr_variable();

    for(i = 0; i < w_size; i++) {

        f += w[i] * w[i];
    }

    f /= 2.0;

    Xv(w, z);

    for(i = 0; i < prob->l; i++) {

        z[i] = y[i] * z[i];
        f_c +=  2 * C_e[i] * classLoss(z[i]);
    }

    for(i = start[0]; i < start[1]; i++) {
        for(j = start[1]; j <l; j++) {

            double t = z[i] - z[j] - pairDistance(i, j);
            f_r += rankLoss(t);
        }
    }

    f += (1.0/l)*(f_c + f_r);
    return f;
}

double l2r_huber_primal_fun::pairDistance(int i, int j) {

    return 1.0/(C_e[i] + C_e[j]);
}

void l2r_huber_primal_fun::pairGrad(double *w, int i, int j, double *g) {

    // assume g[i] = 0
    double t = 0;
    double dl = 0, dl_i = 0, dl_j = 0;

    t = wTx(w, i) - wTx(w, j) - pairDistance(i, j);
    dl = rankLossGrad(t);
    dl_i = classLossGrad(wTx(w, i));
    dl_j = classLossGrad(wTx(w, j));

    feature_node **s = prob->x;
    feature_node *s_i = s[i];
    feature_node *s_j = s[j];

    while(s_i->index != -1) {

        g[s_i->index-1] = 
            dl * (s_i->value - s_j->value)
            + dl_i * s_i->value 
            + dl_j * s_j->value;
        s_i++;
        s_j++;
    }
}

double l2r_huber_primal_fun::pairLoss(double *w, int i, int j) {

    double t = 0;
    double dl = 0;

    t = wTx(w, i) - wTx(w, j) - pairDistance(i, j);
    dl = rankLoss(t);
    dl += classLoss((prob->y[i]) * wTx(w, i));
    dl += classLoss((prob->y[j]) * wTx(w, j));

    return dl;
}

double l2r_huber_primal_fun::rankLoss(double t) {

    if(t < 1 && t > 0) {

        return (1.0 - t) * (1.0 - t);
    }

    if(t <= 0) {

        return 1.0 - 2.0 * t;
    } 
    return 0;
}

double l2r_huber_primal_fun::rankLossGrad(double t) {

    if(t < 1 && t > 0) {

        return 2.0 * t - 2.0;
    }

    if(t <= 0) {

        return -1.0;
    }

    return 0;
}

double l2r_huber_primal_fun::classLoss(double t) {

    if(t < 1) {

        return (1.0 - t) * (1.0 - t);
    }

    return 0;
}

double l2r_huber_primal_fun::classLossGrad(double t) {

    if(t < 1) {

        return 2*t - 2;
    }
    
    return 0;
}

double l2r_huber_primal_fun::wTx(double *w, int i) {

    double dotprod = 0;
    feature_node **s = prob->x;
    feature_node *s_i = s[i];
    while(s_i->index != -1) {
        dotprod += s_i->value * w[s_i->index-1];
        s_i++;
    }
    return dotprod;
}

int l2r_huber_primal_fun::get_nr_variable(void) {

    return prob->n;
}

int l2r_huber_primal_fun::get_nr_positive(void) {

    return count[0];
}

int l2r_huber_primal_fun::get_nr_negative(void) {

    return count[1];
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
