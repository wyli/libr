#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cstring>
#include "tron.h"

static void print_string_l2r_huber(const char *s) {

    fputs(s, stdout);
    fflush(stdout);
}

l2r_huber_primal_fun::l2r_huber_primal_fun(const problem *prob, double C_r, double p) {

    this->prob = prob;
    set_print_string(print_string_l2r_huber);

    start = prob->start;
    count = prob->count;
    nr_over = 1.0/ (count[0]*count[1]);

    this->C_e = prob->C_e;
    this->C_r = C_r;
    this->C_b = p;

    int l = prob->l;
    z = new double[l]();

    int i;
    for(i = 0; i < l; i++) {
        this->C_e[i] = this->C_e[i] * C_b;
    }
}

l2r_huber_primal_fun::~l2r_huber_primal_fun() {

    delete[] z;
}

double l2r_huber_primal_fun::fun(double *w) {

    int i, j;
    double f = 0, f_c = 0, f_r = 0;
    double *y = prob->y;

    for(i = 0; i < get_nr_variable(); i++) {

        f += w[i] * w[i];
    }

    Xv(w, z);

    for(i = 0; i < prob->l; i++) {

        f_c +=  C_e[i] * classLoss(y[i]*z[i] - 1);
    }

    f_c *= 2.0;

    if(C_r > 0) {
        for(i = start[0]; i < start[1]; i++) {
            for(j = start[1]; j < prob->l; j++) {

                f_r += rankLoss(z[i] - z[j] - pairDistance(i, j));
            }
        }
    }

    f += (f_c + C_r * f_r);
    return nr_over * f;
}

double l2r_huber_primal_fun::pairDistance(int i, int j) {

    return 1.0/(C_e[i] + C_e[j]);
}

void l2r_huber_primal_fun::pairGrad(double *wa, int i, int j, double *g) {

    feature_node **s = prob->x;
    feature_node *s_i = s[i];
    feature_node *s_j = s[j];

    while(s_i->index != -1 && s_j->index != -1) {

        g[s_i->index-1] = wa[0] * s_i->value + wa[1] * s_j->value;
        s_i++;
        s_j++;
    }
    while(s_i->index != -1) {

        g[s_i->index-1] = wa[0] * s_i->value;
        s_i++;
    }
    while(s_j->index != -1) {

        g[s_j->index-1] = wa[1] * s_j->value;
        s_j++;
    }
}

void l2r_huber_primal_fun::wTa(double *w, int i, int j, double *wa) {

    double a;
    wa[0] = 0.0;
    wa[1] = 0.0;
    feature_node **s = prob->x;
    feature_node *s_i = s[i];
    feature_node *s_j = s[j];
    while(s_i->index != -1) {
        wa[0] += s_i->value * w[s_i->index-1];
        wa[1] -= s_j->value * w[s_j->index-1];
        s_i++;
        s_j++;
    }
    while(s_i->index != -1) {
        wa[0] += s_i->value * w[s_i->index-1];
        s_i++;
    }
    while(s_j->index != -1) {
        wa[1] -= s_j->value * w[s_j->index-1];
        s_j++;
    }
    if(C_r > 0) {

        a = rankLossGrad(wa[0] + wa[1] - pairDistance(i, j)) * C_r;
        wa[0] = classLossGrad(wa[0] - 1) * C_e[i] + a;
        wa[1] = classLossGrad(wa[1] - 1) * C_e[j] * (-1.0) - a;
    } else {

        wa[0] = classLossGrad(wa[0] - 1) * C_e[i];
        wa[1] = classLossGrad(wa[1] - 1) * C_e[j] * (-1.0);
    }
}

double l2r_huber_primal_fun::pairLoss(double *w, int i, int j) {

    double wx_i = 0.0;// wTx(w, i) - 1;
    double wx_j = 0.0;// wTx(w, j) - 1;

    feature_node **s = prob->x;
    feature_node *s_i = s[i];
    feature_node *s_j = s[j];
    while(s_i->index != -1) {
        wx_i += s_i->value * w[s_i->index-1];
        wx_j -= s_j->value * w[s_j->index-1];
        s_i++;
        s_j++;
    }
    while(s_i->index != -1) {
        wx_i += s_i->value * w[s_i->index-1];
        s_i++;
    }
    while(s_j->index != -1) {
        wx_j -= s_j->value * w[s_j->index-1];
        s_j++;
    }

    double dl = classLoss(wx_i - 1) * C_e[i];
    dl += classLoss(wx_j - 1) * C_e[j];

    if(C_r > 0){
        dl += rankLoss(wx_i + wx_j - pairDistance(i, j)) * C_r;
    }

    return dl;
}

inline
double l2r_huber_primal_fun::rankLoss(double t) {

    // huber loss
    if(t <= 0) {

        return 1.0 - t;
    } 

    if(t < 1) {

        return (1.0 - t) * (1.0 - t);
    }

    return 0;
}

inline
double l2r_huber_primal_fun::rankLossGrad(double t) {


    if(t <= 0) {

        return -1.0;
    }

    if(t < 1) {

        return 2.0 * t - 2.0;
    }

    return 0;
}

inline
double l2r_huber_primal_fun::classLoss(double t) {

    // squared soft margin
    if(t < 0) {

        return t * t;
    }

    return 0;
}

double l2r_huber_primal_fun::classLossGrad(double t) {

    if(t < 0) {

        return 2.0 * t;
    }
    
    return 0;
}

inline
double l2r_huber_primal_fun::wTx(double *w, int i) {

    double dotprod = 0.0;
    feature_node **s = prob->x;
    feature_node *s_i = s[i];
    while(s_i->index != -1) {
        dotprod += s_i->value * w[s_i->index-1];
        s_i++;
    }
    return dotprod;
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

