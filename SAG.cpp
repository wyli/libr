#include "tron.h"
#include <stdarg.h>
#include <stdio.h>
#include <cmath>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

    extern double dnrm2_(int *, double *, int *);
#ifdef __cplusplus
}
#endif

static void print_string_sag(const char *buf) {

    fputs(buf, stdout);
    fflush(stdout);
}

void SAG::info(const char *fmt, ...) {

    char buf[BUFSIZ];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    (*tron_print_string)(buf);
}

SAG::SAG(const function *fun_obj, double eps) {

    set_print_string(print_string_sag);
    this->fun_obj = const_cast<function *>(fun_obj);
    w_size = this->fun_obj->get_nr_variable();
    sumy = new double[w_size]();
    pos = this->fun_obj->get_nr_positive();
    neg = this->fun_obj->get_nr_negative();
    int i, j;
    this->eps = eps;
    cache = new double**[pos]();
    for(i = 0; i < pos; i++) {

        cache[i] = new double*[neg]();
        for(j = 0; j < neg; j++) {

            cache[i][j] = new double[w_size]();
        }
    }
    tron_print_string = print_string_sag;
}

SAG::~SAG() {

    int i, j;
    for(i = 0; i < pos; i++) {
        for(j = 0; j < neg; j++) {

            delete[] cache[i][j];
        }
        delete[] cache[i];
    }
    delete[] cache;
    delete[] sumy;
}

void SAG::solver(double *w_out) {

    int pass = 0;
    int i, j, k;
    int notConverged = 1;
    double old_score = HUGE_VAL;
    double alpha;
    double n = double(pos*neg);

    double *w = new double[w_size]();
    double L = 1;
    while(notConverged) {

        for(i = pos-1; i >= 0; i--) {
            for(j = 0; j < neg; j++) {


                double *grad = new double[w_size]();
                fun_obj->pairGrad(w, i, j+pos, grad);
 
                L = lineSearchWrapper(L, grad, w, i, j);
                alpha = 2.0/(L + 1);

                for(k = 0; k < w_size; k++) {

                    sumy[k] = sumy[k] - cache[i][j][k] + grad[k];
                    w[k] = (1.0 - alpha/n) * w[k] - (alpha/n) * sumy[k];
                    cache[i][j][k] = grad[k];
                }
                delete[] grad;
            }
        }
        
        double now_score = fun_obj->fun(w);
        if(old_score < now_score || pass > 200) {

            notConverged = 0;
            if(pass > 200) {

                info("Max_iter reached, please try larger step size.\n");
            }
        }
        info("Pass: %d, L: %f, fun %f, pre %f\n", pass, L, now_score, old_score);
        old_score = now_score;
        pass++;
    }
    memcpy(w_out, w, sizeof(double)*(size_t)w_size);
    delete[] w;
}

double SAG::lineSearchWrapper(double L, double *grad, double *w, int i, int j) {

    double diff;
    do {
        diff = lineSearch(L, grad, w, i, j);
        if(diff > eps) {
            L = L * 2;
        }
    } while(diff > 1);

    return L;
}

double SAG::lineSearch(double L, double *grad, double *w, int i, int j) {

    int inc = 1, k;
    double a;

    double normF = dnrm2_(&w_size, grad, &inc);
    if(normF > 1e-8) {

        double *temp = new double[w_size]();
        for(k = 0; k < w_size; k++) {

            temp[k] = w[k] - (1.0/L)*grad[k];
        }

        a = fun_obj->pairLoss(temp, i, j+pos) 
            - fun_obj->pairLoss(w, i, j+pos)
            + (1.0/(2.0*L)) * normF;

        delete[] temp;
    } else {
        a = 0;
    }
    return a;
}


void SAG::set_print_string(void (*print_string) (const char *buf)) {

    tron_print_string = print_string;
}
