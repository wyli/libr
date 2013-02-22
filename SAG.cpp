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

            cache[i][j] = new double[2]();
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

    info("eps: %f\n", eps);
    info("allocated %f MB\n", (pos*neg*2*8)/1000000.0);
    int pass = 0;
    int i, j, k;
    int inc = 1;
    int notConverged = 1;
    double old_score = HUGE_VAL;
    double alpha;
    double n = double(pos*neg);
    double one_coeff = 0;
    double m = 0;

    double *w = new double[w_size]();
    double L = 1;
    while(notConverged) {

        double *grad = new double[w_size]();
        double *wa = new double[2]();
        double *prev = new double[w_size]();
        for(i = pos-1; i >= 0; i--) {
            for(j = 0; j < neg; j++) {


                if(m < n) m++;

                fun_obj->wTa(w, i, j+pos, wa);

                fun_obj->pairGrad(wa, i, j+pos, grad);

 
                L = lineSearchWrapper(L, grad, w, i, j);
                alpha = 2.0/(L + 1);
                one_coeff = 1.0 - alpha/n;

                if(pass > 0) {

                    fun_obj->pairGrad(cache[i][j], i, j+pos, prev);
                }


                for(k = 0; k < w_size; k++) {

                    sumy[k] = sumy[k] - prev[k] + grad[k];
                    w[k] = one_coeff * w[k] - (alpha/m) * sumy[k];
                    grad[k] = 0;
                    prev[k] = 0;
                    //cache[i][j][k] = grad[k];
                }
                cache[i][j][0] = wa[0];
                cache[i][j][1] = wa[1];
                wa[0] = 0;
                wa[1] = 0;
            }
        }
        delete[] grad;
        delete[] wa;
        delete[] prev;

        double now_score = dnrm2_(&w_size, sumy, &inc) / n;
        if(old_score < now_score) {

            notConverged = 0;
        }
        info("\nPass: %d, L: %e, sumY: %f,  fun: %f\n",
                pass, L, now_score, fun_obj->fun(w));
        if(pass > 20) {

            notConverged = 0;
            info("Max_iter reached\n");
        }
        old_score = now_score;
        pass++;
    }
    memcpy(w_out, w, sizeof(double)*(size_t)w_size);
    delete[] w;
}

double SAG::lineSearchWrapper(double L, double *grad, double *w, int i, int j) {

    int inc = 1;
    double diff;
    double pairloss_ij = fun_obj->pairLoss(w, i, j+pos);
    double normF = dnrm2_(&w_size, grad, &inc);

    diff = lineSearch(L, normF, grad, w, i, j) - pairloss_ij;
    while(diff > eps && L < 1e100) {
        info(".");
        L = L * 2.0;
        diff = lineSearch(L, normF, grad, w, i, j) - pairloss_ij;
    }

    return L;
}

double SAG::lineSearch(double L, double normF, double *grad, double *w, int i, int j) {

    int k;
    double a;

    if(normF > 1e-8) {

        double *temp = new double[w_size]();
        for(k = 0; k < w_size; k++) {

            temp[k] = w[k] - (1.0/L)*grad[k];
        }

        a = fun_obj->pairLoss(temp, i, j+pos) 
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
