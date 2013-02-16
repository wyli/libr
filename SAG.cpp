#include "tron.h"
#include <stdarg.h>
#include <stdio.h>
#include <cmath>

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

SAG::SAG(const function *fun_obj, double eps, double L_0) {

    set_print_string(print_string_sag);
    this->fun_obj = const_cast<function *>(fun_obj);
    int w_size = this->fun_obj->get_nr_variable();
    sumy = new double[w_size]();
    this->eps = eps;
    this->L = L_0;
    pos = this->fun_obj->get_nr_positive();
    neg = this->fun_obj->get_nr_negative();
    int i, j;
    cache = new double**[pos];
    for(i = 0; i < pos; i++) {

        cache[i] = new double*[neg];
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

void SAG::solver(double *w) {

    int i, j, k, inc = 1;
    int w_size = fun_obj->get_nr_variable();
    int total = pos + neg;

    w = new double[w_size]();
    while(1) {

        for(i = 0; i < pos; i++) {
            for(j = 0; j < neg; j++) {
                eps = 1/L;

                double *grad = new double[w_size]();
                fun_obj->pairGrad(w, i, j+pos, grad);
                for(k = 0; k < w_size; k++) {

                    sumy[k] = sumy[k] - cache[i][j][k] + grad[k];
                    w[k] = (1 - eps*0.5) * w[k] - (eps/total)*sumy[k];
                    cache[i][j][k] = grad[k];
                }

                double normF = dnrm2_(&w_size, grad, &inc);

                if(normF > 1e-8) {

                    double *temp = new double[w_size]();
                    for(int k = 0; k < w_size; k++) {

                        temp[k] = w[k] - (1/L)*grad[k];
                    }

                    double a = fun_obj->pairLoss(grad, i, j+pos) 
                        - fun_obj->pairLoss(temp, i, j+pos) + (1/2*L) * normF;

                    if(a > 0) {

                        L = L * 2;
                    }

                    delete[] temp;
                }
                info("fun: %.5f\n", fun_obj->fun(w));
                delete[] grad;
                double k = (1.0/total) * dnrm2_(&w_size, sumy, &inc);
                if(k < 0.001) {
                    
                    return;
                }
                L = L * pow(2, (-1/pos*neg));
            }
        }
    }
}

void SAG::set_print_string(void (*print_string) (const char *buf)) {

    tron_print_string = print_string;
}
