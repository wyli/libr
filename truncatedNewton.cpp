#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <cstring>
#include "tron.h"

#ifdef __cplusplus
extern "C" {
#endif
    extern double dnrm2_(int *, double *, int *);
    extern double ddot_(int *, double *, int *, double *, int *);
#ifdef __cplusplus
}
#endif

TruncatedNewton::TruncatedNewton(const function *fun_obj) {

    this->fun_obj = const_cast<function *>(fun_obj);
}

TruncatedNewton::~TruncatedNewton() {
}

void TruncatedNewton::info(const char *fmt, ...) {

    char buf[BUFSIZ];
    va_list ap;
    vsprintf(buf, fmt, ap);
    va_end(ap);
    (*tron_print_string)(buf);
}

void TruncatedNewton::searchW(double *w) {

    n = fun_obj->get_nr_variable();
    double *g = new double[n];
    double *deltaW = new double[n];
    double a;
    int i, iter;

    iter = 0;
    while(1) {

        fun_obj->grad(w, g);
        linearCG(deltaW, g, 100, 0.001);
        a = 0;

        for(i = 0; i < n; i++) {

            a += deltaW[i] * deltaW[i];
        }

        for(i = 0; i < n; i++) {

            w[i] = w[i] - deltaW[i];
        }

        info("object: %f\n", fun_obj->fun(w));
        
        if(a < 0.001) break;
        if(iter > 100) break;
        iter++;
    }

    delete[] g;
    delete[] deltaW;
}

void TruncatedNewton::linearCG(
        double *x, double *b, int max_iter, double tol) {

    double resid;
    int inc = 1;
    int i;
    int n = fun_obj->get_nr_variable();
    double alpha = 0, beta = 0, rho = 0, rho_1 = 0;
    double *r = new double[n];
    double *p = new double[n];
    double *q = new double[n];

    double normb;
    normb = dnrm2_(&n, r, &inc);
    fun_obj->Hv(x, r);
    for(i = 0; i < n; i++) {

        r[i] = b[i] - r[i];
    }

    if(normb == 0.0) {

        normb = 1;
    }

    resid = dnrm2_(&n, r, &inc)/normb;

    if(resid <= tol) {

        tol = resid;
        max_iter = 0;
        return;
    }

    for(i = 1; i < max_iter; i++) {

        rho = ddot_(&n, r, &inc, r, &inc);

        if(i == 1) {

            for(int j = 0; j < n; j++) {

                p[j] = r[j];
            }
        } else {
            
            beta = rho/rho_1;
            for(int j = 0; j < n; j++) {

                p[j] = r[j] + beta * p[j];
            }
        }

        fun_obj->Hv(p, q);
        alpha = rho/ddot_(&n, p, &inc, q, &inc);
        for(int j = 0; j < n; j++) {

            x[j] += alpha * p[j];
            r[j] -= alpha * q[j];
        }

        resid = dnrm2_(&n, r, &inc)/normb;
        if(resid <= tol) {
            
            tol = resid;
            max_iter = i;
            return;
        }
        rho_1 = rho;
    }

    tol = resid;

    delete[] r;
    delete[] p;
    delete[] q;
}

void TruncatedNewton::set_print_string(void (*print_string) (const char *buf)) {

    tron_print_string = print_string;
}
