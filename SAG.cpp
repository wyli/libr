#include "tron.h"
#include <stdarg.h>
#include <stdio.h>
#include <cmath>
#include <cstring>

//#ifdef __cplusplus
//extern "C" {
//#endif
//
//    extern double ddot_(int *, double *, int *, double *, int *);
//    extern double dnrm2_(int *, double *, int *);
//#ifdef __cplusplus
//}
//#endif

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
    temp_grad = new double[w_size]();
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
    delete[] temp_grad;
}

void SAG::solver(double *w_out) {

    info("eps: %f\n", eps);
    info("allocated %f MB\n", (pos*neg*3*8)/1000000.0);
    int i, j, k, pass = 0;
    int notConverged = 1;
    double now_score, old_score = HUGE_VAL;
    double n = double(pos*neg);
    double m = 0.0;
    double *w = new double[w_size]();

    double L = 100;
    double alpha = 2.0/(L + 0.001 * n);
    double one_coeff = 1.0 - 2.0 * alpha / n;
    alpha = 2.0/(L+0.001 * n);
    one_coeff = 1.0 - alpha / n;

    double normy = 0.0;
    double *wa = new double[2]();
    double *tempgrad = new double[2]();
    double *grad = new double[w_size]();
    while(notConverged) {


        for(i = pos-1; i >= 0; i--) {
            for(j = 0; j < neg; j++) {


                if(m < n) m++;

                fun_obj->wTa(w, i, j+pos, wa);
//                fun_obj->pairGrad(wa, i, j+pos, grad);
//                L = lineSearchWrapper(L, grad, w, i, j);

                if(pass > 0) {

                    tempgrad[0] = wa[0] - cache[i][j][0];
                    tempgrad[1] = wa[1] - cache[i][j][1];
                    fun_obj->pairGrad(tempgrad, i, j+pos, grad);
                } else {

                     //first pass
                    fun_obj->pairGrad(wa, i, j+pos, grad);
                }

                //alpha = 2.0/(L+0.001 * n);
                //one_coeff = 1.0 - 2.0 * alpha / n;
                //one_coeff = 1.0 - alpha / n;
                normy = 0.0;

                //for(k = 0; k < w_size; k++) {

                //    sumy[k] = sumy[k] + grad[k];
                //    normy += sumy[k] * sumy[k];
                //    w[k] = one_coeff * w[k] - (alpha/m) * sumy[k];
                //    grad[k] = 0.0;
                //}
                for(k = 0; k < 200; k+=10) {

                    sumy[k] = sumy[k] + grad[k];
                    normy += sumy[k] * sumy[k];
                    w[k] = one_coeff * w[k] - (alpha/m) * sumy[k];
                    grad[k] = 0.0;

                    sumy[k+1] = sumy[k+1] + grad[k+1];
                    normy += sumy[k+1] * sumy[k+1];
                    w[k+1] = one_coeff * w[k+1] - (alpha/m) * sumy[k+1];
                    grad[k+1] = 0.0;

                    sumy[k+2] = sumy[k+2] + grad[k+2];
                    normy += sumy[k+2] * sumy[k+2];
                    w[k+2] = one_coeff * w[k+2] - (alpha/m) * sumy[k+2];
                    grad[k+2] = 0.0;

                    sumy[k+3] = sumy[k+3] + grad[k+3];
                    normy += sumy[k+3] * sumy[k+3];
                    w[k+3] = one_coeff * w[k+3] - (alpha/m) * sumy[k+3];
                    grad[k+3] = 0.0;

                    sumy[k+4] = sumy[k+4] + grad[k+4];
                    normy += sumy[k+4] * sumy[k+4];
                    w[k+4] = one_coeff * w[k+4] - (alpha/m) * sumy[k+4];
                    grad[k+4] = 0.0;

                    sumy[k+5] = sumy[k+5] + grad[k+5];
                    normy += sumy[k+5] * sumy[k+5];
                    w[k+5] = one_coeff * w[k+5] - (alpha/m) * sumy[k+5];
                    grad[k+5] = 0.0;

                    sumy[k+6] = sumy[k+6] + grad[k+6];
                    normy += sumy[k+6] * sumy[k+6];
                    w[k+6] = one_coeff * w[k+6] - (alpha/m) * sumy[k+6];
                    grad[k+6] = 0.0;

                    sumy[k+7] = sumy[k+7] + grad[k+7];
                    normy += sumy[k+7] * sumy[k+7];
                    w[k+7] = one_coeff * w[k+7] - (alpha/m) * sumy[k+7];
                    grad[k+7] = 0.0;

                    sumy[k+8] = sumy[k+8] + grad[k+8];
                    normy += sumy[k+8] * sumy[k+8];
                    w[k+8] = one_coeff * w[k+8] - (alpha/m) * sumy[k+8];
                    grad[k+8] = 0.0;

                    sumy[k+9] = sumy[k+9] + grad[k+9];
                    normy += sumy[k+9] * sumy[k+9];
                    w[k+9] = one_coeff * w[k+9] - (alpha/m) * sumy[k+9];
                    grad[k+9] = 0.0;
                }
                cache[i][j][0] = wa[0];
                cache[i][j][1] = wa[1];
                wa[0] = 0.0;
                wa[1] = 0.0;
                tempgrad[0] = 0.0;
                tempgrad[1] = 0.0;

                now_score = normy;

                if(old_score < now_score && pass > 5) {

                    //info("\nconverged i: %d, j:%d", i, j);
                    notConverged = 0;
                    goto outofloop;
                }
                old_score = now_score;
            }
        }

outofloop:
        old_score = now_score;

        if(pass > 10) {

            notConverged = 0;
            info("Max_iter reached\n");
        }

        //info("\nPass: %d, L: %e, sumY: %e, fun: %e\n",
        //        pass, L, now_score, fun_obj->fun(w));
        pass++;
    }

    memcpy(w_out, w, sizeof(double)*(size_t)w_size);
    delete[] w;
    delete[] wa;
    delete[] grad;
    delete[] tempgrad;
}

double SAG::lineSearchWrapper(double L, double *grad, double *w, int i, int j) {

    //int inc = 1;

    //double normF = ddot_(&w_size, grad, &inc, grad, &inc);
    //if(normF < 1e-8) return L;

    //double pairloss_ij = fun_obj->pairLoss(w, i, j+pos);

    //double diff = lineSearch(L, normF, grad, w, i, j) - pairloss_ij;

    //while(diff > 0 && L < 1e10) {
    //    info(".");
    //    L *=  2.0;
    //    diff = lineSearch(L, normF, grad, w, i, j) - pairloss_ij;
    //}

    return L;
}

double SAG::lineSearch(double L, double normF, double *grad, double *w, int i, int j) {

    double delta = 1.0/L;
    for(int k = 0; k < w_size; k++) {

        temp_grad[k] = w[k] - delta*grad[k];
    }

    return fun_obj->pairLoss(temp_grad, i, j+pos) + delta * normF / 2.0;
}


void SAG::set_print_string(void (*print_string) (const char *buf)) {

    tron_print_string = print_string;
}
