#include "tron.h"

double l2r_l2_primal_fun::rankLoss(double t) {

    if(t < 0) {

        return 0.5 - t;
    }
    if(t < 1) {
        
        return (1 - t) * (1 - t) * 0.5;
    }
    return 0;
} 

double l2r_l2_primal_fun::rankLossGrad(double t) {

    if(t < 0) {
        
        return -1.0;
    }
    if(t < 1) {

        return t - 1.0;
    }
    return 0;
}

double l2r_l2_primal_fun::classLoss(double t) {

    if(t < 1) {

        return (1 - t) * (1 - t);
    }
    return 0;
}

double l2r_l2_primal_fun::classLossGrad(double t) {

    if(t < 1) {

        return 2.0 * t - 2.0;
    }
    return 0;
}
