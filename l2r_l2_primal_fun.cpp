#include "tron.h"

inline
double l2r_l2_primal_fun::rankLoss(double t) {

    if(t < 0) {

        return 0.5 - t;
    }
    if(t < 1) {

        return (1 - t) * (1 - t) * 0.5;
    }
    return 0;
}

inline
double l2r_l2_primal_fun::rankLossGrad(double t) {

    if(t < 0) {

        return -1.0;
    }
    if(t < 1) {

        return t - 1.0;
    }
    return 0;
}

inline
double l2r_l2_primal_fun::classLoss(double t) {

    // squared soft margin
    //if(t < 0) {

    //    return t * t;
    //}
    //return 0;
    return t < 0 ? t*t : 0;
}

inline
double l2r_l2_primal_fun::classLossGrad(double t) {

    //if(t < 0) {

    //    return 2.0 * t;
    //}
    //return 0;
    return t < 0 ? 2.0*t : 0;
}
