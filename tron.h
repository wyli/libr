#ifndef _TRON_H
#define _TRON_H

#include "linear.h"
#include "tnc.h"

class function
{
public:
	virtual double fun(double *w) = 0;
	virtual void pairGrad(double *w, int i, int j, double *g) = 0;
	virtual double pairLoss(double *w, int i, int j) = 0;
    double rankLoss(double);
    double rankLossGrad(double);
    double classLoss(double);
    double classLossGrad(double);


	virtual int get_nr_variable(void) = 0;
    virtual int get_nr_positive(void) = 0;
    virtual int get_nr_negative(void) = 0;
	virtual ~function(void){}
};

class SAG {

public:
    SAG(const function *fun_obj, double eps = 0.0008);
    ~SAG();

    void solver(double *w);
    void set_print_string(void (*i_print) (const char *buf));

private:
    void info(const char *fmt,...);
    void (*tron_print_string)(const char *buf);
    double lineSearch(double L, double *grad, double *w, int i, int j);
    double lineSearchWrapper(double L, double *grad, double *w, int i, int j);

    int pos, neg;
    double *sumy;
    function *fun_obj;
    double ***cache;
    int w_size;
    double eps;
};

class l2r_huber_primal_fun:public function {

    public:
        l2r_huber_primal_fun(const problem *prob, double weight);
        ~l2r_huber_primal_fun();

        double fun(double *w);
        void pairGrad(double *w, int i, int j, double *g);
        double pairLoss(double *w, int i, int j);

        double pairDistance(int, int);
        virtual double rankLoss(double);
        virtual double rankLossGrad(double);
        virtual double classLoss(double);
        virtual double classLossGrad(double);

        int get_nr_variable(void);
        int get_nr_positive(void);
        int get_nr_negative(void);
        void set_print_string(void (*i_print) (const char *buf));

   protected:
        void info(const char *, ...);
        void (*tron_print_string)(const char *);
        void Xv(double *, double *);
        double wTx(double *, int);


        const problem *prob;
        const int *start;
        const int *count;

        double *C_e;
        double C_r;

        double *z;
};

class l2r_l2_primal_fun:public l2r_huber_primal_fun {
public:
    l2r_l2_primal_fun(const problem *prob, double weight):l2r_huber_primal_fun(prob, weight){};
    double rankLoss(double);
    double rankLossGrad(double);
    double classLoss(double);
    double classLossGrad(double);
};
#endif
