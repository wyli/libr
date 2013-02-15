#ifndef _TRON_H
#define _TRON_H

#include "linear.h"
#include "tnc.h"

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class TRON {

public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();

	void tron(double *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};

class TruncatedNewton {

public:
    TruncatedNewton(const function *fun_obj);
    ~TruncatedNewton();

    void searchW(double *w);
    void set_print_string(void (*i_print) (const char *buf));

private:
    void linearCG(double *x, double *b, int iter, double tol);
    void info(const char *fmt, ...);
    void (*tron_print_string)(const char *buf);

    function *fun_obj;
    int n;
};


class l2r_l2_primal_fun:public function {

    public:
        l2r_l2_primal_fun(const problem *prob, double weight);
        ~l2r_l2_primal_fun();

        double fun(double *w);
        void grad(double *w, double *g);
        void Hv(double *s, double *Hs);

        int get_nr_variable(void);
        void set_print_string(void (*i_print) (const char *buf));
        static int wrapper_fun(double x[], double *f, double g[], void *state);
        int run_solver(
            double w[], double *f, double gradient[],
            void *state, double lower_bound[], double upper_bound[],
             tnc_message message, double fmin);
    private:
        const problem *prob;
        const int *start;
        const int *count;

        void Xv(double *v, double *Xv);
        void XTv(double *v, double *XTv);
        void subXv(double *v, double *Xv);
        void subXTv(double *v, double *XTv);
        void findab(double *a, double *b,
                struct Xw* Xw_sorted, const double* notsorted, int** cache);
        void CXT_alpha_zr_beta(
                const double *z_r,
                const double *beta,
                double *g);

        int calculator(double x[], double *f, double g[], void *state);
        void info(const char *fmt, ...);
        void (*tron_print_string)(const char *buf);

        double *C_e; //penalti for regular svm
        double C_r;
        struct Xw *Xw_sorted;
        double *z_r;
        double *z;
        double *Xs;
        int sizeI;
        int *I; //active data points indicator
        double *alpha;
        double *beta;
        int** cache; // index cache
        double totalViolation; // number of rank violating
};

class l2r_huber_primal_fun:public function {

    public:
        l2r_huber_primal_fun(const problem *prob, double weight);
        ~l2r_huber_primal_fun();

        double fun(double *w);
        void grad(double *w, double *g);
        void Hv(double *s, double *Hs);

        int get_nr_variable(void);
        void set_print_string(void (*i_print) (const char *buf));
        int run_solver(
                double *, double *, double *,
                void *, double *, double *,
                tnc_message, double);

   private:
        void info(const char *, ...);
        void (*tron_print_string)(const char *);
        void Xv(double *, double *);
        void XTv(double *, double *);
        void subXv(double *, double *);
        void subXTv(double *, double *);
        void shiftXW(double *);
        void findab(double *, double *, double *, 
                struct Xw *, const double *, int **);

        int calculator(double *, double *, double *, void *);
        static int wrapper_fun(double *, double *, double *, void *);

        const problem *prob;
        const int *start;
        const int *count;

        double *C_e;
        double C_r;

        int sizeI;
        int *I;
        double *z;

        double totalViolation;

        double *z_r;
        double *alpha;
        double *beta;
        double *theta;
        double *xs;
        int **cache;
        struct Xw *xw_sorted;
};
#endif