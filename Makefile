CXX ?= g++
CC ?= gcc
#CFLAGS = -Wall -Wconversion -O3 -fPIC
CFLAGS = -Wall -Wconversion -g
LIBS = blas/blas.a
SHVER = 1
OS = $(shell uname)
#LIBS = -lblas

all: train predict

lib: linear.o l2r_l2_primal_fun.o l2r_huber_primal_fun.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} linear.o l2r_l2_primal_fun.o l2r_huber_primal_fun.o  blas/blas.a -o liblinear.so.$(SHVER)

train: l2r_l2_primal_fun.o l2r_huber_primal_fun.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c l2r_l2_primal_fun.o l2r_huber_primal_fun.o linear.o $(LIBS)

predict: tnc.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c l2r_l2_primal_fun.o l2r_huber_primal_fun.o linear.o $(LIBS)

#tron.o: tron.cpp tron.h tnc.h
	#$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

#truncatedNewton.o: truncatedNewton.cpp tron.h
#	$(CXX) $(CFLAGS) -c -o truncatedNewton.o truncatedNewton.cpp

#tnc.o: tnc.c tnc.h
#	$(CXX) $(CFLAGS) -c -o tnc.o tnc.c

l2r_l2_primal_fun.o: l2r_l2_primal_fun.cpp tron.h tnc.h
	$(CXX) $(CFLAGS) -c -o l2r_l2_primal_fun.o l2r_l2_primal_fun.cpp

l2r_huber_primal_fun.o: l2r_huber_primal_fun.cpp tron.h tnc.h
	$(CXX) $(CFLAGS) -c -o l2r_huber_primal_fun.o l2r_huber_primal_fun.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	make -C matlab clean
	rm -f *~ linear.o train predict liblinear.so.$(SHVER)
	rm -f  l2r_l2_primal_fun.o l2r_huber_primal_fun.o
