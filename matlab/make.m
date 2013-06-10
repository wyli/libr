% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix

try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		mex libsvmread.c
		mex libsvmwrite.c
		mex train.c linear_model_matlab.c ../linear.cpp ../blas/*.c
		mex predict.c linear_model_matlab.c ../linear.cpp ../blas/*.c
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
    else
       % mex CFLAGS="-g" libsvmread.c
       % mex CFLAGS="-g" libsvmwrite.c
       % mex CFLAGS="-g" train.c linear_model_matlab.c ../l2r_l2_primal_fun.cpp ../l2r_huber_primal_fun.cpp ../linear.cpp ../SAG.cpp "../blas/*.c"
       % mex CFLAGS="-std=c99 -g" predict.c linear_model_matlab.c ../l2r_l2_primal_fun.cpp ../l2r_huber_primal_fun.cpp ../linear.cpp ../SAG.cpp "../blas/*.c"
		%mex CFLAGS="-x c++ -v" -largeArrayDims libsvmread.c
		%mex CFLAGS="-x c++ -v" -largeArrayDims libsvmwrite.c
		mex CFLAGS="-O3 -fslp-vectorize -fslp-vectorize-aggressive -mllvm -bb-vectorize-aligned-only -mllvm -unroll-allow-partial -mllvm -unroll-runtime -march=native -ffast-math" -largeArrayDims train.c linear_model_matlab.c ../l2r_l2_primal_fun.cpp ../l2r_huber_primal_fun.cpp ../linear.cpp ../SAG.cpp "../blas/*.c"
		mex CFLAGS="-O3 -fslp-vectorize -fslp-vectorize-aggressive -mllvm -bb-vectorize-aligned-only -mllvm -unroll-allow-partial -mllvm -unroll-runtime -march=native -ffast-math" -largeArrayDims predict.c linear_model_matlab.c ../l2r_l2_primal_fun.cpp ../l2r_huber_primal_fun.cpp ../linear.cpp ../SAG.cpp "../blas/*.c"
	end
catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
