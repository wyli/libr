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
		mex CFLAGS="-march=native -O3 -DNDEBUG" -largeArrayDims libsvmread.c
		mex CFLAGS="-O2 -DNDEBUG" -largeArrayDims libsvmwrite.c
		mex CFLAGS="-O2 -DNDEBUG" -largeArrayDims train.c linear_model_matlab.c ../l2r_l2_primal_fun.cpp ../l2r_huber_primal_fun.cpp ../linear.cpp ../SAG.cpp "../blas/*.c"
		mex CFLAGS="-O2 -DNDEBUG" -largeArrayDims predict.c linear_model_matlab.c ../l2r_l2_primal_fun.cpp ../l2r_huber_primal_fun.cpp ../linear.cpp ../SAG.cpp "../blas/*.c"
	end
catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
