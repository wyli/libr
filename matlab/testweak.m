
close all; clear all;
% Gaussian c
numOfC = 8;
muC = [2, 3.5];
sigmaC = [9 1.5; 1.5 8];
dataC = mvnrnd(muC, sigmaC, numOfC);

% Gaussian d
%numOfD = 50;
%muD = [4, 1];
%sigmaD = [6 3.5; 3.5 3];
%dataD = mvnrnd(muD, sigmaD, numOfD);

% Gaussian H
numOfH = 8;
muH = [10, 0];
sigmaH = [3 1.5; 1.5 8];
dataH = mvnrnd(muH, sigmaH, numOfH);

%samples = [dataD; dataH; dataC];
%targets = [ones(numOfD, 1)*(.8); ones(numOfH, 1)*0.8; -0.8*ones(numOfC,1)];
samples = [dataH; dataC];
targets = [ones(numOfH, 1)*0.8; -0.8*ones(numOfC,1)];

truesamples = [2, 3.5; 10, 0];
samples = [samples; truesamples];
targets = [targets; -0.8; 0.8];

C_L = 0.5;
C_H = 0.5;
%for n = 1:12:60
n = 70;
x = -5:1:5;
for n = [0,2]
    if n == 0;
        c = 0;
    else
        c = 10;
    end
    r = testanddraw(samples, targets, n, 2, 0.00001, c);%normal and high
end
% [R,C] = ndgrid(-10:0.2:15, -10:0.2:15);
% mat = gaussC(R,C, 4, muC);
% mat2 = gaussC(R,C, 3, muD);
% mat3 = gaussC(R,C,3, muH);
% mat = mat - mat2 - mat3;
% figure();
% imagesc(mat);
% colormap(bone);
