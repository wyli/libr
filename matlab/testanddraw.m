function mo = testanddraw( samples, targets, n, type, cost, c)
%TESTANDDRAW Summary of this function goes here
%   Detailed explanation goes here
global mind
global maxd
figure1 = figure;

%rind = randsample(size(targets, 1), n);
%targets(1) = 1.6;
%targets(end) = -1.6;
scaledSamples = scaledata(samples, 1);
%scaledSamples = [scaledSamples, ones(size(samples, 1), 1)*C_1]; 
if n > 0
% rind = 1:n;
%targets(rind) = targets(rind) * 2;
targets(end) = targets(end) * 2;
targets(end-1) = targets(end-1) * 2;
%scaledSamples(rind, end) = 1; % adding end of the matrix for C_n
%scaledSamples(rind, end) = scaledSamples(rind, end) * C_2;
end
sprintf('-s %f -e %.10f', type, cost)
mo = train(sparse(targets), sparse(scaledSamples), sprintf('-s %d -e %.10f -c %f', type, cost, c));


% evaluate svm on meshgrid in order to visualise decision boundary
[x1 x2] = meshgrid(-10:0.5:15, -10:0.5:15);
y = zeros(size(x1));
for i = 1:size(x1, 1)
    for j = 1:size(x2, 1)
        r(i, j) = predict(sparse(0), sparse(scaledata([x1(i, j) x2(i, j)], 0)), mo, '-q')>0;
    end
end
axes1 = axes('parent', figure1);

contour(x1, x2, r, 1,  'Color', [0 0 0.75]);
colormap gray
hold on;
% plot samples
plot(samples(targets==0.8, 1), samples(targets ==0.8, 2), 'o', ...
    'Color', [1 0 0], 'linewidth', 3, 'MarkerSize', 7);
hold on;
plot(samples(targets==-0.8, 1), samples(targets==-0.8, 2), 'o', ...
    'Color', [0 1 0], 'linewidth', 3, 'MarkerSize', 7);
hold on;

plot(samples(targets==1.6, 1), samples(targets ==1.6, 2), 'rs', ...
    'Color', [1 0 0], 'linewidth', 2, 'MarkerSize', 12);
hold on;
plot(samples(targets==-1.6, 1), samples(targets==-1.6, 2), 'rs', ...
    'Color', [0 1 0], 'linewidth', 2, 'MarkerSize', 12);
hold on;


xlim(axes1, [-10, 15]);
ylim(axes1, [-10, 10]);
% plot decision boundary



% plot support vectors
%sv = samples(:, smo.alpha>0);
%for i = 1:size(sv, 2)
%    plot(sv(1,i), sv(2,i), 's', 'linewidth', 0.5,'markersize', 12, 'Color', [0.2, 0.2, 0.2]);
%end
%print('-dpsc', ['C/desktop/eps/', sprintf('%03d', n), '-', int2str(k), '-', 'test.eps']); 
end
function [scaled, meanv] = scaledata(data, newdata)
global mind
global maxd
if newdata > 0
mind = min(data,[],1);
maxd = max(data,[],1);
end
scaled = (data - repmat(mind,size(data,1),1))*spdiags(1./(maxd-mind)',0,size(data,2),size(data,2));

end
