 close all;
 accs = zeros(length(3:-1:-5));
 aucs = zeros(length(3:-1:-5));
 i = 0;
 j = 0;
 for log10c = 3:-1:-5
     i = i + 1;
     j = 0;
     for log10p = 3:-1:-5
         j = j + 1;
        load('~/desktop/small');
        cmd = ['-s 2 -e 0.0001 -c ', num2str(10^log10c), ' -p ', num2str(10^log10p)]
        trainModel = train(sparse(yy), sparse(scaledFeatures), cmd);
        t = (yy>0)*2 - 1;
        [~, acc, scores] = predict(sparse(t),sparse(scaledFeatures), trainModel);
        [~, ~, ~, auc] = perfcurve(t, scores, 1);
        aucs(i, j) = auc;
        accs(i, j) = acc(1);
     end
 end
 imagesc(accs); colormap(gray);
