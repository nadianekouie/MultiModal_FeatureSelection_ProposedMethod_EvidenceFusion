function fit = Evaluation_Func_Ensemble(x)

% convert to binary form
x = x > 0.5;
n = sum(x);

global Xts2 Yts2 Models BestFireFlies Tr
num_test = size(Xts2 , 1);

Labels = zeros(n , num_test);
cnt = 1;

for i = 1 : length(x)
    if x(i) == 1
       iModal = floor((i-1) / 7) + 1;
%        iClassifier = mod( i , 7);
       BestSol = BestFireFlies(iModal).Position;
       selectedFeatures = (BestSol > Tr);
       Yp = predict(Models{i} , Xts2(: , selectedFeatures));
       Labels(cnt , :) = Yp';
       cnt = cnt + 1;
    end
end

Yp_mv = mode(Labels)';
Accuracy = sum(Yts2 == Yp_mv) / num_test;

fit = 1 - Accuracy; % error rate


end

