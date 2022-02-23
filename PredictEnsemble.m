function [Yp,Labels] = PredictEnsemble(x , Xts)

% convert to binary form
x = x > 0.5;
n = sum(x);

global Models BestFireFlies Tr
num_test = size(Xts , 1);

Labels = zeros(n , num_test);
cnt = 1;

for i = 1 : length(x)
    if x(i) == 1
       iModal = floor((i-1) / 7) + 1;
%        iClassifier = mod( i , 7);
       BestSol = BestFireFlies(iModal).Position;
       selectedFeatures = (BestSol > Tr);
       Yp = predict(Models{i} , Xts(: , selectedFeatures));
       Labels(cnt , :) = Yp';
       cnt = cnt + 1;
    end
end

Yp = mode(Labels)';



end

