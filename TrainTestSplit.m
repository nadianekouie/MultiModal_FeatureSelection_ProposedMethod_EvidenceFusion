function [Xtr , Ytr , Xts , Yts] = TrainTestSplit(X , Y , trainPercent)

N = size(X , 1);
ridx = randperm(N);

X = X(ridx , :);
Y = Y(ridx);

Xtr = X(1 : floor(N * trainPercent) , :);
Ytr = Y(1 : floor(N * trainPercent) );

Xts = X(1 + floor(N * trainPercent) : end , :);
Yts = Y(1 + floor(N * trainPercent) : end );

end

