clc
clear
close all

rng(0);

%% Load dataset
dsname = 'Leukemia';
[X , Y] = LoadDataset(dsname);
[N , d] = size(X);
numClass = length(unique(Y));
%% Display dataset properties
disp(['Dataset name: ',dsname]);
disp(['Number of data: ',num2str(N)]);
disp(['Number of dimention: ',num2str(d)]);
disp(['Number of class: ',num2str(numClass)]);

%% Train & test split
global Xtr Ytr
trainPercent = 0.7;
[Xtr , Ytr , Xts , Yts] = TrainTestSplit(X , Y , trainPercent);
Ntr = size(Xtr , 1);
Nts = size(Xts , 1);
disp(['Number of train data: ',num2str(Ntr)]);
disp(['Number of test data: ',num2str(Nts)]);

%% Compute relevence between features and class label
disp('Computing feature relevence');
global FeatureRel
FeatureRel = ComputeFeatureRelevence(Xtr , Ytr);
disp('End of computing feature relevence');

%% Feature selction using FFO
global Tr
Tr = 0.95;
BestSol = FireFly_MultiModal(d);
selectedFeatures = find(BestSol > Tr);

Xtr = Xtr( : , selectedFeatures);
Xts = Xts( : , selectedFeatures);

disp(['Number of features after reduction: ',num2str(size(Xtr,2))]);

%% Train a classification model
model = fitcknn(Xtr , Ytr);

%% Evaluate trained model over test data
Yp = predict(model , Xts);

%% Display results
Evaluator(Yts , Yp);