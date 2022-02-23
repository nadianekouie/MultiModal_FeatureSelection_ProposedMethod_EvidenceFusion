clc
clear
close all

rng(0);

%% Load dataset
dsname = 'Leukemia';
[X , Y] = LoadDataset(dsname);
[N , d] = size(X);
numClass = length(unique(Y));

X = (X - min(X)) ./ (max(X) - min(X) + eps);

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
global Models BestFireFlies
global Tr
Tr = 0.95;
BestFireFlies = FireFly_MultiModal_ProposedMethod(d);

for i = 1 : length(BestFireFlies)
    NumF = sum(BestFireFlies(i).Position > Tr);
    disp(['Number of features in modal ',num2str(i),' is: ',num2str(NumF)]);
    
    FeatureModal(: , i) = BestFireFlies(i).Position > Tr;
end

%% Plot modal
fidx = find(sum(FeatureModal , 2) ~= 0);
bar(FeatureModal(fidx , :),'stacked')
leg = cell(length(BestFireFlies) , 1);
for i = 1 : length(BestFireFlies)
    leg{i} = ['modal ',num2str(i)];
end
legend(leg);
xticks(1:length(fidx))
xticklabels(fidx);
xtickangle(90);


%% Search for the optimal ensemble using PSO
% KNN k = 1
% KNN k = 3
% KNN k = 5
% L-SVM
% Naive Bayes
% Tree C4.5
% Random forest

global Xts2 Yts2

trainPercent = 0.5;
[Xtr2 , Ytr2 , Xts2 , Yts2] = TrainTestSplit(Xtr , Ytr , trainPercent);
num_classifier = 7;
disp('Training models');
Models = TrainModels(BestFireFlies , Xtr2 , Ytr2);
disp('End of Training models');

disp('PSO search for optimal ensemble');
nvar = length(BestFireFlies) * num_classifier;
options = optimoptions('particleswarm','SwarmSize',50,'Display','iter');
BestEnsemble = particleswarm(@Evaluation_Func_Ensemble , nvar, zeros(nvar,1) , ones(nvar,1),options);
disp('End of PSO');

% [ 0 0 0 1 0 0 1 0 0 0 0 0 0 ]

%% Evaluate trained model over test data
Yp = PredictEnsemble(BestEnsemble , Xts);

%% Display results
Evaluator(Yts , Yp);

