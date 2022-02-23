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
BestFireFlies = FireFly_MultiModal_Ensemble(d);

for i = 1 : length(BestFireFlies)
    NumF = sum(BestFireFlies(i).Position > Tr);
    disp(['Number of features in modal ',num2str(i),' is: ',num2str(NumF)]);
    
    FeatureModal(: , i) = BestFireFlies(i).Position > Tr;
end

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


Xtr_copy  = Xtr;
Xts_copy = Xts;
for modal = 1 : length(BestFireFlies)
    
    BestSol = BestFireFlies(modal).Position;
    selectedFeatures = find(BestSol > Tr);

    Xtr = Xtr_copy( : , selectedFeatures);
    Xts = Xts_copy( : , selectedFeatures);

    disp(['Number of features after reduction: ',num2str(size(Xtr,2))]);

    %% Train a classification model
    model = fitcknn(Xtr , Ytr);
    
    %% Evaluate trained model over test data
    Yp = predict(model , Xts);
    
    PredictedLabels(modal , :) = Yp';
    
    Accuracy = sum(Yp == Yts) ./ Nts;
    disp(['Accuracy on modal ',num2str(modal),' is: ',num2str(Accuracy * 100) , ' %']);
end

MV_label = mode(PredictedLabels)';

%% Display results
Evaluator(Yts , Yp);