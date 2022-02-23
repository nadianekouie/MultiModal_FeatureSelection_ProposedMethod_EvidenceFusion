clc
clear
close all

maxRun = 10;

for run = 1 : maxRun
    
    rng(run);
    %% Load dataset
    dsname = 'llgse1577';
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
    
    tic;
    %% Feature selction using FFO
    global Tr
    Tr = 0.95;
    BestSol = FireFly_MultiModal(d);
    selectedFeatures = find(BestSol > Tr);
    
    Xtr = Xtr( : , selectedFeatures);
    Xts = Xts( : , selectedFeatures);
    
    disp(['Number of features after reduction: ',num2str(size(Xtr,2))]);
    
    %% Train a classification model
    model = fitcensemble(Xtr , Ytr);
    
    %% Evaluate trained model over test data
    Yp = predict(model , Xts);
    
    Time(run) = toc;
    %% Display results
    [ACC(run) ,Fscore(run),Sensitivity(run),Specificity(run),Precision(run)] = Evaluator(Yts , Yp);
    
end

A = cell(9,maxRun + 3);
A{1,1} = 'Method';
A{1,2} = 'MultiModal_FeatureSelection';
A{2,1} = 'dataset';
A{2,2} = dsname;

A{4,1} = 'Accuracy (%)';
A{5,1} = 'Fscore (%)';
A{6,1} = 'Sensitivity (%)';
A{7,1} = 'Specificity (%)';
A{8,1} = 'Precistion (%)';
A{9,1} = 'Time (s)';

for run = 1 : maxRun
    A{3,run+1} = ['run',num2str(run)];
    A{4,run+1} = ACC(run)*100;
    A{5,run+1} = Fscore(run)*100;
    A{6,run+1} = Sensitivity(run)*100;
    A{7,run+1} = Specificity(run)*100;
    A{8,run+1} = Precision(run)*100;
    A{9,run+1} = Time(run);
end

A{3,maxRun+2} = 'Average';
A{4,maxRun+2} = mean(ACC)*100;
A{5,maxRun+2} = mean(Fscore)*100;
A{6,maxRun+2} = mean(Sensitivity)*100;
A{7,maxRun+2} = mean(Specificity)*100;
A{8,maxRun+2} = mean(Precision)*100;
A{9,maxRun+2} = mean(Time);

xlswrite(['MultiModal_FeatureSelection_',dsname,'.xls'],A);

