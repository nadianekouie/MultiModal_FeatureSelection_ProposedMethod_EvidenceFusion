clc
clear
close all

maxRun = 10;

for run = 1 : maxRun
    
    rng(run);
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
    
    Yp = mode(PredictedLabels)';
    
    Time(run) = toc;
    %% Display results
    [ACC(run) ,Fscore(run),Sensitivity(run),Specificity(run),Precision(run)] = Evaluator(Yts , Yp);
    
end

A = cell(9,maxRun + 3);
A{1,1} = 'Method';
A{1,2} = 'MultiModal_FeatureSelection_Ensemble';
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

xlswrite(['MultiModal_FeatureSelection_Ensemble_',dsname,'.xls'],A);

