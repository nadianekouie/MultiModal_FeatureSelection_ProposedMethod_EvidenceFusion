clc
clear
close all

dsname = 'leukemia';
maxRun = 10;

start_run = 1;

if start_run > 1
    filename = ['log/MultiModal_FeatureSelection_ProposedMethod_EvidenceFusion_' , dsname , '_run_' , num2str(start_run - 1)];
    load(filename);
end

for run = start_run : maxRun
    
    rng(run);
    
    %% Load dataset
    
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
    
    tic;
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
    
    
    
    trainPercent = 0.5;
    [Xtr2 , Ytr2 , Xval , Yval] = TrainTestSplit(Xtr , Ytr , trainPercent);
    
    num_classifier = 7;
    EV = getEvidenveMatrix(BestFireFlies , Xtr2 , Ytr2 , Xval , Yval);
    
    CM = pdist2(EV' , EV' , 'minkowski'); % conflict matrix
    SM = 1 - CM; % similarity matrix
    Sup = sum(SM , 2) - 1; % support degree of each
    CR = Sup / sum(Sup); % Credibility degree
    q = 2; % When q->1, Tsallis entropy converges to Shannon entropy, when q->2, Tsallis entropy is equivalent to the Gini index
    Infos = (1 / (q - 1)) * (1 - sum(CM .^ q , 2));
    Infos = Infos ./ max(Infos);
    
    Weight = CR ./ Infos; % eq.14
    m = Weight' .* EV;
    
    P = prod(m , 1);
    K = sum(P); % eq. 17
    
    ClassifierWeight = P ./ K; % eq. 16
    
    figure
    hold on
    for i = 1 : size(EV , 1)
        plot(EV(i , :),'linewidth',2);
    end
    CC = cell(1,7);
    CC{1} = 'KNN, k = 1';
    CC{2} = 'KNN, k = 3';
    CC{3} = 'KNN, k = 5';
    CC{4} = 'Ecoc SVM';
    CC{5} = 'Naive bayes';
    CC{6} = 'Tree';
    CC{7} = 'Random forest';
    xticklabels(CC);
    ylabel('Accuracy');
    legend('Modal 1','Modal 2','Modal 3','Modal 4','Modal 5');
    axis([0 8 , 0 , 1]);
    grid  on
    
    %% Ensemble
    Models = TrainModels(BestFireFlies , Xtr , Ytr);
    [~ , Labels] = PredictEnsemble(ones(num_classifier*length(BestFireFlies),1) , Xts);
    
    Yp = getWeightedVoteLabels(Labels , ClassifierWeight);
    
    
    Time(run) = toc;
    %% Display results
    [ACC(run) ,Fscore(run),Sensitivity(run),Specificity(run),Precision(run)] = Evaluator(Yts , Yp');
    
    filename = ['log/MultiModal_FeatureSelection_ProposedMethod_EvidenceFusion_' , dsname , '_run_' , num2str(run)];
    save(filename ,'ACC' , 'Fscore','Sensitivity','Specificity','Precision','Time');
    
end

A = cell(9,maxRun + 3);
A{1,1} = 'Method';
A{1,2} = 'MultiModal_FeatureSelection_ProposedMethod_EvidenceFusion';
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

xlswrite(['MultiModal_FeatureSelection_ProposedMethod_EvidenceFusion_',dsname,'.xls'],A);

