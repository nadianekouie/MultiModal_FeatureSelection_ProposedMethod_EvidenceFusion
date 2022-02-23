function [X , Y] = LoadDataset(dsname)

switch lower(dsname)
    case 'colon'
        X = xlsread('dataset\ColonX.csv');
        X(1 , :) = [];
        X(: , 1) = [];
        
        Y = xlsread('dataset\ColonY.csv');
        Y(: , 1) = [];
        
    case 'leukemia'
        X = xlsread('dataset\Leukemia_X.csv');
        Y = xlsread('dataset\Leukemia_Y.csv');
        
    case 'lung'
        load('dataset\lung.mat');
        X = lung;
        Y = cell(size(X,1),1);
        for i = 1 : size(X,1)
            y = cell2mat(Classlabels{i});
            Y{i} = y; 
        end
        
        Y = grp2idx(Y);
        
    case 'prostate'
        load('dataset\Prostate.mat');
        
    case 'srbct'
        load('dataset\SRBCT.mat');
        
    case 'braintumor40'
        load('dataset\BrainTumor40.mat');
        
    case 'braintumor'
        load('dataset\braintumor.mat');
        
    case 'mll'
        load('dataset\MLL.mat');
        
    case 'gastricgse2685'
        load('dataset\gastricGSE2685.mat');
        
    case 'llgse1577'
        load('dataset\LLGSE1577.mat');
        
    case 'dlbcl'
        load('dataset\DLBCL47Main.mat');
        X = knnimpute(X')';
    
end

end

