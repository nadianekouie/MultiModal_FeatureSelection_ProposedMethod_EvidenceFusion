function [ACC ,Fscore,Sensitivity ,Specificity,Precision] = Evaluator(Y , Yp)

K = length(unique(Y));

for i = 1 : K
    YY = Y == i;
    YYp = Yp == i;
    
    TP = sum(YY == 1 & YYp == 1);
    TN = sum(YY == 0 & YYp == 0);
    FP = sum(YY == 0 & YYp == 1);
    FN = sum(YY == 1 & YYp == 0);
    
    ACC(i) = (TP + TN) / (TP + TN + FP + FN + eps);
    Fscore(i) = 2*TP / (2*TP + FP + FN + eps);
    Sensitivity(i) = TP / (TP + FN + eps); % recall
    Specificity(i) = TN / (TN + FP + eps);
    Precision(i) = TP / (TP + FP + eps);    
end

disp(['Accuracy: ',num2str(mean(ACC)*100) ,' %']);
disp(['Fscore: ',num2str(mean(Fscore)*100) ,' %']);
disp(['Sensitivity: ',num2str(mean(Sensitivity)*100) ,' %']);
disp(['Specificity: ',num2str(mean(Specificity)*100) ,' %']);
disp(['Precision: ',num2str(mean(Precision)*100) ,' %']);


ACC = mean(ACC);
Fscore = mean(Fscore);
Sensitivity = mean(Sensitivity);
Specificity = mean(Specificity);
Precision = mean(Precision);


end

