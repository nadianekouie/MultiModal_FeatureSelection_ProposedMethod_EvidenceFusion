function Models = TrainModels(BestFireFlies , Xtr2 , Ytr2)
% KNN k = 1
% KNN k = 3
% KNN k = 5
% L-SVM
% Naive Bayes
% Tree C4.5
% Random forest

global Tr

Models = cell(length(BestFireFlies) * 7 , 1);

cnt = 1;
for modal = 1 : length(BestFireFlies)
    for j = 1 : 7
        BestSol = BestFireFlies(modal).Position;
        selectedFeatures = (BestSol > Tr);
        if j == 1
            model = fitcknn(Xtr2(: , selectedFeatures) , Ytr2);
        elseif j == 2
            model = fitcknn(Xtr2(: , selectedFeatures) , Ytr2,'NumNeighbors',3);
        elseif j == 3
            model = fitcknn(Xtr2(: , selectedFeatures) , Ytr2,'NumNeighbors',5);
        elseif j == 4
            model = fitcecoc(Xtr2(: , selectedFeatures) , Ytr2);
        elseif j == 5
            try
                model = fitcnb(Xtr2(: , selectedFeatures) , Ytr2);
            catch
                model = fitcecoc(Xtr2(: , selectedFeatures) , Ytr2);
            end
        elseif j == 6
            model = fitctree(Xtr2(: , selectedFeatures) , Ytr2);
        else
            model = fitcensemble(Xtr2(: , selectedFeatures) , Ytr2);
        end
        
        Models{cnt} = model;
        cnt = cnt + 1;
        
    end
end

end

