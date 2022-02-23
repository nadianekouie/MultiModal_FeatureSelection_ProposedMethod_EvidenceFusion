function Yp = getWeightedVoteLabels(Labels , ClassifierWeight)

numClassifier = length(ClassifierWeight);
numModal = size(Labels , 1) / numClassifier;

ClassifierWeight = repmat(ClassifierWeight' , numModal , 1);
ClassifierWeight = floor(ClassifierWeight * 1000);
ClassifierWeight(ClassifierWeight == 0) = 1; % for zeros weighted classifiers

for t = 1 : size(Labels , 2)
    all_label = [];
    for i = 1 : numClassifier * numModal
        l = Labels(i , t);
        l = repmat(l , ClassifierWeight(i) , 1);
        all_label = [all_label ; l]; 
    end
    
    Yp(t) = mode(all_label);
    
end


end

