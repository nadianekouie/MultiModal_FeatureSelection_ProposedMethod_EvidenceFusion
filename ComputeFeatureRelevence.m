function FeatureRel = ComputeFeatureRelevence(Xtr , Ytr)
d = size(Xtr , 2);
FeatureRel = zeros(d , 1);
for i = 1 : d
   FeatureRel(i) = abs(corr(Xtr(: , i) , Ytr));
end