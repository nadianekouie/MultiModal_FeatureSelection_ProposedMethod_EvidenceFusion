function popModal = getNewPopulation_ProposedMethod(nPopModal , VarSize , pop)

VarSize = VarSize(2);
numCondidate = 1000;
condidatePop = unifrnd(0 , 1 , numCondidate , VarSize);

Pos = zeros(length(pop) , VarSize);
for i = 1 : length(pop)
   Pos(i , :) = pop(i).Position;
end

dist = pdist2(condidatePop , Pos);
dist = min(dist , [] , 2);

[~ , Imax] = max(dist);
winerPoint = condidatePop(Imax , :);

% Cov = eye(VarSize) .* (1/12)^2;

% popModal = mvnrnd(winerPoint , Cov , nPopModal);
popModal = winerPoint + randn(nPopModal,length(winerPoint)) / 12;


end

