function fit = FitnessFunc(x)

% global Xtr Ytr
global newsol

if nargin == 0
    x = newsol.Position;
end


global Tr
% dim = length(x);

global FeatureRel

% fit = 0;
% for i = 1 : dim
%     if x(i) > Tr
%         % Mutual Information
% %         MI = mi(Xtr(: , i) , Ytr);
% %         fit = fit + MI;
% 
% %         CC = corr(Xtr(: , i) , Ytr);
%         CC = FeatureRel(i);
%         fit = fit + abs(CC);
%         
%     end
% end

fit = sum(FeatureRel(x > Tr));

fit = 1 / (fit + eps);

fit = fit + 0.01 * sum(x > Tr); % todo: adaptive weight

end

