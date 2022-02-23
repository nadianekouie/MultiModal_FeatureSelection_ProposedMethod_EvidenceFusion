function BestFireFlies = FireFly_MultiModal_Ensemble(dim)

addpath('./MIToolbox/matlab/');

%% Problem Definition

numModal = 5;

nVar = dim;                 % Number of Decision Variables

VarSize = [1 nVar];       % Decision Variables Matrix Size

VarMin = 0;             % Decision Variables Lower Bound
VarMax = 1;             % Decision Variables Upper Bound

%% Firefly Algorithm Parameters

MaxIt = 2;            % Maximum Number of Iterations

nPop = 50;            % Number of Fireflies (Swarm Size)

gamma = 1;            % Light Absorption Coefficient

beta0 = 1;            % Attraction Coefficient Base Value

alpha = 0.2;          % Mutation Coefficient

alpha_damp = 0.98;    % Mutation Coefficient Damping Ratio

delta = 0.05*(VarMax-VarMin);     % Uniform Mutation Range

m = 2;

if isscalar(VarMin) && isscalar(VarMax)
    dmax  =  (VarMax-VarMin)*sqrt(nVar);
else
    dmax  =  norm(VarMax-VarMin);
end

%% Initialization

% Empty Firefly Structure
firefly.Position = [];
firefly.Cost = inf;

% Initialize Population Array
pop = repmat(firefly,nPop,1);

% Create Initial Fireflies
positions = zeros(nPop , nVar);
for i = 1 : nPop
    pop(i).Position = unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost = FitnessFunc(pop(i).Position);
    
    positions(i,:) = pop(i).Position;
end

% clustering
[~ , U] = fcm(positions , numModal);

[~ , Imax] = max(U);

for i = 1 : numModal
    disp(['Number of populations in modal ',num2str(i),' is: ',num2str(sum(Imax == i))]);
end

% run firefly in each cluster
BestFireFlies = repmat(firefly,numModal,1);
% Array to Hold Best Cost Values
% BestCost = zeros(MaxIt,numModal);
%% Firefly Algorithm Main Loop
for it = 1 : MaxIt
    
    disp(['iter ',num2str(it),' from ',num2str(MaxIt)]);
    
    for modal = 1 : numModal
        disp(['modal ',num2str(modal),' from ',num2str(numModal)]);

        nPopModal = sum(Imax == modal);
        if nPopModal == 0
            BestFireFlies(modal).Cost = inf;
            continue;
        end
        
        popModal = pop(Imax == modal);
        [~ , Imin] = min([popModal.Cost]);
        BestSol.Cost = popModal(Imin).Cost;
        BestSol.Position = popModal(Imin).Position;
        
        newpop = repmat(firefly,nPopModal,1);
        B = 0;
        for i = 1:nPopModal
            newpop(i).Cost  =  inf;
            for j = 1:nPopModal
                if popModal(j).Cost < popModal(i).Cost
                    rij = norm(popModal(i).Position-popModal(j).Position)/dmax;
                    beta = beta0*exp(-gamma*rij^m);
                    e = delta*unifrnd(-1,+1,VarSize);
                    %e = delta*randn(VarSize);
                    
                    global newsol
                    
                    newsol.Position  =  popModal(i).Position ...
                        + beta*rand(VarSize).*(popModal(j).Position-popModal(i).Position) ...
                        + alpha*e;
                    
                    newsol.Position = max(newsol.Position,VarMin);
                    newsol.Position = min(newsol.Position,VarMax);
                    
                    newsol.Cost = FitnessFunc();
                    
                    if newsol.Cost < newpop(i).Cost
                        B = B + 1;
                        newpop(i)  =  newsol;
                        if newpop(i).Cost <= BestSol.Cost
                            BestSol = newpop(i);
                        end
                    end
                    
                end
            end
        end
        
        % Merge
        popModal = [popModal
            newpop];  %#ok
        
        % Sort
        [~, SortOrder] = sort([popModal.Cost]);
        popModal = popModal(SortOrder);
        
        % Truncate
        popModal = popModal(1:nPopModal);
        
        % Check stabality of current modal
        if B == 0
            disp(['population ',num2str(modal),' is not stable']);
            disp('regenerating new population');
            Pos = getNewRandomPopulation(nPopModal , VarSize);
            for i = 1 : length(popModal)
                popModal(i).Position = Pos(i,:);
            end
        end
        
        
        pop(Imax == modal) = popModal;
        
        %         % Store Best Cost Ever Found
        %         BestCost(it) = BestSol.Cost;
        
        % Show Iteration Information
        %         disp(['Iteration ' num2str(it) ': Best Cost  =  ' num2str(BestCost(it))]);
        
        if BestFireFlies(modal).Cost > BestSol.Cost
            BestFireFlies(modal)  =  BestSol;
        end
        
    end
    
    % Damp Mutation Coefficient
    alpha = alpha * alpha_damp;
    
end

% 
% [~ , Imin] = min([BestFireFlies.Cost]);
% BestPosition = BestFireFlies(Imin).Position;

% %% Results
% figure;
% %plot(BestCost,'LineWidth',2);
% semilogy(BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;

end

