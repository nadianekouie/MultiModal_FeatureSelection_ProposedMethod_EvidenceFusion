clc
disp('Nemeyni Test');

% rank_ave from Friedman test
rank_ave = [5.75 6.85	7.45 4.95	6.75	6.50	7.90	8.15	6.20	8.40	4.00	1.65];
[~,k] = size(rank_ave);
n = input('Enter Size of Dataset: ');
confidence_level = input('Enter Confidence Level(just 0.9 or 0.95) : ');
q = 0;
if(confidence_level == 0.9)
    switch k
        case 2
            q=1.645;
        case 3
            q=2.052;
        case 4
            q=2.291;
        case 5
            q=2.459;
        case 6
            q=2.589;
        case 7
            q=2.693;
    end
elseif(confidence_level == 0.95)
    switch k
        case 2
            q=1.960;
        case 3
            q=2.343;
        case 4
            q=2.569;
        case 5
            q=2.728;
        case 6
            q=2.850;
        case 7
            q=2.693;
    end
else
    disp('Bad Input!')
    return;
end;

CD = q * sqrt ( (k*(k+1))/(6*n) );
for i = 1:k
    for j = i+1 : k
        if(rank_ave(i)-rank_ave(j) < CD)
            disp('Method ')
            disp(i);
            disp(' is Better than ')
            disp(j);
            disp(' With Confidence = ')
            disp(confidence_level*100)
            disp(' %');
        elseif(rank_ave(j)-rank_ave(i) < CD)
            disp('Method ')
            disp(j);
            disp(' is Better than ')
            disp(i);
            disp(' With Confidence = ')
            disp(confidence_level*100)
            disp(' %');
        end  
    end
end
