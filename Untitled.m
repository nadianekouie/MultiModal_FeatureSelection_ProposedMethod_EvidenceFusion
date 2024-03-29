clc
disp('Friedman Test');
x = [0.886 0.763 0.906 0.585 0.893 0.868 0.776 0.898 0.947 0.743; 
     0.859 0.789 0.907 0.675 0.786 0.752 0.737 0.923 0.923 0.791;
     0.818 0.700 0.891 0.692 0.840 0.854 0.742 0.907 0.929 0.793;
     0.845 0.742 0.915 0.636 0.886 0.854 0.757 0.953 0.947 0.803;
     0.863 0.789 0.894 0.665 0.800 0.864 0.721 0.884 0.936 0.783;
     0.768 0.742 0.915 0.621 0.833 0.850 0.767 0.953 0.918 0.811;
     0.827 0.647 0.901 0.724 0.833 0.746 0.750 0.939 0.894 0.758;
     0.827 0.763 0.904 0.700 0.780 0.826 0.777 0.913 0.907 0.770;
     0.904 0.742 0.742 0.590 0.853 0.858 0.785 0.925 0.910 0.746;
     0.859 0.731 0.890 0.658 0.806 0.846 0.749 0.919 0.892 0.780;
     0.822 0.822 0.822 0.874 0.874 0.874 0.874 0.961 0.934 0.823;
     0.913 0.831 0.962 0.678 0.853 0.946 0.830 0.961 0.983 0.826];

x=x';

[n,k] = size(x);



x_rank = zeros(n,k);
for i=1:n
    if(range(x(i,:))==0)
        x_rank(i,:) = (k+1)/2;  % k(k+1)/2/k => (k+1)/2
        continue;
    end;
    
    areEqual =1;
    for j=1:k
        if(x(i,j) ~= x(i,1))
            areEqual =0;
        end;
    end;
    if(areEqual == 1)
        
        [~, ~, x_rank(i,:)] = unique(x(i,:));
        continue;
    end;
    
    r=1;
    for j=1:k
        max = -1;
        Imax=0;
        for u=1:k
            if(x_rank(i,u) ==0 && x(i,u)> max)
                max =x(i,u);
                Imax = u;
            end;
        end;
     
        if(Imax==0)
            break;
        end;
        findtwoEq=0;
        temp = max;
        for jj = 1:k
            if(x_rank(i,j) ==0 && temp == x(i,jj) && jj~=Imax)
                x_rank(i,Imax)=(2*r+1)/2;
                x_rank(i,jj)=(2*r+1)/2;
                r=r+2;
                findtwoEq=1;
                break;
            end;
        end;
        if(findtwoEq ==0)
            x_rank(i,Imax) = r;
            r = r+1;
        end;
                
    end;
end;
%x_rank;

rank_ave = mean(x_rank(:,1:k));
sumsqrank=0;
for i=1:k
    sumsqrank = sumsqrank + rank_ave(i)*rank_ave(i);
end;

chi2 = 12*n/(k*(k+1))*(sumsqrank - k*(k+1)*(k+1)/4);

p=chi2cdf(chi2,k-1);
alpha = 0.95;
if(p >= alpha)
    disp('This Classifiers are Significantly Different With');
    alpha
end;
disp('Real Significantly Confidence Degree is:');
p







