clc
clear


fm = csvread('fm.csv');
m1 = find(fm(: , 1) == 1);
m2 = find(fm(: , 2) == 1);
m3 = find(fm(: , 3) == 1);

disp('intersect of three modal: %');
T = intersect(intersect(m1 , m2) , m3);
U = union(union(m1 , m2) , m3);
100*length(T)/ length(U)


disp('intersect of  m1 , m2:');
100*length(setdiff(intersect(m1 , m2),T))/length(U)

disp('intersect of  m1 , m3:');
100*length(setdiff(intersect(m1 , m3),T))/length(U)

disp('intersect of  m2 , m3:');
100*length(setdiff(intersect(m2 , m3),T))/length(U)

disp('len only m1:');
100*length(setdiff(m1 , [m2;m3]))/length(U)

disp('len only m2:');
100*length(setdiff(m2 , [m1;m3]))/length(U)

disp('len only m3:');
100*length(setdiff(m3 , [m1;m2]))/length(U)

