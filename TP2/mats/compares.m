close all
clear all

% A_exact = readtable("svpi2022_tp2_seq_ALL.txt");
A_exact = readtable("svpi2022_tp2_seq_291.txt");

B = readtable("tp2_92993.txt");

Diff = table2array(B)-table2array(A_exact);
At = table2array(A_exact);


Erro = Diff./At*100;
Erro(At==0) = 100; % remove Infs
Erro(Diff==0) = 0; 

MediaErros = mean(abs(Erro),1);

grade1  = (100 - mean(abs(MediaErros(4:end)))) * 20 / 100;

idx = 1;
Erro2 = zeros(18,1);
for col = 4:20
    Erro2(idx) = sum(abs(Diff(:,col)))./sum(At(:,col));
    idx = idx + 1;
end
grade2  = (1- mean(abs(Erro2))) * 20;

save comparison.mat