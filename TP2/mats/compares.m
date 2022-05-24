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

grade  = (100 - mean(abs(MediaErros(4:end)))) * 20 / 100;

save comparison.mat