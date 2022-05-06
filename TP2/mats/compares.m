close all
clear all


A_exact = readtable("svpi2022_tp2_seq_ALL.txt");

B = readtable("tp2_92993.txt");

Diff = table2array(B)-table2array(A_exact);
Erro = Diff./table2array(A_exact)*100;
MediaErros = mean(Erro,1);
MediaErrosImp = mean(Erro(:,2:end-1),1);

% T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
% Tcut = table(NumSeq, NumImg, tDom, tDice, tCard, RDO, RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros);

save comparison.mat