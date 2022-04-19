close all
clear all


A_exact160 = readtable("tp1_seq_160_results.txt");
A_exact530 = readtable("tp1_seq_530_results.txt");


% A_exact160.VariableNames = ["NumMec", "NumSeq", "NumImg", "tDom", "tDice", "tCard", "RDO",   RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT]



B = readtable("tp1_92993.txt");

B160 = B(1:40,:);
B530 = B(41:end,:);




Diff160 = table2array(B160)-table2array(A_exact160);

Diff530 = table2array(B530)-table2array(A_exact530);

Erro160 = Diff160./table2array(A_exact160)*100;

Erro530 = Diff530./table2array(A_exact530)*100;


MediaErros160 = mean(Erro160,1);
MediaErrosImp160 = mean(Erro160(:,2:end-1),1);

MediaErros530 = mean(Erro530,1);
MediaErrosImp530 = mean(Erro530(:,2:end-1),1);

% T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
% Tcut = table(NumSeq, NumImg, tDom, tDice, tCard, RDO, RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros);

save comparison.mat