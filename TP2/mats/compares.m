close all
clear all

A_exact = readtable("svpi2022_tp2_seq_ALL.txt");

B = readtable("tp2_92993.txt");

Diff = table2array(B)-table2array(A_exact);
At = table2array(A_exact);

Erro = zeros(size(At));
for i = 1:size(At,1)
    for j = 1:size(At,2)
        if At(i,j)==0
            Erro(i,j) = 100; 
        else
            Erro(i,j) = Diff(i,j)/At(i,j)*100;
        end
        
    end
end

MediaErros = mean(Erro,1);

% T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
% Tcut = table(NumSeq, NumImg, tDom, tDice, tCard, RDO, RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros);

save comparison.mat