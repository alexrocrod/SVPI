close all
clear all
clc

addpath('../Seq29x')
listaF=dir('../Seq29x/svpi2022_TP2_img_*1_*.png');
fileExact = fopen("svpi2022_tp2_seq_291.txt","r"); nLineExact = 0;
classe = 1;

MaxImg = size(listaF,1);
means = zeros(MaxImg,1);
modes = zeros(MaxImg,1);

for idxImg = [7,10,20,23]
    fprintf("idxImg:%d\n",idxImg);

    imName = listaF(idxImg).name;

    A0 = im2double(imread(imName));
    Ahsv = rgb2hsv(A0);
    H = Ahsv(:,:,1);

    means(idxImg) = mean(H,"all");
    modes(idxImg) = mode(H,"all");
    
    figure;
    imshow(A0)
    xlabel(sprintf("<>=%d,mode=%d",means(idxImg),modes(idxImg)));

end

%%
% Verde-> Means 2.00e-1 Mode 2.50e-1
% Azul Tabua -> Means 4.15e-1 4.03e-1 Mode 5.42e-1 5.42e-1
% Azul -> Means 4.29e-1 4.32e-1 Mode 5.67e-1 5.67e-1
% Branco -> Means 1.42e-1 1.40e-1  Mode 1.67e-1 1.67e-1 1.67e-1
% Azul Escuro -> Means 4.89e-1 4.92e-1 Mode 6.42e-1 6.46e-1 6.42e-1
% Preto22-> Mean 5.69e-1 Mode 8.67e-1
% Preto19 -> Mean 1.15e-1 Mode 1.22e-1
% Pretos -> Means 3.4e-2 3.5e-2 3.36e-2 3.12e-2 3.04e-2 3.69e-2 2.61e-2 Mode 0

