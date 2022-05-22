close all
clear all

%% Funciona
%
% Branco -> 11 -> perde bolachas a 0, 0.05, 0.005
% Azul1 -> 12,25 -> so algumas elipses para 0
% Azul2 -> 13,26 -> so algumas linhas para 0
% Azul3 -> 20 -> muito bom para 0.005 e 0
% Azul4 -> 23 -> corta oreos partidas a 0, deixa pequenos ruidos para 0.005
% Branco -> 24 -> alguns pequenos ruidos a 0
% Verde-> 27 -> funciona com 0.005 e 0



%%


B=imread('fundoVerde.png');
HSV=rgb2hsv(B); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);
% tol=[0.005 0.995]; 
% tol=[0.05 0.95]; 
tol = 0;
Hlims=stretchlim(H,tol);
Slims=stretchlim(S,tol);
Vlims=stretchlim(V,tol);

figure;
imshow(B)


A=im2double(imread("../Seq29x/svpi2022_TP2_img_291_11.png"));
figure;
imshow(A)

HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);
mask= (H > Hlims(1) & H < Hlims(2)); %select by Hue
mask=mask & (S > Slims(1) & S < Slims(2)); %add a condition for saturation
mask=mask & (V > Vlims(1) & V < Vlims(2)); %add a condition for value
mask=~mask; %mask for objects (negation of background)
mask=bwareaopen(mask,100); %in case we need some cleaning of "small" areas.

figure;
imshow(mask)

figure;
imshow(mask.*A)