close all
clear all
clc

%% Funciona
% Branco -> 11,24 -> alguns pequenos ruidos a 0 mean 0.33 0.29
% Azul1 -> 12,25 -> so algumas elipses para 0 mean 0.30 0.30
% Azul2 -> 13,26 -> so algumas linhas para 0  mean 0.32 0.30 tambem da bem para o Azul1
% Azul3 -> 20 -> muito bom para 0 mean 0.32
% Azul4 -> 23 -> corta oreos partidas a 0, deixa pequenos ruidos para 0.005,  mean 0.31
% Verde -> 27 -> funciona a 0, mean 0.31

%% ColorThresholder
% Branco -> 11,24 -> v2 funciona bem com minS 100 means 0.30 0.27
% Azul1 -> 12,25 -> funciona bem com 10 means means 0.29 mas 0.43 no Azul2
% Azul2 -> 13,26 -> 2 pontos de azul para 100 means 0.32 mas 0.33 no Azul1 0.30 mas 0.31 no Azul1 
% Azul3 -> 7,20 -> perfeito para 10, mean 0.30 mas no Azul4 tambem 0.30
% Azul4 -> 10,23 -> algumas coisinhas para 100, mean 0.30 com 0.52 no Azul3, mean 0.31
% Verde -> 27 -> perfeito para 10, mean 0.31
% Preto -> 9,22 -> deixa falhas 0.32 mas 0.36 no  n5 0.35 mas 0.39
% Preto2 -> 8,21 -> perfeito 0.28, 0.30


%% 
A=im2double(imread("../Seq29x/svpi2022_TP2_img_291_10.png"));
figure;
imshow(A)



%% colorThresholder
% [mask,res] = createMaskPreto09(A,100);
% figure;
% imshow(mask)
% figure;
% imshow(res)
% fprintf("mean %.2f\n",mean(mask,"all"))
% 
% 
% return

%% All Fundos ColorThres

% Branco v0 0.129	0.068	0.909
%           0.185	0.174	1


FundoLims = zeros(8,3,2);

FundoLims(:,:,1)=[  0.112	0.076	0.911
                    0.514	0.268	0.188
                    0.516	0	    0
                    0.614	0.132	0.019
                    0.588	0	    0
                    0.206	0.146	0.519
                    0.194	0	    0
                    0.995	0	    0];


FundoLims(:,:,2)=[  0.185	0.163	1
                    0.602	1	    1
                    0.569	1	    1
                    0.704	1	    0.493
                    0.929	1	    1
                    0.274	1	    1
                    1	    1	    0.181
                    0.008	0.014	0.190];

minSizes = [100 10 100 10 100 10 1000 10]; 

%% depois do colorThresh

for idx = 1:8
    HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);

    if FundoLims(idx,1,1) > FundoLims(idx,1,2) 
        mask = (H >= FundoLims(idx,1,1) | H <= FundoLims(idx,1,2)) & (S >= FundoLims(idx,2,1) & S <= FundoLims(idx,2,2)) & (V >= FundoLims(idx,3,1) & V <= FundoLims(idx,3,2)); %add a condition for value
    else
        mask = (H >= FundoLims(idx,1,1) & H <= FundoLims(idx,1,2)) & (S >= FundoLims(idx,2,1) & S <= FundoLims(idx,2,2)) & (V >= FundoLims(idx,3,1) & V <= FundoLims(idx,3,2)); %add a condition for value
    end
    
%     mask = bwmorph(mask,"hbreak",inf);
    mask=bwareaopen(mask,minSizes(idx));

    mask=~mask; %mask for objects (negation of background)
       
%     mask = bwmorph(mask,"hbreak",inf);
    mask = bwareaopen(mask,minSizes(idx));

    fprintf("fundo n%d\n",idx)
    fprintf("mean %.2f\n",mean(mask,"all"))

    figure;
    imshow(mask)
    
    % Initialize output masked image based on input image.
    maskedRGBImage = A;
    
    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~mask,[1 1 3])) = 0;

    figure;
    imshow(maskedRGBImage)
%     imshow(mask.*A)
end

return

%% All Fundos

fundos = ["fundoBranco.png","fundoAzul1.png","fundoAzul2.png","fundoAzul3.png","fundoAzul4.png","fundoVerde.png"];
FundoLims = zeros(length(fundos),3,2);
idx = 1;
for strF = fundos
    B=imread(strF);
    figure;
    imshow(B)
    HSV=rgb2hsv(B); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);
    tol = 0;

    if strF == "fundoAzul4.png"
        tol = [0.005 0.995];
    end

    Hlims=stretchlim(H,tol);
    Slims=stretchlim(S,tol);
    Vlims=stretchlim(V,tol);
    FundoLims(idx,:,:) = [Hlims Slims Vlims]';
    

    HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);
    mask= (H > Hlims(1) & H < Hlims(2)); %select by Hue
    mask=mask & (S > Slims(1) & S < Slims(2)); %add a condition for saturation
    mask=mask & (V > Vlims(1) & V < Vlims(2)); %add a condition for value
    mask=~mask; %mask for objects (negation of background)
   
    
    mask0=mask;
    mask=bwareaopen(mask,1000); %in case we need some cleaning of "small" areas.
    fprintf("fundo n%d\n",idx)
    fprintf("mean %.2f\n",mean(mask,"all"))
    figure;
    imshow(mask)
    
    figure;
    imshow(mask.*A)
    
    figure;
    imshow((mask0 & ~mask).*A)

    idx = idx + 1;
end

return;
%%
B=imread("fundoBranco.png");
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

% mask = bwmorph(mask,"close",inf);
% mask = imfill(mask,"holes");

mask0=mask;
mask=bwareaopen(mask,1000); %in case we need some cleaning of "small" areas.

figure;
imshow(mask)

figure;
imshow(mask.*A)

figure;
imshow((mask0 & ~mask).*A)