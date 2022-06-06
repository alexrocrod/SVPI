close all
clear all
clc

%% ColorThresholder
% 1 Branco -> 11,24 -> v2 funciona bem com minS 100 means 0.30 0.27
% 2 Azul1 -> 12,25 -> funciona bem com 10 means means 0.29 mas 0.43 no Azul2  
% 3 Tabua Azul2 -> 13,26 -> 2 pontos de azul para 100 means 0.32 mas 0.33 no Azul1 0.30 mas 0.31 no Azul1 
% 4 Azul3 -> 7,20 -> perfeito para 10, mean 0.30 mas no Azul4 tambem 0.30
% 5 Azul4 -> 10,23 -> algumas coisinhas para 100, mean 0.30 com 0.52 no Azul3, mean 0.31
% 6 Verde -> 27 -> perfeito para 10, mean 0.31
% 7 Preto -> 9,22 -> deixa falhas 0.32 mas 0.36 no  n5 0.35 mas 0.39
% 8 Preto2 -> 1,2,3,4,5,8,14,15,16,17,18,21,28,29,30 -> 0.25 a 0.33
% 9 Preto3 -> 6,19 -> 0.31
% 10 Preto22-> 9,22....

% da palmiers cheias qd usa close e fill
%% 
A=im2double(imread("../Seq29x/svpi2022_TP2_img_291_10.png"));
figure;
imshow(A)


%% colorThresholder
% [mask,res] = createMaskPreto22(A,100);
% figure;
% imshow(mask)
% figure;
% imshow(res)
% fprintf("mean %.2f\n",mean(mask,"all"))
% 
% 
% return

%% All Fundos ColorThres

FundoLims = zeros(9,3,2);

FundoLims(:,:,1)=[  0.112	0.076	0.911
                    0.514	0.268	0.188
                    0.516	0	    0
                    0.614	0.132	0.019
                    0.588	0	    0
                    0.206	0.146	0.519
                    0.194	0	    0
                    0.995	0	    0
                    0.040	0	    0];


FundoLims(:,:,2)=[  0.185	0.163	1
                    0.602	1	    1
                    0.569	1	    1
                    0.704	1	    0.493
                    0.929	1	    1
                    0.274	1	    1
                    1	    1	    0.181
                    0.008	0.014	0.190
                    0.185	1   	0.241];

minSizes = [100 10 100 10 100 10 1000 10 20]; 

%% depois do colorThresh

for idx = 1:length(FundoLims)
    minS = minSizes(idx);
    HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);

    if FundoLims(idx,1,1) > FundoLims(idx,1,2) 
        mask = (H >= FundoLims(idx,1,1) | H <= FundoLims(idx,1,2)) & (S >= FundoLims(idx,2,1) & S <= FundoLims(idx,2,2)) & (V >= FundoLims(idx,3,1) & V <= FundoLims(idx,3,2)); %add a condition for value
    else
        mask = (H >= FundoLims(idx,1,1) & H <= FundoLims(idx,1,2)) & (S >= FundoLims(idx,2,1) & S <= FundoLims(idx,2,2)) & (V >= FundoLims(idx,3,1) & V <= FundoLims(idx,3,2)); %add a condition for value
    end
    
    if idx==10
        mask=bwareaopen(mask,minS);
    
        mask=~mask; %mask for objects (negation of background)
    
        mask=bwareaopen(mask,minS); %in case we need some cleaning of "small" areas.
    
        %%%% Sempre??
        mask = bwmorph(mask,"close",inf);
        mask = imfill(mask,"holes");
        %%%%
    else
        mask = bwmorph(mask,"close",inf);
        mask = bwmorph(mask,"bridge",inf);
        mask = imfill(mask,"holes");
        
        windowSize = 7;
        kernel = ones(windowSize) / windowSize ^ 2;
        blurryImage = conv2(single(mask), kernel, 'same');
        mask = blurryImage > 0.5; % Rethreshold
        
        mask = bwareaopen(mask,minS);
        mask = bwmorph(mask,"bridge",inf);
        mask = imfill(mask,"holes");
        mask = bwareaopen(mask,minS);
    end

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
