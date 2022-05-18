close all
clear all

Z = im2double(imread("../svpi2022_TP2_img_001_01.png"));


A0 = im2double(Z);

A = rgb2gray(A0);

Abin = A<1;

% figure;
% imshow(A0)
% 
% figure;
% imshow(Abin)
% 
% figure;
% imshow(Abin.*A0)

B=Abin;
B = bwareaopen(B,10);
[L,Nb] = bwlabel(B);

figure;
% imshow(B)
imshow(A0)
hold on

idx = 1;
for k=1:Nb
    D=(L==k);
    
        
    ARGB = D.*A0;
    ARGB = ARGB(:,any(D,1),:);
    ARGB = ARGB(any(D,2),:,:);

    Rprops = regionprops(D,"Centroid","Circularity","Eccentricity","EquivDiameter","Area");

    D = D(:,any(D,1),:);
    D = D(any(D,2),:,:);


    meanR = mean(ARGB(:,:,1),'all');
    meanG = mean(ARGB(:,:,2),'all');
    meanB = mean(ARGB(:,:,3),'all');

    Ahsv = rgb2hsv(ARGB);

%     H = Ahsv(:,:,1);
%     meanH = mean(H(D),'all');
    meanH = mean(Ahsv(:,:,1),'all');
    meanS = mean(Ahsv(:,:,2),'all');
    meanV = mean(Ahsv(:,:,3),'all');
%     ola = real(log(invmoments(Agray)));
%     feats = [meanR meanG meanB meanH ola s.Eccentricity s.Solidity]';

    r = Rprops.Centroid(1);
    c = Rprops.Centroid(2);
  
    str1=['\bf \color{red}' num2str(idx)];
    str2 = sprintf('\\color{black}\\rm%0.3f',Rprops.Circularity);
    str3 = sprintf('\\color{black}\\rm%0.3f',Rprops.Eccentricity);
%     str4 = sprintf('\\color{black}\\rm%0.3f',Rprops.EquivDiameter);
%     str5 = sprintf('\\color{black}\\rm%d',Rprops.Area);
    str4 = sprintf('\\color{black}\\rm%0.3f %0.3f %0.3f',meanH, meanS, meanV);
    str5 = sprintf('\\color{black}\\rm%0.3f %0.3f %0.3f',meanR, meanG, meanB);
    text(r,c,{str1,str2,str3,str4,str5},'HorizontalAlignment','center','BackgroundColor','w',FontSize=10);
    idx = idx + 1;
end



