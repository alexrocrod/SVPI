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
imshow(B)
hold on

idx = 1;
for k=1:Nb
    D=(L==k);



    Rprops = regionprops(D,"Centroid","Circularity","Eccentricity","EquivDiameter","Area");

    r = Rprops.Centroid(1);
    c = Rprops.Centroid(2);
  
    str1=['\bf \color{red}' num2str(idx)];
    str2 = sprintf('\\color{black}\\rm%0.3f',Rprops.Circularity);
    str3 = sprintf('\\color{black}\\rm%0.3f',Rprops.Eccentricity);
    str4 = sprintf('\\color{black}\\rm%0.3f',Rprops.EquivDiameter);
    str5 = sprintf('\\color{black}\\rm%d',Rprops.Area);
    text(r,c,{str1,str2,str3,str4,str5},'HorizontalAlignment','center');
    idx = idx + 1;
end



