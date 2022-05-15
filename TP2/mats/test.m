close all
clear all

Z = imread("../Seq29x/svpi2022_TP2_img_291_12.png");

% R = Z(:,:,1);
% G = Z(:,:,2);
% B = Z(:,:,3);
% Z(:,:,1) = medfilt2(R);
% Z(:,:,2) = medfilt2(G);
% Z(:,:,3) = medfilt2(B);

A0 = im2double(Z);

figure(1)
imshow(A0)

A = rgb2gray(A0);

A1 = A;

ss = 5;
figure(2)

subplot(1,ss,1)
imshow(A)

ola = mode(A,"all");

A = (A< 0.99*ola | A>1.01*ola);

% Fiso = [1 1 1; 1 -8 1; 1 1 1];
% temp = filter2(Fiso,not(A));
% 
% A(temp>0) = 0;
A = bwareaopen(A,10);
subplot(1,ss,2)
imshow(A)

A = bwmorph(A,"majority",inf);
subplot(1,ss,3)
imshow(A)

A = bwareaopen(A,200);
A = imfill(A,"holes");
subplot(1,ss,4)
imshow(A)

A = bwmorph(A,"clean",inf);
subplot(1,ss,5)
imshow(A)

%% imsegkmeans
figure(3)
I = Z;
[L,Centers] = imsegkmeans(I,2);
B = labeloverlay(I,L);
imshow(B)
title("Labeled Image")

%%
figure(4)
% wavelength = 2.^(0:5) * 3;
wavelength = 2.^(0:2:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength,orientation);
I = im2gray(im2single(Z));
gabormag = imgaborfilt(I,g);
montage(gabormag,"Size",[4 3]) %4 6
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma); 
end
montage(gabormag,"Size",[4 3])
nrows = size(Z,1);
ncols = size(Z,2);
[X,Y] = meshgrid(1:ncols,1:nrows);
featureSet = cat(3,I,gabormag,X,Y);
L2 = imsegkmeans(featureSet,2,"NormalizeInput",true);
C = labeloverlay(Z,L2);
imshow(C)
title("Labeled Image with Additional Pixel Information")


%%
figure(5)
mask = L2==2;
mask = bwmorph(mask,"close",inf);
mask = bwmorph(mask,"bridge",inf);
mask = imfill(mask,"holes");
imshow(mask)

figure(6)
res = mask.*A0;
imshow(res)

%%
figure(10)
RGB = res;
imshow(res)
hold on
L = superpixels(RGB,500);

Ahsv = rgb2hsv(RGB);
minS = 0.1; %0.4
minV = 0.1; % 0.2

rgbImg = A0; %RGB
[idx,map] = rgb2ind(rgbImg, 0.03, 'nodither'); %// consider changing tolerance here
m = mode(idx);
temp = map(m, : );
temp( ~any(temp,2), : ) = [];  %rows
% frequentRGB = mode(map(m, : ));
frequentRGB = mode(temp);
tol = 0.1;
% [r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
% % [r2,c2] = find(BW);
% % r = [r;r2];
% % c = [c;c2];
% % temp = [r,c];
% % temp = unique(temp,"rows");
% % r = temp(:,1);
% % c = temp(:,2);
% background = sub2ind(size(RGB),r,c);

% background = bwmorph(background,"close",inf);
bgMask = (abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
% fIso2 = ones(9);
% fIso2(5,5)=-80;
% filtImg = filter2(fIso2,bgMask) ~= 0;
filtImg = bgMask;
[r,c] = find(filtImg);
background = sub2ind(size(RGB),r,c);
% background = bwmorph(background,"dilate",5);
background = bwmorph(background,"bridge",inf);
% background = bwareaopen(background,10);
% background = imfill(background,"holes");
background = bwmorph(background,"skeleton");

[r,c] = ind2sub(size(RGB),background);
plot(c,r,'g*')

RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
% RMask = ( Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
RMask = bwmorph(RMask,"close",inf);
RMask = bwareaopen(RMask,5);
RMask = imfill(RMask,"holes");

points = bwmorph(RMask,"shrink",inf);
% points = RMask;
[r,c] = find(points);
foreground = sub2ind(size(RGB),r,c);
for ixy = foreground
    if ismember(ixy,background)
        foreground(foreground==ixy)=[];
    end
end
[r,c] = ind2sub(size(RGB),foreground);
plot(c,r,'r*')

BW = lazysnapping(RGB,L,foreground,background);

figure(62)
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
imshow(maskedImage)


%%
function Ibin = autobin(I)
    Ibin = double(imbinarize(I));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end



