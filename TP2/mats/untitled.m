close all
clear all

Z = imread("../Seq29x/svpi2022_TP2_img_291_12.png");

A0 = im2double(Z);

figure(1)
imshow(A0)

A = rgb2gray(A0);

A1 = A;

% ss = 5;
% figure(2)
% 
% subplot(1,ss,1)
% imshow(A)
% 
% ola = mode(A,"all");
% 
% A = (A< 0.99*ola | A>1.01*ola);
% 
% Fiso = [1 1 1; 1 -8 1; 1 1 1];
% temp = filter2(Fiso,not(A));
% 
% A(temp>0) = 0;
% A = bwareaopen(A,10);
% subplot(1,ss,2)
% imshow(A)
% 
% A = bwmorph(A,"majority",inf);
% subplot(1,ss,3)
% imshow(A)
% 
% A = bwareaopen(A,200);
% A = imfill(A,"holes");
% subplot(1,ss,4)
% imshow(A)
% 
% A = bwmorph(A,"clean",inf);
% subplot(1,ss,5)
% imshow(A)
% 
% %% 
% figure(3)
% I = Z;
% [L,Centers] = imsegkmeans(I,3);
% B = labeloverlay(I,L);
% imshow(B)
% title("Labeled Image")

%% lazysnapping

figure(20)
imshow(A0);
RGB = A0;
L = superpixels(RGB,500);
f = drawrectangle(gca,'Position',[250 0 100 25],'Color','g');
f2 = drawrectangle(gca,'Position',[250 150 35 15],'Color','g');
foreground = createMask(f,RGB) + createMask(f2,RGB);


b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
background = createMask(b1,RGB); %+ createMask(b2,RGB);
BW = lazysnapping(RGB,L,foreground,background);

pause(1)
figure(21)
imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))

figure(22)
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
imshow(maskedImage)

%% lazysnapping v2
% nao da para usar os templates como foreground


% load("matlab.mat","regionsRGBRef")
% 
% % figure(40)
% % imshow(A0);
% RGB = A0;
% L = superpixels(RGB,500);
% 
% min1 = 0.3;
% len1 = 0.7;
% B = regionsRGBRef{1};
% sx = size(B,1);
% sy = size(B,2);
% minx = round(min1*sx);
% lenx = round(len1*sx);
% miny = round(min1*sx);
% leny = round(len1*sx);
% 
% figure(100)
% imshow(B)
% 
% f = drawrectangle(gca,'Position',[minx miny lenx leny],'Color','g');
% foreground = createMask(f,RGB);
% 
% for k=2:length(regionsRGBRef)
%     B = regionsRGBRef{k};
%     sx = size(B,1);
%     sy = size(B,2);
%     minx = round(min1*sx);
%     lenx = round(len1*sx);
%     miny = round(min1*sx);
%     leny = round(len1*sx);
% 
%     
% 
%     f = drawrectangle(gca,'Position',[minx miny lenx leny],'Color','g');
%     foreground = foreground + createMask(f,RGB);
% end
% 
% figure(40)
% imshow(A0);
% RGB = A0;
% b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% % b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
% background = createMask(b1,RGB); %+ createMask(b2,RGB);
% BW = lazysnapping(RGB,L,foreground,background);
% 
% 
% pause(1)
% figure(41)
% imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))
% 
% figure(42)
% maskedImage = RGB;
% maskedImage(repmat(~BW,[1 1 3])) = 0;
% imshow(maskedImage)

%% lazysnapping v3


figure(20)
imshow(A0);
RGB = A0;
sx = size(RGB,1);
sy = size(RGB,2);

L = superpixels(RGB,500);
f = drawrectangle(gca,'Position',[250 0 100 25],'Color','g');
f2 = drawrectangle(gca,'Position',[250 150 35 15],'Color','g');
foreground = createMask(f,RGB) + createMask(f2,RGB);


% b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');

minx = round(0.98*sy);
maxx = sy-minx;

b1 = drawrectangle(gca,'Position',[minx sx/4 maxx sx/3],'Color','r');
background = createMask(b1,RGB); %+ createMask(b2,RGB);
BW = lazysnapping(RGB,L,foreground,background);

pause(1)
figure(21)
imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))

figure(22)
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
imshow(maskedImage)

%% Lazy superpixels
close all
figure(50)
imshow(A0);
hold on
RGB = A0;
sx = size(RGB,1);
sy = size(RGB,2);

L = superpixels(RGB,500);

Ahsv = rgb2hsv(RGB);
minS = 0.4;
minV = 0.2;

% im=RGB;
% Im1=im (:,:,1)*100+im (:,:,2)*10+im (:,:,3);
% MostFrequent=mode(Im1(:));
% MfR=MostFrequent/100;
% MfG=(MostFrequent-MfR*100)/10;

rgbImg = RGB;
[idx,map] = rgb2ind( rgbImg, 0.03, 'nodither'); %// consider changing tolerance here
m = mode( idx );
frequentRGB = mode(map(m, : ));
tol = 0.1;
[r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
background = sub2ind(size(RGB),r,c);

% background = bwmorph(background,"close",inf);
% background = bwareaopen(background,150);
% background = imfill(background,"holes");
% background = bwmorph(background,"shrink",inf);

% [r,c] = ind2sub(size(RGB),background);
plot(c,r,'g*')

RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
% RMask = bwmorph(RMask,"close",inf);
% RMask = bwareaopen(RMask,150);
% RMask = imfill(RMask,"holes");

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

pause(1)
figure(51)
imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))

figure(52)
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
imshow(maskedImage)


%% Lazy superpixels AGAIN
close all
RGB = A0;
RGB(repmat(BW,[1 1 3])) = 0;
figure(60)
imshow(RGB);
hold on
sx = size(RGB,1);
sy = size(RGB,2);

L = superpixels(RGB,500);

Ahsv = rgb2hsv(RGB);
minS = 0.2; %0.4
minV = 0.2; % 0.2

% im=RGB;
% Im1=im (:,:,1)*100+im (:,:,2)*10+im (:,:,3);
% MostFrequent=mode(Im1(:));
% MfR=MostFrequent/100;
% MfG=(MostFrequent-MfR*100)/10;

rgbImg = RGB;
[idx,map] = rgb2ind(rgbImg, 0.03, 'nodither'); %// consider changing tolerance here
m = mode(idx);
temp = map(m, : );
temp( ~any(temp,2), : ) = [];  %rows
% frequentRGB = mode(map(m, : ));
frequentRGB = mode(temp);
tol = 0.1;
[r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
[r2,c2] = find(BW);
r = [r;r2];
c = [c;c2];
temp = [r,c];
temp = unique(temp,"rows");
r = temp(:,1);
c = temp(:,2);
background = sub2ind(size(RGB),r,c);

% background = bwmorph(background,"close",inf);
% background = bwareaopen(background,150);
% background = imfill(background,"holes");
% background = bwmorph(background,"shrink",inf);

% [r,c] = ind2sub(size(RGB),background);
plot(c,r,'g*')

% RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
RMask = ( Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
% RMask = bwmorph(RMask,"close",inf);
% RMask = bwareaopen(RMask,150);
% RMask = imfill(RMask,"holes");

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

pause(1)
figure(61)
imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))

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



