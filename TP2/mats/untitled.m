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

%% grabcut
% RGB = A0;
% L = superpixels(RGB,500);
% figure(30)
% imshow(RGB)
% h1 = drawpolygon('Position',[72,105; 1,231; 0,366; 104,359;...
%         394,307; 518,343; 510,39; 149,72]);
% roiPoints = h1.Position;
% roi = poly2mask(roiPoints(:,1),roiPoints(:,2),size(L,1),size(L,2));
% BW = grabcut(RGB,L,roi);
% figure(31);imshow(BW)
% 
% maskedImage = RGB;
% maskedImage(repmat(~BW,[1 1 3])) = 0;
% figure(32);imshow(maskedImage)
% 
% %% imgeodesic
% 
% RGB = A0;
% figure(40)
% imshow(RGB)
% 
% f = drawrectangle(gca,'Position',[250 0 100 25],'Color','g');
% f2 = drawrectangle(gca,'Position',[250 150 35 15],'Color','g');
% foreground = createMask(f,RGB) + createMask(f2,RGB);
% 
% b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% % b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
% background = createMask(b1,RGB); %+ createMask(b2,RGB);
% 
% [L,P] = imseggeodesic(RGB,foreground,background);
% 
% figure(41)
% imshow(label2rgb(L))
% title('Segmented Labels')
% 
% figure(42)
% imshow(labeloverlay(RGB,L))
% title('Labels Overlaid on Original Image')
% 
% %% imsegfmm
% 
% I = A1;
% figure(50);imshow(I)
% title('Original Image')
% 
% f = drawrectangle(gca,'Position',[250 0 100 25],'Color','g');
% f2 = drawrectangle(gca,'Position',[250 150 35 15],'Color','g');
% foreground = createMask(f,RGB) + createMask(f2,RGB);
% 
% b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% % b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
% background = createMask(b1,RGB); %+ createMask(b2,RGB);
% 
% % mask = false(size(I)); 
% % mask(170,70) = true;
% 
% W = graydiffweight(I,background);
% 
% thresh = 0.01;
% [BW, D] = imsegfmm(W, mask, thresh);
% figure
% imshow(BW)
% title('Segmented Image')

%%
% I = A1;
% P = rgb2gray(im2double(imread("../svpi2022_TP2_img_001_01.png")));
% P(P==1) = 0; 
% P = imadjust(P);
% 
% I = imclearborder(I);
% figure; imshow(I)
% 
% % Templates
% [L1, numObj1]= bwlabel(P);
% figure; imshow(P)
% s1 = regionprops(L1, 'Circularity', 'Solidity', 'Eccentricity');
% 
% % Imagem
% [L, numObj]= bwlabel(I);
% s = regionprops(L, 'Circularity', 'Solidity', 'Eccentricity');
% 
% X_templates = [s1.Circularity; s1.Solidity; s1.Eccentricity]';
% Y = [s.Circularity; s.Solidity;s.Eccentricity]';
% 
% dist = mahal(Y, X_templates);
% 
% dist = dist/max(dist);
% idx = find(dist < 0.03);
% size(idx)
% m = ismember(L, idx);
% figure; imshow(m);


%%


function Ibin = autobin(I)
    Ibin = double(imbinarize(I));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end



