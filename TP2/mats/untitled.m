close all
clear all

Z = imread("../Seq29x/svpi2022_TP2_img_291_27.png");

% Z = imadjust(Z,[0 .8 .8;1 1 1],[]);
% Z = imadjust(Z,[0 0 0;1 1 1],[]); % nao faz nada
A0 = im2double(Z);

A = rgb2gray(A0);

A1 = A;

figure;
imshow(A0)


%% most frequent
tol = 0.2;
rgbImg = A0;
[idx,map] = rgb2ind(rgbImg, 0.03, 'nodither'); %// consider changing tolerance here
m = mode( idx );
frequentRGB = mode(map(m, : ));
[val,freqChanel] = max(frequentRGB);

%% mode

mode = 3;

if sum(abs(frequentRGB-zeros(1,3)),'all')<tol
    mode = 1;
    fprintf("mostly black\n")
elseif sum(abs(frequentRGB-ones(1,3)),'all')<tol
    mode = 2;
    fprintf("mostly white\n")
else
    fprintf("other mode\n")
end
%% 
if frequentRGB == zeros(1,3) % Pure Black
    maskPureBlack = A>0;
%     maskPureBlack = autobin(maskPureBlack);
    maskPureBlack = bwareaopen(maskPureBlack,100000);
    figure;
    imshow(maskPureBlack)
    figure;
    imshow(edge(maskPureBlack,"canny"))
    figure;
    imshow(maskPureBlack.*A0)
end


%% hists rgb

figure;
subplot(1,3,1)
imhist(A0(:,:,1))
subplot(1,3,2)
imhist(A0(:,:,2))
subplot(1,3,3)
imhist(A0(:,:,3))

%%
if mode == 1   %%%%% tentar meter as cenas a preto para branco e usar mode 2
    figure;
    imshow(A)
    figure;
    % Aed = edge(A,'canny');
    Aed = imbinarize(A,0.05);
%     Aed = bwareaopen(Aed,1000);
%     Aed = bwmorph(Aed,"close",inf);
%     Aed = bwmorph(Aed,"bridge",inf);
%     Aed = imfill(Aed,"holes");
    Aed = bwareaopen(Aed,1000);
    imshow(Aed)
    
    figure;
    imshow(Aed.*A0)
elseif mode == 2
    figure;
    imshow(A)
    figure;
    % Aed = edge(A,'canny');
    Aed = imbinarize(1-A,0.1);
    Aed = bwmorph(Aed,"close",inf);
    Aed = bwmorph(Aed,"bridge",inf);
    Aed = imfill(Aed,"holes");
    Aed = bwareaopen(Aed,10);
    imshow(Aed)
    
    figure;
    imshow(Aed.*A0)
end

%%
figure;
Abin = A0;
for i = 1:3
%     Abin(:,:,i) = imbinarize(A0(:,:,i));
    Abin(:,:,i) = autobin(Abin(:,:,i));
    Abin(:,:,i) = bwmorph(Abin(:,:,i),"close",inf);
    Abin(:,:,i) = imfill(Abin(:,:,i),"holes");
    Abin(:,:,i) = bwareaopen(Abin(:,:,i),100);
end

subplot(1,3,1)
imshow(Abin(:,:,1))
subplot(1,3,2)
imshow(Abin(:,:,2))
subplot(1,3,3)
imshow(Abin(:,:,3))

figure;
imshow(Abin)

Abin(:,:,freqChanel) = 0;

B = sum(Abin,3)>0;
B = bwmorph(B,"close",inf);
B = bwmorph(B,"bridge",inf);
B = imfill(B,"holes");
B = bwareaopen(B,1000);

figure;
imshow(B)

figure;
imshow(B.*A0)

% figure;
% binburro = im2bw(A0);
% binburro = bwmorph(binburro,"close",inf);
% binburro = bwmorph(binburro,"bridge",inf);
% binburro = imfill(binburro,"holes");
% imshow(binburro)

%% 

A2 = A0;
% A2R = A2(:,:,1);
% A2R(B) = frequentRGB(1);
% A2G = A2(:,:,2);
% A2G(B) = frequentRGB(2);
% A2B = A2(:,:,3);
% A2B(B) = frequentRGB(3);
% 
% A2 = cat(3,A2R,A2G,A2B);
% 
% figure;
% imshow(A2)

saveB=B;

pause(1)

%% clean most common
RGB = B.*A0;

tol = 0.1;
% [r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
% figure(110)
% imshow(RGB)
% hold on
% plot(c,r,'g*')

A2R = RGB(:,:,1);
A2R(abs(A2R-frequentRGB(1))<tol) = 0;
A2G = RGB(:,:,2);
A2G(abs(A2G-frequentRGB(2))<tol) = 0;
A2B = RGB(:,:,3);
A2B(abs(A2B-frequentRGB(3))<tol) = 0;

RGB = cat(3,A2R,A2G,A2B);

figure;
imshow(RGB)
%%

figure;
Abin = ~saveB.*A2;
A2R = Abin(:,:,1);
A2R(B) = frequentRGB(1);
A2G = Abin(:,:,2);
A2G(B) = frequentRGB(2);
A2B = Abin(:,:,3);
A2B(B) = frequentRGB(3);
Abin = cat(3,A2R,A2G,A2B);

Abin =  imadjust(Abin,[0 0 0;1 1 1],[]);
imshow(Abin)

figure;
imshow(autobin(rgb2gray(Abin)));
title("autobin do gray de Abin")
for i = 1:3
%     Abin(:,:,i) = imbinarize(A0(:,:,i));
    Abin(:,:,i) = autobin(Abin(:,:,i));
    Abin(:,:,i) = bwmorph(Abin(:,:,i),"close",inf);
    Abin(:,:,i) = imfill(Abin(:,:,i),"holes");
    Abin(:,:,i) = bwareaopen(Abin(:,:,i),2000);
end

figure;
subplot(1,3,1)
imshow(Abin(:,:,1))
subplot(1,3,2)
imshow(Abin(:,:,2))
subplot(1,3,3)
imshow(Abin(:,:,3))

figure;
imshow(Abin)

Abin(:,:,freqChanel) = 0;

B = sum(Abin,3)>0;
B = bwareaopen(B,1000);
B = bwmorph(B,"open",inf);
B = bwareaopen(B,1000);
% B = bwmorph(B,"bridge",inf);
% B = imfill(B,"holes");


% RGB = B.*A0;
% A2R = RGB(:,:,1);
% A2G = RGB(:,:,2);
% A2B = RGB(:,:,3);
% B(abs(A2R-frequentRGB(1))<tol & abs(A2G-frequentRGB(2))<tol & abs(A2B-frequentRGB(3))<tol) = 0;

% RGB = B.*A0;
% [r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
% B(r,c) = 0;

% RGB = B.*A0;
% [r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
% B(r,c) = 0;

figure;
imshow(B)

figure;
imshow(B.*A0)

saveB2=B|saveB;

%% HSV

figure;
Abin = ~saveB2.*A2;
A2R = Abin(:,:,1);
A2R(saveB2) = frequentRGB(1);
A2G = Abin(:,:,2);
A2G(saveB2) = frequentRGB(2);
A2B = Abin(:,:,3);
A2B(saveB2) = frequentRGB(3);
Abin = cat(3,A2R,A2G,A2B);


imshow(Abin)

figure;
imshow(rgb2hsv(Abin));
title("hsv Abin")
Abin = rgb2hsv(Abin);


figure;
imshow(Abin)


B = autobin(Abin(:,:,1));
B = bwareaopen(B,1000);
B = bwmorph(B,"open",inf);
B = bwareaopen(B,1000);
B = bwmorph(B,"bridge",inf);
B = imfill(B,"holes");



figure;
imshow(B)

figure;
imshow(B.*A0)


%%
figure;
imshow((B|saveB2).*A0)

return

%%
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

% %% lazysnapping
% 
% figure(20)
% imshow(A0);
% RGB = A0;
% L = superpixels(RGB,500);
% f = drawrectangle(gca,'Position',[250 0 100 25],'Color','g');
% f2 = drawrectangle(gca,'Position',[250 150 35 15],'Color','g');
% foreground = createMask(f,RGB) + createMask(f2,RGB);
% 
% 
% b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% % b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
% background = createMask(b1,RGB); %+ createMask(b2,RGB);
% BW = lazysnapping(RGB,L,foreground,background);
% 
% pause(1)
% figure(21)
% imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))
% 
% figure(22)
% maskedImage = RGB;
% maskedImage(repmat(~BW,[1 1 3])) = 0;
% imshow(maskedImage)
% 
% %% lazysnapping v2
% % nao da para usar os templates como foreground
% 
% 
% % load("matlab.mat","regionsRGBRef")
% % 
% % % figure(40)
% % % imshow(A0);
% % RGB = A0;
% % L = superpixels(RGB,500);
% % 
% % min1 = 0.3;
% % len1 = 0.7;
% % B = regionsRGBRef{1};
% % sx = size(B,1);
% % sy = size(B,2);
% % minx = round(min1*sx);
% % lenx = round(len1*sx);
% % miny = round(min1*sx);
% % leny = round(len1*sx);
% % 
% % figure(100)
% % imshow(B)
% % 
% % f = drawrectangle(gca,'Position',[minx miny lenx leny],'Color','g');
% % foreground = createMask(f,RGB);
% % 
% % for k=2:length(regionsRGBRef)
% %     B = regionsRGBRef{k};
% %     sx = size(B,1);
% %     sy = size(B,2);
% %     minx = round(min1*sx);
% %     lenx = round(len1*sx);
% %     miny = round(min1*sx);
% %     leny = round(len1*sx);
% % 
% %     
% % 
% %     f = drawrectangle(gca,'Position',[minx miny lenx leny],'Color','g');
% %     foreground = foreground + createMask(f,RGB);
% % end
% % 
% % figure(40)
% % imshow(A0);
% % RGB = A0;
% % b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% % % b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
% % background = createMask(b1,RGB); %+ createMask(b2,RGB);
% % BW = lazysnapping(RGB,L,foreground,background);
% % 
% % 
% % pause(1)
% % figure(41)
% % imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))
% % 
% % figure(42)
% % maskedImage = RGB;
% % maskedImage(repmat(~BW,[1 1 3])) = 0;
% % imshow(maskedImage)
% 
% %% lazysnapping v3
% 
% 
% figure(20)
% imshow(A0);
% RGB = A0;
% sx = size(RGB,1);
% sy = size(RGB,2);
% 
% L = superpixels(RGB,500);
% f = drawrectangle(gca,'Position',[250 0 100 25],'Color','g');
% f2 = drawrectangle(gca,'Position',[250 150 35 15],'Color','g');
% foreground = createMask(f,RGB) + createMask(f2,RGB);
% 
% 
% % b1 = drawrectangle(gca,'Position',[100 70 70 70],'Color','r');
% % b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
% 
% minx = round(0.98*sy);
% maxx = sy-minx;
% 
% b1 = drawrectangle(gca,'Position',[minx sx/4 maxx sx/3],'Color','r');
% background = createMask(b1,RGB); %+ createMask(b2,RGB);
% BW = lazysnapping(RGB,L,foreground,background);
% 
% pause(1)
% figure(21)
% imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))
% 
% figure(22)
% maskedImage = RGB;
% maskedImage(repmat(~BW,[1 1 3])) = 0;
% imshow(maskedImage)

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

%% clean most common
RGB = maskedImage;
[idx,map] = rgb2ind(A0, 0.03, 'nodither'); %// consider changing tolerance here
m = mode(idx);
temp = map(m, : );
temp( ~any(temp,2), : ) = [];  %rows
frequentRGB = mode(temp);
tol = 0.1;
[r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
figure(110)
imshow(RGB)
hold on
plot(c,r,'g*')


RGB(r,c,:) = 0;

figure(111)
imshow(RGB)


%% clean least common

figure(201)
imhist(RGB(:,:,2))
% [idx,map] = rgb2ind(A0, 0.3, 'nodither'); %// consider changing tolerance here
% 
% % m = mode(idx);
% V = idx(:);
% uv = unique(V);
% n = histc(V,uv);
% [m,i] = min(n); 
% minmode = uv(i);
% 
% 
% % temp = map(m, : );
% temp = map(minmode, : );
% 
% temp(~any(temp,2), : ) = [];  %rows
% frequentRGB = mode(temp,1);
% tol = 0.1;
% [r,c] = find(abs(RGB(:,:,1) - frequentRGB(1)) < tol & abs(RGB(:,:,2) - frequentRGB(2)) < tol & abs(RGB(:,:,3) - frequentRGB(3)) < tol);
% figure(112)
% imshow(RGB)
% hold on
% plot(c,r,'g*')
% 
% 
% RGB(r,c,:) = 0;
% 
% figure(113)
% imshow(RGB)


return
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
%     Ibin = double(imbinarize(I));

%     T = adaptthresh(I,0.2,'ForegroundPolarity','dark');
    [counts,x] = imhist(I,16);
    T = otsuthresh(counts);
    Ibin = double(imbinarize(I,T));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end



