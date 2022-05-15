close all
clear all

Z = imread("../Seq29x/svpi2022_TP2_img_291_27.png");


A0 = im2double(Z);

A = rgb2gray(A0);

A1 = A;

figure;
imshow(A0)

Ahsv = rgb2hsv(A0);
figure;
imshow(Ahsv)

H = Ahsv(:,:,1);
S = Ahsv(:,:,2);
V = Ahsv(:,:,3);

figure;
imshow(H)

figure;
imshow(S)

figure;
imshow(V)


figure;
imshow(H>0 & Ahsv(:,:,2)>0.1 & Ahsv(:,:,3)>0.1)

figure;
imshow(imadjust(H))


%% HSV
% 
% figure;
% Abin = ~saveB2.*A2;
% A2R = Abin(:,:,1);
% A2R(saveB2) = frequentRGB(1);
% A2G = Abin(:,:,2);
% A2G(saveB2) = frequentRGB(2);
% A2B = Abin(:,:,3);
% A2B(saveB2) = frequentRGB(3);
% Abin = cat(3,A2R,A2G,A2B);
% 
% 
% imshow(Abin)
% 
% figure;
% imshow(rgb2hsv(Abin));
% title("hsv Abin")
% Abin = rgb2hsv(Abin);
% 
% 
% figure;
% imshow(Abin)
% 
% 
% B = autobin(Abin(:,:,1));
% B = bwareaopen(B,1000);
% B = bwmorph(B,"open",inf);
% B = bwareaopen(B,1000);
% B = bwmorph(B,"bridge",inf);
% B = imfill(B,"holes");
% 
% 
% 
% figure;
% imshow(B)
% 
% figure;
% imshow(B.*A0)
% 

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



