close all
clear all

Z = imread("../Seq29x/svpi2022_TP2_img_291_12.png");

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

Fiso = [1 1 1; 1 -8 1; 1 1 1];
temp = filter2(Fiso,not(A));

A(temp>0) = 0;
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

%% 
figure(3)
I = Z;
[L,Centers] = imsegkmeans(I,3);
B = labeloverlay(I,L);
imshow(B)
title("Labeled Image")


function Ibin = autobin(I)
    Ibin = double(imbinarize(I));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end



