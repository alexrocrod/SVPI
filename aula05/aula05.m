% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Aula 05

exlist = {'ex1','ex2','ex3','ex4','ex5','ex6','ex7'};

if ismember('ex1',exlist)
%% Ex1
figure(1)

M1 = im2double(imread("matches1.png"));
subplot(2,3,1)
imshow(M1)
subplot(2,3,2)
imhist(M1)
thres = 0.2;
M1_white = M1;
M1_white(M1<thres)=1;
subplot(2,3,3)
imshow(M1_white)

M2 = im2double(imread("matches2.png"));
subplot(2,3,4)
imshow(M2)
subplot(2,3,5)
imhist(M2)
M2_white = M2;
M2_white(M2<thres)=1;
subplot(2,3,6)
imshow(M2_white)

% falha nas sombras

end
if ismember('ex2',exlist)
%% Ex2
clearvars -except exlist
figure(2)

M1 = im2double(imread("matches1.png"));
thres = 0.2;
M1_white = M1;
M1_white(M1<thres)=1;
subplot(1,3,1)
imshow(M1_white)

M1_adj = imadjust(M1_white);
subplot(1,3,2)
imshow(M1_adj)

T = graythresh(M1_adj);
M1_f = M1_adj;
M1_f(M1_f<T)=0.5;
M1_f(M1<thres)=M1(M1<thres);
subplot(1,3,3)
imshow(M1_f)

end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)

A = im2double(imread("trimodalchess.png"));
subplot(1,2,1)
imshow(A)
subplot(1,2,2)
imhist(A)
hold on

Ts = multithresh(A,2);
line([Ts(1) Ts(1)],[0 1e3],'Color','r')
line([Ts(2) Ts(2)],[0 1e3],'Color','r')

figure(2)
subplot(2,3,1)
title("Pretos e Brancos")
B=zeros(size(A));
B(A<Ts(1) | A>Ts(2)) = 1;
imshow(B)

subplot(2,3,2)
title("Cinzentos")
B=zeros(size(A));
B(A>Ts(1) & A<Ts(2)) = 1;
imshow(B)

subplot(2,3,3)
title("Pretos")
B=zeros(size(A));
B(A<Ts(1)) = 1;
imshow(B)

subplot(2,3,4)
title("Brancos")
B=zeros(size(A));
B(A>Ts(2)) = 1;
imshow(B)

subplot(2,3,5)
title("Pretos e Cinzentos")
B=zeros(size(A));
B(A>Ts(1)) = 1;
imshow(B)

subplot(2,3,6)
title("Brancos e Cinzentos")
B=zeros(size(A));
B(A<Ts(2)) = 1;
imshow(B)


end
if ismember('ex4',exlist)
%% Ex4
clearvars -except exlist
figure(4)


seeds = im2double(imread("seeds.png"));
subplot(2,3,1)
imshow(seeds)

seeds2 = double(imbinarize(seeds));
subplot(2,3,2)
imshow(seeds2)

subplot(2,3,3)
imshow(autobin(seeds))


seedsI = im2double(imread("seeds_inv.png"));
subplot(2,3,4)
imshow(seedsI)
seedsI2 = imbinarize(seedsI);

subplot(2,3,5)
imshow(seedsI2)

subplot(2,3,6)
imshow(autobin(seeds2))

end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)


end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)

end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)


end
