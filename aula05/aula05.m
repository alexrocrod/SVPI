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
B=zeros(size(A));
B(A<Ts(1) | A>Ts(2)) = 1;
imshow(B)
title("Pretos e Brancos")

subplot(2,3,2)
B=zeros(size(A));
B(A>Ts(1) & A<Ts(2)) = 1;
imshow(B)
title("Cinzentos")

subplot(2,3,3)
B=zeros(size(A));
B(A<Ts(1)) = 1;
imshow(B)
title("Pretos")

subplot(2,3,4)
B=zeros(size(A));
B(A>Ts(2)) = 1;
imshow(B)
title("Brancos")

subplot(2,3,5)
B=zeros(size(A));
B(A>Ts(1)) = 1;
imshow(B)
title("Pretos e Cinzentos")

subplot(2,3,6)
B=zeros(size(A));
B(A<Ts(2)) = 1;
imshow(B)
title("Brancos e Cinzentos")


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

A = im2double(imread("rice.png"));

A_1hist = imbinarize(A);

xsplit = round(size(A,1)/2);
ysplit = round(size(A,2)/2);

B = zeros(size(A));

B(1:xsplit,1:ysplit) = imbinarize(A(1:xsplit,1:ysplit));
B(1:xsplit,ysplit:end) = imbinarize(A(1:xsplit,ysplit:end));
B(xsplit:end,1:ysplit) = imbinarize(A(xsplit:end,1:ysplit));
B(xsplit:end,ysplit:end) = imbinarize(A(xsplit:end,ysplit:end));

subplot(1,3,1)
imshow(A)
hold on
line([xsplit xsplit],[0 size(A,2)],'Color','r')
line([0 size(A,1)],[ysplit ysplit],'Color','r')

subplot(1,3,2)
imshow(B)

subplot(1,3,3)
imshow(A_1hist)

figure(1)
subplot(2,2,1)
imshow(A_1hist)
title("Limiar Global")

subplot(2,2,2)
imshow(B)
title("Multi-histograma")

subplot(2,2,3)
diffs = xor(A_1hist,B);
imshow(diffs)
title("Diferenças")

subplot(2,2,4)
B_med = medfilt2(B);
imshow(B_med)
title("Multi-hist + mediana")

end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)

A = im2double(imread("rice.png"));
N=5;
M=3;

xsplit = round(size(A,1)/M);
ysplit = round(size(A,2)/N);

subplot(1,2,1)
imshow(A)
hold on

for i=2:M
    for j=2:N
        xi = (i-1)*xsplit+1;
        yi = (j-1)*ysplit+1;

        line([xi xi],[0 size(A,2)],'Color','r')
        line([0 size(A,1)],[yi yi],'Color','r')
    end
end

B = MultiRegionBin(A,N,M);
subplot(1,2,2)
imshow(B)



end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

A = im2double(imread("rice.png"));
subplot(1,3,1)
imshow(A)

M = circularROI(100,50,80,120,A);
subplot(1,3,2)
imshow(M)

B = autobinwithmask(A,M);
subplot(1,3,3)
imshow(B)


end

function B = autobinwithmask(A,M)
    A2 = A(logical(M));
    A2_bin = imbinarize(A2);

    B = A;
    B(logical(M)) = A2_bin;
end


function M = circularROI(y0,x0,ri,re,A)
    M = zeros(size(A));
    for i=1:size(A,1)
        for j=1:size(A,2)
            temp = (i-x0)^2 + (j-y0)^2;
            if ((temp >= ri^2) && (temp <= re^2))
                M(i,j) = 1;
            end
        end
    end
end