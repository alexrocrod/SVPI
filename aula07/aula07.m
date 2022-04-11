% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Aula 07

exlist = {'ex1','ex2','ex3','ex4','ex5','ex6','ex7'};

if ismember('ex1',exlist)
%% Ex1
figure(1)

Z = im2double(imread("Manycoins.png"));
A = imbinarize(Z,0.18);
X = bwmorph(A, 'erode',3);
Y = bwmorph(A, 'dilate',3);

subplot(2,3,1)
imshow(A)

subplot(2,3,2)
imshow(X)

subplot(2,3,3)
imshow(Y)

Ax = bwmorph(X, 'dilate',3);
Ay = bwmorph(Y, 'erode',3);

subplot(2,3,4)
imshow(Ax)

subplot(2,3,5)
imshow(Ay)


end
if ismember('ex2',exlist)
%% Ex2
clearvars -except exlist
figure(2)

A = im2double(imread("Manycoins.png"));
B = imbinarize(A,0.18);
BW = B;
tot = nnz(B);   
part = tot;
n = 0;
perc = 0.33;

while (part > tot*perc)
    BW = bwmorph(BW,'erode');
    part = nnz(BW);
    n = n+1;
end

subplot(1,3,1)
imshow(B)
title(sprintf('Origuinal image\n %d pixels',tot))

subplot(1,3,2)
imshow(BW)
title(sprintf('Eroded %d\n %d pixels',n,part))

BW = bwmorph(B,'erode',n-1);
subplot(1,3,3)
imshow(BW)
title(sprintf('Eroded %d\n %d pixels',n-1,nnz(BW)))

end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)

A = im2double(imread("Manycoins.png"));
B = imbinarize(A,0.18);

tot = nnz(B);

subplot(1,3,1)
imshow(B)
title("Manycoins.pnf a 18%")
xlabel(sprintf("Total pixels: %d",tot))

BW = bwmorph(B,'close');
comC = nnz(BW(B==1)); 
subplot(1,3,2)
imshow(BW)
title("close")
xlabel(sprintf("Pixels em comum: %d",comC))

BW = bwmorph(B,'open');
comO = nnz(BW(B==1));
subplot(1,3,3)
imshow(BW)
title("open")
xlabel(sprintf("Pixels em comum: %d",comO))


end
if ismember('ex4',exlist)
%% Ex4
clearvars -except exlist
figure(4)

A = zeros(300,300);
A(10:20:end,10:20:end) = 1;

subplot(1,2,1)
imshow(A)

SE1 = ones(1,10);
B = imdilate(A,SE1);

subplot(1,2,2)
imshow(B)

end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)

A = rand(300,300)>0.9995;

subplot(1,2,1)
imshow(A)

D = strel('diamond',8);
B = imdilate(A,D);

subplot(1,2,2)
imshow(B)


end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)

A = im2double(imread("pcb2.png"));
B = im2bw(A);
C = bwmorph(B,'skeleton');

subplot(1,2,1)
imshow(C)

cruz = [0 1 0
        1 1 1
        0 1 0];
T = [1 1 1
    0 1 0
    0 1 0];

D = bwhitmiss(C,cruz,T);
subplot(1,2,2)
imshow(D)


end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

end

function B = autobinwithmask(A,M)
    B = A;
    B(M) = autobin(A(M));
end


function M = circularROI(y0,x0,ri,re,A)
    M = zeros(size(A),'logical');
    for i=1:size(A,1)
        for j=1:size(A,2)
            temp = (i-x0)^2 + (j-y0)^2;
            if ((temp >= ri^2) && (temp <= re^2))
                M(i,j) = 1;
            end
        end
    end
end