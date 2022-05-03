% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Aula 07

exlist = {'ex1','ex2','ex3','ex4','ex5','ex6','ex7','ex8','ex9','ex10','ex11','ex12'};

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
% B = im2bw(A);
B = imbinarize(rgb2gray(A));
C = bwmorph(B,'skel',inf);

subplot(1,2,1)
imshow(C)

cruz = [0 1 0
        1 1 1
        0 1 0];
T = [1 1 1
    0 1 0
    0 1 0];

% cruz = [0 0 1 0 0
%         0 0 1 0 0
%         1 1 1 1 1
%         0 0 1 0 0
%         0 0 1 0 0];
% T= [1 1 1 1 1
%     0 0 1 0 0
%     0 0 1 0 0
%     0 0 1 0 0
%     0 0 1 0 0];

D = bwhitmiss(C,cruz,~cruz) | bwhitmiss(C,T,~T);
subplot(1,2,2)
imshow(D)


end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

A = im2double(imread("pcb.png"));
% B = im2bw(A);
B = imbinarize(rgb2gray(A));

subplot(1,4,1)
imshow(B)

C = bwmorph(B,"shrink",inf);
subplot(1,4,2)
imshow(C)

Fiso = [1 1 1; 1 -8 1; 1 1 1];
isos = filter2(Fiso,C)==-8;
imshow(isos)

recon = imreconstruct(isos,B);
subplot(1,4,3)
imshow(recon)

rec2 = ~recon;
subplot(1,4,4)
imshow(rec2)

B = B.*rec2;
C = bwmorph(B,'skel',inf);

figure(17)
subplot(1,2,1)
imshow(C)

cruz = [0 1 0
        1 1 1
        0 1 0];
T = [1 1 1
    0 1 0
    0 1 0];

% cruz = [0 0 1 0 0
%         0 0 1 0 0
%         1 1 1 1 1
%         0 0 1 0 0
%         0 0 1 0 0];
% T= [1 1 1 1 1
%     0 0 1 0 0
%     0 0 1 0 0
%     0 0 1 0 0
%     0 0 1 0 0];

D = bwhitmiss(C,cruz,~cruz) | bwhitmiss(C,T,~T);
subplot(1,2,2)
imshow(D)

end

if ismember('ex8',exlist)
%% Ex8
clearvars -except exlist
figure(8)

A = im2double(imread("lixa10.png"));
% B = im2bw(A);
% B = imbinarize(rgb2gray(A));
B = imbinarize(A);
B = ~B;

subplot(1,3,1)
imshow(A) % A

C = bwmorph(B,"shrink",inf);

Fiso = [1 1 1; 1 -8 1; 1 1 1];
isos = filter2(Fiso,C)==-8;

subplot(1,3,2)
imshow(C)

rec2 = imdilate(isos,ones(15));

A(rec2) = 0.4;

subplot(1,3,3)
imshow(A)


end

if ismember('ex9',exlist)
%% Ex9
clearvars -except exlist
figure(9)

A = im2double(imread("pcb_holes.png"));
B = imbinarize(rgb2gray(A));

subplot(1,2,1)
imshow(B)

[Bx,~,Nb,~] = bwboundaries(B);

ObjOK = 0;
sx = size(B,1);
sy = size(B,2);

subplot(1,2,2)
imshow(B)
hold on

for k = Nb+1:length(Bx)
    boundary = Bx{k};
    mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
    if nnz(mask) > 50, continue, end 

    plot(boundary(:,2),boundary(:,1),'r','LineWidth',2);
%     pause(0.01)
    
    ObjOK = ObjOK + 1;
end
xlabel(ObjOK)



end

if ismember('ex10',exlist)
%% Ex10
clearvars -except exlist
figure(10)

A = im2double(imread("porcas.png"));
B = ~imbinarize(A);

subplot(1,2,1)
imshow(A)

C = imfill(B,"holes");

subplot(1,2,2)
imshow(C)

end

if ismember('ex11',exlist)
%% Ex11
clearvars -except exlist
figure(11)

A = im2double(imread("porcas.png"));
B = ~imbinarize(A);

subplot(1,3,1)
imshow(A)

C = bwmorph(B,"shrink",inf);

Fiso = [1 1 1; 1 -8 1; 1 1 1];
isos = filter2(Fiso,C)==-8;

recon = imreconstruct(isos,B);

subplot(1,3,2)
imshow(recon)

rec2 = B.*~recon;
subplot(1,3,3)
imshow(rec2)


end

if ismember('ex12',exlist)
%% Ex12
clearvars -except exlist
figure(12)

A = im2double(imread("HappySad.png"));
B = ~imbinarize(A);

subplot(1,2,1)
imshow(A)




end

