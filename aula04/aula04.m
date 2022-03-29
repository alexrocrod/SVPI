% SVPI
% Alexandre Rodrigues 92993
% Março 2022
% Aula 04

exlist = {'ex1','ex2','ex3','ex4','ex5','ex6','ex7'};

if ismember('ex1',exlist)
%% Ex1
figure(1)
lins = 200;
cols = 200;
A = zeros(lins,cols);
dx = 50;
dy = 80;
cx = lins/2-dx/2;
cy = cols/2-dy/2;
A(cx:cx+dx,cy:cy+dy) = 1;

subplot(1,4,1)
imshow(A)

rng(0);

B = imnoise(A,"salt & pepper",0.05);
subplot(1,4,2)
imshow(B)

F = ones(3,3)/9;
C=filter2(F,B);
subplot(1,4,3)
imshow(C)

D = medfilt2(B,[3 3]);
subplot(1,4,4)
imshow(D)

pause(2)

end
if ismember('ex2',exlist)
%% Ex2
clearvars -except exlist
figure(2)
lins = 200;
cols = 200;
A = zeros(lins,cols);
dx = 50;
dy = 80;
cx = lins/2-dx/2;
cy = cols/2-dy/2;
A(cx:cx+dx,cy:cy+dy) = 1;

subplot(1,3,1)
imshow(A)

rng(0);

B = imnoise(A,"salt & pepper",0.01);
subplot(1,3,2)
imshow(B)

Fiso =[-1 -1 -1; -1 8 -1; -1 -1 -1];
temp = filter2(Fiso,B);
C = abs(temp)==8; 
subplot(1,3,3)
imshow(C)

str = sprintf('Isolados: %d',nnz(C));
xlabel(str);

pause(2)

end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)
lins = 200;
cols = 200;
A = zeros(lins,cols);
dx = 50;
dy = 80;
cx = lins/2-dx/2;
cy = cols/2-dy/2;
A(cx:cx+dx,cy:cy+dy) = 1;

subplot(1,3,1)
imshow(A)

rng(0);

T = imnoise(A,"salt & pepper",0.005);

F = zeros(3,3,4);
F(:,:,1) = [ 1  1  1;  1  -8  1;  1  1  1];
F(:,:,2) = [ 1  2  1;  2 -12  2;  1  2  1];
F(:,:,3) = [-1  1 -1;  1   4  1; -1  1 -1];
F(:,:,4) = [ 1  2  3;  4 -100 5;  6  7  8];


whiteIsol = zeros(3,3); whiteIsol(2,2)=1;
w1 = sum(sum(whiteIsol.*F(:,:,:)));
% W = reshape(w1,1,4);
W = squeeze(w1);

blackIsol = not(whiteIsol);
MW1 = sum(sum(blackIsol.*F(:,:,:)));
% MW = reshape(MW1,1,4);
MW = squeeze(MW1);

for n=1:4
    X = filter2(F(:,:,n),T);
    ZW = (X==W(n));
    ZB = (X==MW(n));

    subplot(2,4,n)
    imshow(ZW)
    str = sprintf('Total %d',nnz(ZW));
    xlabel(str);

    subplot(2,4,4+n)
    imshow(ZB)
    str = sprintf('Total %d',nnz(ZB));
    xlabel(str);
end

pause(2)

end
if ismember('ex4',exlist)
%% Ex4
clearvars -except exlist
figure(4)
A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];

subplot(1,3,1)
imshow(A)

F = [ 0  1  0;  1  -4  1;  0  1  0];
temp = filter2(F,A);
B = (temp==-1); 
subplot(1,3,2)
imshow(B)

subplot(1,3,3)
imshow(A)
% [xs,ys] = ind2sub(size(A),find(B==1));
% text(ys,xs,'X','Color','r')
hold on;
[r,c] = find(B);
plot(c,r,'b*')


pause(2)

end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)
A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];
  
Z=A;
F = [ 0  1  0;  1  -4  1;  0  1  0];

temp = filter2(F,Z);
B = (temp==-1);
nint = nnz(B);
disp(nint)

subplot(1,3,1)
imshow(Z)

% [xs,ys] = ind2sub(size(A),find(B==1));
% text(ys,xs,'x','Color','b')
hold on;
[r,c] = find(B);
plot(c,r,'b*')

str = sprintf('Nint: %d',nint);
xlabel(str);

while nint>0
    X = ones(size(A));
    totpix = size(A,1)*size(A,2);
    pts = randperm(totpix,round(0.025*totpix));
    
    elims = zeros(size(A));
    elims(pts) = 1;
    subplot(1,3,2)
    imshow(not(elims))

    Z(pts)=0;

    temp = filter2(F,Z);
    B = (temp==-1);
    nint = nnz(B);

    subplot(1,3,3)
    imshow(Z)
%     [xs,ys] = ind2sub(size(A),find(B==1));
%     text(ys,xs,'x','Color','r')
    hold on;
    [r,c] = find(B);
    plot(c,r,'rx')

%     [xs2,ys2] = ind2sub(size(A),pts);
%     text(ys2,xs2,'o','Color','b')
    hold on;
    [r,c] = find(elims);
    plot(c,r,'bo')

    str = sprintf('Nint: %d',nint);
    xlabel(str);

    pause(0.1)
end

pause(2)

end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)
A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];

subplot(1,3,1)
imshow(A)

F = [ 0  1  0;  1  -4  1;  0  1  0];
temp = filter2(F,A);
B = (temp==-3); 
B(1,:) = 0; B(:,1) = 0; B(end,:) = 0; B(:,end) = 0;

% F = [ 1  1  1;  1  -4  1;  1  1  1];
% F = [ 1   5  1
%       5  30  5
%       1   5  1];
% temp = filter2(F,A);
% B = (temp==35); 

subplot(1,3,2)
imshow(B)

subplot(1,3,3)
imshow(A)
% [xs,ys] = ind2sub(size(A),find(B==1));
% text(ys,xs,'X','Color','r')

hold on;
[r,c] = find(B);
plot(c,r,'rx')

pause(2)

end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)
A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];

A2=A;

for ola=1:4
    Z=A2;
    subplot(1,3,1)
    imshow(Z)
    
%     F = [ 0  1  0;  1  -4  1;  0  1  0];
    F = [ 1   5  1
      5  30  5
      1   5  1];

    nbecos=1;
    while nbecos>0
        temp = filter2(F,Z);
        B = (temp==-3); 
        B(1,:) = 0; B(:,1) = 0; B(end,:) = 0; B(:,end) = 0;
%         B = (temp==35); 
        subplot(1,3,2)
        imshow(B)

        nbecos=nnz(B);
        str = sprintf('Nbecos: %d',nbecos);
        xlabel(str);
     
        Z(B)=0;
        subplot(1,3,3)
        imshow(Z)
    
        pause(0.1)
    end
    A2=rot90(A2);
end



end
