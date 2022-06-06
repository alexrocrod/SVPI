% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Aula 06

exlist = {'ex1','ex2','ex3','ex4','ex5','ex6','ex7'};

if ismember('ex1',exlist)
%% Ex1
figure(1)

A = im2double(imread("coins.png"));

Sx = [-1 0 1
    -2 0 2
    -1 0 1];

Sy = [-1 -2 -1
        0 0 0
        1 2 1];

Agx = abs(filter2(Sx,A));
Agy = abs(filter2(Sy,A));
Ag = Agx + Agy;

subplot(1,3,1)
imshow(Agx)

subplot(1,3,2)
imshow(Agy)

subplot(1,3,3)
imshow(Ag)

end
if ismember('ex2',exlist)
%% Ex2
clearvars -except exlist
figure(2)

A = im2double(imread("coins.png"));
[Gx,Gy] = imgradientxy(A,'sobel');

subplot(1,4,1)
imshow(Gx,[0 1])

subplot(1,4,2)
imshow(Gy,[0 1])

G = Gx+Gy;

subplot(1,4,3)
imshow(G,[0 1])

[Gmag,Gdir]=imgradient(Gx,Gy);
% [Gmag,Gdir]=imgradient(A,'sobel');

subplot(1,4,4)
imshow(Gdir,[0 1])


end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)

A = im2double(imread("coins.png"));
B = edge(A,'sobel');
imshow(B)


end
if ismember('ex4',exlist)
%% Ex4
clearvars -except exlist
figure(4)

A = im2double(imread("coins.png"));

subplot(2,2,1)
imshow(edge(A,'sobel'))

subplot(2,2,2)
imshow(edge(A,'canny'))

subplot(2,2,3)
imshow(edge(A,'prewitt'))

subplot(2,2,4)
imshow(edge(A,'log'))

end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)

A = im2double(imread("Tcomluz.jpg"));

B = edge(A,'sobel');
subplot(2,2,1)
imshow(B)

subplot(2,2,2)
imshow(edge(A,'canny'))

subplot(2,2,3)
imshow(edge(A,'prewitt'))

subplot(2,2,4)
imshow(edge(A,'log'))


end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)

A = im2double(imread("coins.png"));
Z = edge(A,'sobel');
X = false(size(Z));

subplot(1,2,1);
imshow(Z);
title('All edges');

subplot(1,2,2);
imshow(X);
hold on;
title('Selected edges')

minSize = 100;

[L,N] = bwlabel(Z);

for k = 1:N
    C = (L==k);
    if ( sum(C,'all') < minSize), continue; end
    X = X | C;
    subplot(1,2,2);
    imshow(X);
    pause(0.2)
end


end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

A = im2double(imread("coins.png"));
Z = edge(A,'canny');
X = false(size(Z));

subplot(1,3,1);
imshow(Z);
title('All edges');

subplot(1,3,2);
imshow(X);
hold on;
title('Large edges')

minSize = 160;

[L,N] = bwlabel(Z);

for k = 1:N
    C = (L==k);
    if ( sum(C,'all') < minSize), continue; end
    X = X | C;
    subplot(1,3,2);
    imshow(X);
    pause(0.1)
end

Y = false(size(Z));
subplot(1,3,3);
imshow(Y);
hold on;
title('Small edges')

[L,N] = bwlabel(Z);

for k = 1:N
    C = (L==k);
    if ( sum(C,'all') > minSize), continue; end
    Y = Y | C;
    subplot(1,3,3);
    imshow(Y);
    pause(0.1)
end


end


if ismember('ex8',exlist)
%% Ex8
clearvars -except exlist
figure(8)

A = im2double(imread("coins.png"));
Z = edge(A,'sobel');
X = false(size(Z));

subplot(1,2,1);
imshow(Z);
hold on
axis on
title({'Edges overlayed with','larger outer countours'});

myAxis = axis;
subplot(1,2,2), hold on, axis ij, axis equal, axis(myAxis), grid on;
title({'Separate plot of the','larger outer countours'})

minSize = 100;
[L,N] = bwlabel(Z);

for k = 1:N
    C = (L==k);
    if ( nnz(C) < minSize), continue; end

    BB = bwboundaries(C,'noholes');
    boundary = BB{1};

    subplot(1,2,1);
    plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);

    subplot(1,2,2);
    plot(boundary(:,2),boundary(:,1),'b');

    pause(0.5)
end


end

if ismember('ex9',exlist)
%% Ex9
clearvars -except exlist
figure(9)

A = im2double(imread("Tcomluz.jpg"));
Z = edge(A,'canny');
X = false(size(Z));

subplot(1,2,1);
imshow(A);
hold on
axis on
title({'Edges overlayed with','larger outer countours'});

myAxis = axis;
subplot(1,2,2), hold on, axis ij, axis equal, axis(myAxis), grid on;
imshow(X)
title({'Separate plot of the','larger outer countours'})

[L,N] = bwlabel(Z);
max_size = 0;
max_k = 0;

for k = 1:N
    C = (L==k);

    BB = bwboundaries(C,'noholes');
    boundary = BB{1};

    if length(boundary)> max_size
        max_k = k;
        max_size = length(boundary);
    end
end

C = (L==max_k);
BB = bwboundaries(C,'noholes');
boundary = BB{1};

subplot(1,2,1);
plot(boundary(:,2),boundary(:,1),'r');

subplot(1,2,2);
plot(boundary(:,2),boundary(:,1),'r');


for k = 1:N
    if ( k==max_k), continue; end
    C = (L==k);

    BB = bwboundaries(C,'noholes');
    boundary = BB{1};

    subplot(1,2,2);
    plot(boundary(:,2),boundary(:,1),'g');

%     pause(0.01)
end

%% Ex9_2
figure(10)

A = im2double(imread("Tcomluz.jpg"));
Z = edge(A,'canny',[0 0.5],2);
X = false(size(Z));

subplot(1,2,1);
imshow(A);
hold on
axis on
title({'Edges overlayed with','larger outer countours'});

myAxis = axis;
subplot(1,2,2), hold on, axis ij, axis equal, axis(myAxis), grid on;
imshow(X)
title({'Separate plot of the','larger outer countours'})

[L,N] = bwlabel(Z);
max_size = 0;
max_k = 0;

for k = 1:N
    C = (L==k);

    BB = bwboundaries(C,'noholes');
    boundary = BB{1};

    if length(boundary)> max_size
        max_k = k;
        max_size = length(boundary);
    end
end

C = (L==max_k);
BB = bwboundaries(C,'noholes');
boundary = BB{1};

subplot(1,2,1);
plot(boundary(:,2),boundary(:,1),'r');

subplot(1,2,2);
plot(boundary(:,2),boundary(:,1),'r');


for k = 1:N
    if ( k==max_k), continue; end
    C = (L==k);

    BB = bwboundaries(C,'noholes');
    boundary = BB{1};

    subplot(1,2,2);
    plot(boundary(:,2),boundary(:,1),'g');

%     pause(0.01)
end


end