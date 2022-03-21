% SVPI
% Alexandre Rodrigues 92993
% Março 2022
% Aula 03

exlist = {
    'ex1',
    'ex2a',
    'ex2b',
    'ex2c',
    'ex2d',
    'ex3a',
    'ex3b',
    'ex4a',
    'ex4b',
    'ex4c',
    'ex4d',
    'ex4e',
    'ex4f',
    'ex4g',
    'ex5'
};

if ismember('ex1',exlist)
%% Ex1
Z = im2double(imread('rice.png'));
a = pi/4;

newZ2 = imrotate(Z, -a*180/pi);
figure(1); imshow(newZ2);

T = rot(a);
tf = affine2d(T');
newZ1 = imwarp(Z, tf);
figure(2); imshow(newZ1);


end
if ismember('ex2a',exlist)
%% Ex2a
close all
A = im2double(imread('bolt1.png'));
A(1,:) = 1;
A(:,1) = 1;
A(end,:) = 1;
A(:,end) = 1;

cols = 600;
lins = 400;

D = round(norm(size(A)));

x = D + randi(cols-2*D);
y = D + randi(lins-2*D);

T = trans2(x,y)*rot(a);
tf = affine2d(T');
Ro = imref2d([lins cols]);
accumul = imwarp(A,tf, 'OutputView',Ro,'SmoothEdges',true);

imshow(accumul)

end
if ismember('ex2b',exlist)
%% Ex2b

A = im2double(imread('bolt1.png'));
A(1,:) = 1;
A(:,1) = 1;
A(end,:) = 1;
A(:,end) = 1;

cols = 600;
lins = 400;

D = round(norm(size(A)));

x = D + randi(cols-2*D);
y = D + randi(lins-2*D);
a = 2*pi*rand(1);

T = trans2(x,y)*rot(a);
tf = affine2d(T');

wObj = size(A,2);
hObj = size(A,1);

imxlim = [-wObj/2 wObj/2];
imylim = [-hObj/2 hObj/2];

Ri = imref2d(size(A),imxlim,imylim);

Ro = imref2d([lins cols]);
accumul = imwarp(A,Ri,tf, 'OutputView',Ro,'SmoothEdges',true,'Interp','nearest');

imshow(accumul)

end
if ismember('ex2c',exlist)
%% Ex2c

A = im2double(imread('bolt1.png'));
A(1,:) = 1;
A(:,1) = 1;
A(end,:) = 1;
A(:,end) = 1;

cols = 600;
lins = 400;

accumul = zeros(lins,cols);

for i=1:5
    D = round(norm(size(A)));

    x = D + randi(cols-2*D);
    y = D + randi(lins-2*D);
    a = 2*pi*rand(1);
    
    T = trans2(x,y)*rot(a);
    tf = affine2d(T');
    
    wObj = size(A,2);
    hObj = size(A,1);
    
    imxlim = [-wObj/2 wObj/2];
    imylim = [-hObj/2 hObj/2];
    
    Ri = imref2d(size(A),imxlim,imylim);
    
    Ro = imref2d([lins cols]);
    singleBolt = imwarp(A,Ri,tf, 'OutputView',Ro,'SmoothEdges',true,'Interp','nearest');

    
    % Add to image
    mask = (singleBolt>0);
    accumul(mask) = singleBolt(mask);

    imshow(accumul)
end

end
if ismember('ex2d',exlist)
%% Ex2d

A = im2double(imread('bolt1.png'));
% A(1,:) = 1; A(:,1) = 1; A(end,:) = 1; A(:,end) = 1;

cols = 600;
lins = 400;

accumul = zeros(lins,cols);
accumul(1,:) = 1; accumul(:,1) = 1; accumul(end,:) = 1; accumul(:,end) = 1;

tent = 0;
i = 0;
while i<5
    tent = tent+1;

    x = randi(cols);
    y = randi(lins);
    a = 2*pi*rand(1);
    
    T = trans2(x,y)*rot(a);
    tf = affine2d(T');
    
    wObj = size(A,2);
    hObj = size(A,1);
    
    imxlim = [-wObj/2 wObj/2];
    imylim = [-hObj/2 hObj/2];
    
    Ri = imref2d(size(A),imxlim,imylim);
    
    Ro = imref2d([lins cols]);
    singleBolt = imwarp(A,Ri,tf, 'OutputView',Ro,'SmoothEdges',true,'Interp','nearest');
    
    mask = (singleBolt>0);
    if any(accumul(mask)>0)
        disp('overlaped')
        continue
    end
    
    % Add to image
    
    accumul(mask) = singleBolt(mask);

%     imshow(accumul)

    i = i+1;
end
imshow(accumul)



end
if ismember('ex3a',exlist)
%% Ex3a
close all

A = create_domino(6,6);
imshow(A)



end

if ismember('ex3b',exlist)
%% Ex3b

A = create_domino(6,6);

cols = 800;
lins = 600;

accumul = zeros(lins,cols);
accumul(1,:) = 1; accumul(:,1) = 1; accumul(end,:) = 1; accumul(:,end) = 1;

visited = zeros(1000,2);
dom = randi(6,1,2);
i = 0;
tent = 0;
while i<20
    while any(visited(:,1)==dom(1) & visited(:,2)==dom(2))
        dom = randi(6,1,2);
        disp('domino repetido')
    end
    tent = tent+1;
    A = create_domino(dom(1),dom(2));

    x = randi(cols);
    y = randi(lins);
    a = 2*pi*rand(1);
    
    T = trans2(x,y)*rot(a);
    tf = affine2d(T');
    
    wObj = size(A,2);
    hObj = size(A,1);
    
    imxlim = [-wObj/2 wObj/2];
    imylim = [-hObj/2 hObj/2];
    
    Ri = imref2d(size(A),imxlim,imylim);
    
    Ro = imref2d([lins cols]);
    singleBolt = imwarp(A,Ri,tf, 'OutputView',Ro,'SmoothEdges',true,'Interp','nearest');
    
    mask = (singleBolt>0);
    if any(accumul(mask)>0)
        disp('overlaped')
        continue
    end

%     if nnz(singleBolt & accumul)>0
%         disp('overlaped')
%         continue
%     end
    
    % Add to image
    accumul(mask) = singleBolt(mask);
%     imshow(accumul)

    i = i+1;
    visited(i,:) = dom;
end
imshow(accumul)

end

if ismember('ex4a',exlist)
%% Ex4a

imLins = 240;
imCols = 320;
A = zeros(imLins,imCols);

t = linspace(0, 2*pi, 50);
P = [ 5*cos(t);
      5*sin(t);
      30*ones(size(t))];

% fill(P(1,:),P(2,:),'y')



end

if ismember('ex4b',exlist)
%% Ex4b

P = [P; ones(1,size(P,2))];

P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;



end
if ismember('ex4c',exlist)
%% Ex4c

plot3(P(1,:),P(2,:),P(3,:),'.');
hold on; grid on; axis equal;
axis([-5 5 -5 5 0 40]);
zlabel('Z'); xlabel('X'); ylabel('Y');
line([0 0], [0 0], [0 50]);
fill3([4 -4 -4 4], [-3 -3 3 3], [0 0 0 0], 'k')



end

if ismember('ex4d',exlist)
%% Ex4d

alpha = [500 500];
center = [size(A,2) size(A,1)]/2;
K = PespectiveTransform(alpha, center);


end

if ismember('ex4e',exlist)
%% Ex4e

Ch = K*P;
C = round( Ch(1:2,:) ./ repmat(Ch(3,:),2,1));
C(2,:) = size(A,1) - C(2,:);

end
if ismember('ex4f',exlist)
%% Ex4f

Oks = (C(2,:)>0 & C(2,:)<= imLins) & (C(1,:)>0 & C(1,:)<= imCols);
C2 = C(2,Oks);
C1 = C(1,Oks);
C = [C1;C2];


end

if ismember('ex4g',exlist)
%% Ex4g

idx = sub2ind(size(A), C(2,:), C(1,:));
A(idx) = 1;

imshow(A)

end

if ismember('ex5',exlist)
%% Ex5


% A = [0 0 0]';
% B = [3 6 12]';
% P = getnpts(A,B,4)


imLins = 240;
imCols = 320;
A = zeros(imLins,imCols);

load("P.mat")

P = [P; ones(1,size(P,2))];

P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;

plot3(P(1,:),P(2,:),P(3,:),'.');
hold on; grid on; axis equal;
axis([-5 5 -5 5 0 40]);
zlabel('Z'); xlabel('X'); ylabel('Y');
line([0 0], [0 0], [0 50]);
fill3([4 -4 -4 4], [-3 -3 3 3], [0 0 0 0], 'k')


alpha = [500 500];
center = [size(A,2) size(A,1)]/2;
K = PespectiveTransform(alpha, center);


Ch = K*P;
C = round( Ch(1:2,:) ./ repmat(Ch(3,:),2,1));
C(2,:) = size(A,1) - C(2,:);


Oks = (C(2,:)>0 & C(2,:)<= imLins) & (C(1,:)>0 & C(1,:)<= imCols);
C2 = C(2,Oks);
C1 = C(1,Oks);
C = [C1;C2];

idx = sub2ind(size(A), C(2,:), C(1,:));
A(idx) = 1;

imshow(A)

end

%% functions

function K = PespectiveTransform(alpha, center)
    K = [alpha(1)    0        center(1) 0
            0       alpha(2)  center(2) 0
            0        0          1       0];
end