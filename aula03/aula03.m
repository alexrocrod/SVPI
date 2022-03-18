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
A = imread('bolt1.png');

cols = 600;
lins = 400;

x = randi(cols);
y = randi(lins);

T = trans2(x,y)*rot(a);
tf = affine2d(T');
Ro = imref2d([lins cols]);
tempA = imwarp(A,tf, 'OutputView',Ro,'SmoothEdges',true);


end
if ismember('ex2b',exlist)
%% Ex2b

A = imread('bolt1.png');

cols = 600;
lins = 400;

x = randi(cols);
y = randi(lins);

T = trans2(x,y)*rot(a);
tf = affine2d(T');

wObj = size(A,2);
hObj = size(A,1);

imxlim = [-wObj/2 wObj/2];
imylim = [-hObj/2 hObj/2];

Ri = imref2d(size(A),imxlim,imylim);

Ro = imref2d([lins cols]);
tempA = imwarp(A,Ri,tf, 'OutputView',Ro,'SmoothEdges',true);



end
if ismember('ex2c',exlist)
%% Ex2c


end
if ismember('ex2d',exlist)
%% Ex2d


end
if ismember('ex3a',exlist)
%% Ex3a
close all



end

if ismember('ex3b',exlist)
%% Ex3b


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



end

if ismember('ex5',exlist)
%% Ex5


A = [0 0 0]';
B = [3 6 12]';
P = getnpts(A,B,4)


end