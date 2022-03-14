% Aula 1 de SVPI
% Alexandre da Rocha Rodrigues
% NMec: 92993

exlist = {
    'ex1a',
    'ex1b',
    'ex2a',
    'ex2b',
    'ex3a',
    'ex3b',
    'ex3c',
    'ex3d',
    'ex4a',
    'ex4b',
    'ex4c',
    'ex4d',
    'ex4e',
    'ex5'
};

if ismember('ex1a',exlist)
%% Ex1a
close
Z = zeros(100,200);
imshow(Z);
end
if ismember('ex1b',exlist)
%% Ex1b
close
Z = zeros(100,200);
Z(30:70, 50:90) = 255;
imshow(Z)
end
if ismember('ex2a',exlist)
%% Ex2a

Z = zeros(100,200);
Z(30:70, 50:90) = 255;
Z(30:70, 120:160) = 128;
imshow(Z)

end
if ismember('ex2b',exlist)
%% Ex2b

Z = zeros(100,200);
Z(30:70, 50:90) = 1;
Z(30:70, 120:160) = 0.5;
imshow(Z)

figure(2)
Z = zeros(100,200,'uint8');
Z(30:70, 50:90) = 255;
Z(30:70, 120:160) = 128;
imshow(Z)

whos Z
end

if ismember('ex3b',exlist)
%% Ex3b 
close all
Z = zeros(100,200);
Z = AddSquare(Z,20,30);
imshow(Z)

end

if ismember('ex3c',exlist)
%% Ex3c

Z = zeros(100,200);
for cc=10:20:180
    Z = AddSquare(Z,10,cc);
end
imshow(Z)

end
if ismember('ex3d',exlist)
%% Ex3d

Z = zeros(100,200);
for ll=10:20:100
    for cc=10:20:200
        Z = AddSquare(Z,ll,cc);
    end
end
imshow(Z)

end

if ismember('ex4a',exlist)
%% Ex4a
Z = zeros(100,200);
x0 = 50;
y0 = 60;
r = 20;

for x=1:100
    for y=1:100
        if (x-x0)^2 + (y-y0)^2 <= r^2
            Z(y,x) = 1;
        end
    end
end
imshow(Z)


end

if ismember('ex4b',exlist)
%% Ex4b

Z = zeros(100,200);
x0 = 50;
y0 = 60;
r = 20;
x = 1:size(Z,2);
y = 1:size(Z,1);

[X,Y] = meshgrid(x,y);
C = ((X-x0).^2 + (Y-y0).^2 <= r*r);
Z(C) = 1;
imshow(Z) 

end

if ismember('ex4d',exlist)
%% Ex4d
Z = zeros(100,200);

for ll=13:25:100
    for cc=13:25:200
        Z = AddCircle(Z,cc,ll,11);
    end
end
imshow(Z)


end

if ismember('ex4e',exlist)
%% Ex4e

Z = zeros(100,200);

for ll=13:25:100
    for cc=13:25:200
        Z = AddCircle(Z,cc,ll,rand()*11);
    end
end
imshow(Z)

end

if ismember('ex5',exlist)
%% Ex4e

fact = 30;
Z = zeros(100,200);
x0 = 50;
y0 = 100;
x = 1:size(Z,2);
y = 1:size(Z,1);
x = x/fact;
y = y/fact;

[X,Y] = meshgrid(x,y);
C = (((X-x0).^2 + (Y-y0).^2 - 1).^3 - (X-x0).^2.*(Y-y0).^3 <= 0);
Z(C) = 1;
flipud(Z);
imshow(Z) 

end
