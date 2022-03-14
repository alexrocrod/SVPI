% Aula 2 de SVPI
% Alexandre da Rocha Rodrigues
% NMec: 92993

exlist = {
    'ex1a',
    'ex1b',
    'ex2a',
    'ex2b',
    'ex2c',
    'ex3a',
    'ex3b',
    'ex3c',
    'ex4a',
    'ex4b',
    'ex4c',
    'ex5',
    'ex6'
};

if ismember('ex1a',exlist)
%% Ex1a

P = [3 0]';
plot(P(1),P(2),'*');
a = pi/3;
Rot = [cos(a) -sin(a)
    sin(a) cos(a)];
axis([-1 4 -1 4]);
hold on; grid on; axis square;
Pc = Rot * P; 
plot(Pc(1), Pc(2), '*r')


end
if ismember('ex1b',exlist)
%% Ex1b
close 

P = [3 0]';
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 20;
angs = linspace(0, 2*pi, N);
for a=angs
    Q  = rota(a)*P;
    plot(Q(1), Q(2), '*r');
    pause(0.1);
end


end
if ismember('ex2a',exlist)
%% Ex2a
% close 

P = [3 0]';
h =  plot(P(1), P(2), 'dr');
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 100;
angs = linspace(0, 2*pi, N);
for a=angs
    Q  = rota(a)*P;
    set(h, 'Xdata', Q(1), 'Ydata', Q(2));
    pause(0.01);
end


end
if ismember('ex2b',exlist)
%% Ex2b
close all

P = [3 0]';
P2 = [2 0]';
h =  plot(P(1), P(2), '*r');
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 500;
h2 =  plot(P2(1), P2(2), 'ob');
angs = linspace(0, 10*pi, N);
for a=angs
    Q  = rota(a)*P;
    Q2  = rota(a)*P2;
    set(h, 'Xdata', Q(1), 'Ydata', Q(2));
    set(h2, 'Xdata', Q2(1), 'Ydata', Q2(2));
    pause(0.01);
end


end

if ismember('ex2c',exlist)
%% Ex2c

close all

P = [3 0]';
P2 = [2 0]';
h =  plot(P(1), P(2), '*r');
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 500;
h2 =  plot(P2(1), P2(2), 'ob');
angs = linspace(0, 10*pi, N);
for a=angs
    Q  = rota(a)*P;
    Q2  = rota(2*a)*P2;
    set(h, 'Xdata', Q(1), 'Ydata', Q(2));
    set(h2, 'Xdata', Q2(1), 'Ydata', Q2(2));
    pause(0.01);
end

end

if ismember('ex3a',exlist)
%% Ex3a
close all
P = [-0.5 0.5 0
      0   0   2];
h = fill(P(1,:), P(2,:),'y');
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 200;
angs = linspace(0, 20*pi, N);
for a=angs
    Q  = rota(a)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end


end

if ismember('ex3b',exlist)
%% Ex3b 

close all
P = [-0.5 0.5 0
      0   0   2];
P = P +[3 0]';
h = fill(P(1,:), P(2,:),'y');
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 200;
angs = linspace(0, 20*pi, N);
for a=angs
    Q  = rota(a)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    h.FaceColor = rand(1,3);
    pause(0.1);
end


end

if ismember('ex3c',exlist)
%% Ex3c

close all
P = 2*rand(2,10);
h = fill(P(1,:), P(2,:),'y');
axis([-4 4 -4 4]);
hold on; grid on; axis square;
N = 200;
angs = linspace(0, 20*pi, N);
for a=angs
    Q  = rota(a)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    h.FaceColor = rand(1,3);
    pause(0.1);
end

end

if ismember('ex4a',exlist)
%% Ex4a
close all

addpath ../lib/

P = [-0.5 0.5 0
      0   0   2
      1   1   1];
h = fill(P(1,:), P(2,:),'y');
axis([-1 4 -1 4]);
hold on; grid on; axis square;

T1 = trans2(3,0);
T2 = rot(pi/4);

Q1 = T1*T2*P;
Q2 = T2*T1*P;
h1 = fill(Q1(1,:), Q1(2,:),'r');
h2 = fill(Q2(1,:), Q2(2,:),'g');

text(mean(P(1,:)), mean(P(2,:)), 'P')
text(mean(Q1(1,:)), mean(Q1(2,:)), 'Q1')
text(mean(Q2(1,:)), mean(Q2(2,:)), 'Q2')

end

if ismember('ex4b',exlist)
%% Ex4b
addpath ../lib/

close all
P = [-0.5 0.5 0
      0   0   2
      1   1   1];
h = fill(P(1,:), P(2,:),'y');
axis([-1 6 -1 6]);
hold on; grid on; axis square;

for t = linspace(0,3,20)
    Q  = trans2(0,t)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end


end

if ismember('ex4c',exlist)
%% Ex4c
addpath ../lib/

close all
P = [-0.5 0.5 0
      0   0   2
      1   1   1];
h = fill(P(1,:), P(2,:),'y');
axis([-15 5 -1 19]);
hold on; grid on; axis square;

% Parte 1
for t = linspace(0,3,20)
    Q  = trans2(0,t)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end

% Parte 2
for a = linspace(0,pi/2,20)
    Q  = rot(a)*trans2(0,3)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end

% Parte 3
for t = linspace(0,-6,20)
    Q  = trans2(t,0)* rot(pi/2)*trans2(0,3)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end

% Parte 4
for a = linspace(0,-pi/2,20)
    Q  = trans2(-6,0)* rot(pi/2)*trans2(0,3)*rot(a)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end


end

if ismember('ex5',exlist)
%% Ex5
close all
V = [1  1 0
    -1  1 0
    -1 -1 0
     1 -1 0
     1  1 2
    -1  1 2
    -1 -1 2
     1 -1 2];

F = [ 1 2 3 4
      5 6 7 8
      1 2 6 5
      1 5 8 4
      3 7 8 4
      2 6 7 3];

h = patch('Vertices', V, 'Faces', F, 'Facecolor', 'c');
grid on



end

if ismember('ex6',exlist)
%% Ex6
addpath ../lib/
close all

[x,y] = ginput();
P = [x,y, ones(size(x))]';
h = fill(P(1,:), P(2,:),'y');
pause(1)
axis([-15 5 -1 19]);
hold on; grid on; axis square;

% Parte 1
ts = linspace(0,3,20);
sizes = linspace(1,2,20);
for i = 1:20
    t = ts(i);
    sca = sizes(i);
    Q  = trans2(0,t)*[sca sca 1]'.*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end

% Parte 2
angs = linspace(0,pi/2,20);
sizes = linspace(2,1,20);
for i= 1:20
    a = angs(i);
    sca = sizes(i);
    Q  = rot(a)*trans2(0,3)*[sca sca 1]'.*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end

% Parte 3
ts = linspace(0,-6,20);
angs = linspace(0,2*pi,20);
for i = 1:20
    t = ts(i);
    a = angs(i);
    Q  = trans2(t,0)* rot(pi/2)*trans2(0,3)*rot(a)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    pause(0.05);
end

% Parte 4
angs = linspace(0,-pi/2,20);
colors = parula(20);
for i = 1:20
    a = angs(i);
    Q  = trans2(-6,0)* rot(pi/2)*trans2(0,3)*rot(a)*P;
    set(h, 'Xdata', Q(1,:), 'Ydata', Q(2,:));
    h.FaceColor = colors(i,:);
    pause(0.05);
end

end
