% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Aula 10

exlist = {'ex1','ex1b','ex2','ex3','ex4','ex5','ex6','ex7','ex8','ex9'};

if ismember('ex1a',exlist)
%% Ex1a
figure(1)

Argb = im2double(imread("artemoderna2.png"));

subplot(2,4,1)
imshow(Argb)

subplot(2,4,2)
RedMask = (Argb(:,:,1)==1 & Argb(:,:,2)==0 & Argb(:,:,3)==0);
imshow(RedMask)

subplot(2,4,3)
GreenMask = (Argb(:,:,1)==0 & Argb(:,:,2)==1 & Argb(:,:,3)==0);
imshow(GreenMask)

subplot(2,4,4)
BlueMask = (Argb(:,:,1)==0 & Argb(:,:,2)==0 & Argb(:,:,3)==1);
imshow(BlueMask)

subplot(2,4,6)
RedImg = RedMask.*Argb;
imshow(RedImg)

subplot(2,4,7)
GreenImg = GreenMask.*Argb;
imshow(GreenImg)

subplot(2,4,8)
BlueImg = BlueMask.*Argb;
imshow(BlueImg)

end
if ismember('ex1b',exlist)
%% Ex1b
clearvars -except exlist
figure(301)

Argb = im2double(imread("artemoderna2.png"));

subplot(1,3,1)
imshow(Argb)

subplot(1,3,2)
YMask = (Argb(:,:,1)==1 & Argb(:,:,2)==1 & Argb(:,:,3)==0);
imshow(YMask)

subplot(1,3,3)
YImg = YMask.*Argb;
R = YImg(:,:,1);
G = YImg(:,:,2);
B = YImg(:,:,3);
R(~YMask)= 1;
G(~YMask)= 1;
B(~YMask)= 1;
YImg =  cat(3, R, G, B);
imshow(YImg)


end
if ismember('ex2',exlist)
%% Ex2
clearvars -except exlist
figure(2)

Argb = im2double(imread("mongolia.jpg"));
subplot(2,1,1)
imshow(Argb)

subplot(2,1,2)
[cR,cG,cB,x] = rgbhist(Argb);

end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)

Argb = im2double(imread("morangos.jpg"));
subplot(2,1,1)
imshow(Argb)

subplot(2,1,2)
[cR,cG,cB,x] = rgbhist(Argb);


figure(103)
subplot(1,3,1)
imshow(Argb)

subplot(1,3,2)
MorangoMask = (Argb(:,:,1)>0.5 & Argb(:,:,2)<0.3 & Argb(:,:,3)<0.3);
MorangoImg = MorangoMask.*Argb;
imshow(MorangoImg)

subplot(1,3,3)
LeafMask = (Argb(:,:,1)<0.6 & Argb(:,:,2)>0.5 & Argb(:,:,3)<0.6);
LeafImg = LeafMask.*Argb;
imshow(LeafImg)


end
if ismember('ex4',exlist)
%% Ex4
clearvars -except exlist
figure(4)

Argb = im2double(imread("ArteModerna1.jpg"));

Ahsv= rgb2hsv(Argb);
subplot(1,3,1)
imshow(Argb)

subplot(1,3,2)
YMask = (Ahsv(:,:,1)>0.15 & Ahsv(:,:,1)<0.2 & Ahsv(:,:,2)>0.01);
imshow(YMask)

subplot(1,3,3)
YImg = YMask.*Argb;
imshow(YImg)

end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)

Argb = im2double(imread("ArteModerna1.jpg"));

Ahsv= rgb2hsv(Argb);

                  % >0.9
RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>0.5);
RImg = RMask.*Argb;

YMask = (Ahsv(:,:,1)>0.15 & Ahsv(:,:,1)<0.2 & Ahsv(:,:,2)>0.5);
YImg = YMask.*Argb;

GMask = (Ahsv(:,:,1)>0.25 & Ahsv(:,:,1)<0.4 & Ahsv(:,:,2)>0.5);
GImg = GMask.*Argb;

BMask = (Ahsv(:,:,1)>0.55 & Ahsv(:,:,1)<0.75 & Ahsv(:,:,2)>0.5);
BImg = BMask.*Argb;

subplot(1,4,1)
imshow(RImg)

subplot(1,4,2)
imshow(YImg)

subplot(1,4,3)
imshow(GImg)

subplot(1,4,4)
imshow(BImg)

end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)

deltaH = 1/1000;
A = ones(101,1001,3);
H = repmat(0:deltaH:1,[101 1]);
A(:,:,1) = H;

Argb = hsv2rgb(A);

subplot(2,1,1)
imshow(Argb)
grid on
axis on
xlabel("Hue")

deltaS = 1/100;
var = 0:deltaS:1;
S = repmat(var',[1 1001]);
A(:,:,2) = S;

Argb = hsv2rgb(A);

subplot(2,1,2)
imshow(Argb)
grid on
axis on
xlabel("Hue")
ylabel("Saturation")

end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

Argb = im2double(imread("feet2.jpg"));

Ahsv= rgb2hsv(Argb);

minS = 0.4;
minV = 0.2;

                  % >0.9
RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
RMask = bwmorph(RMask,"close",inf);
RMask = bwareaopen(RMask,50);
RMask = imfill(RMask,"holes");
RImg = RMask.*Argb;

YMask = (Ahsv(:,:,1)>0.15 & Ahsv(:,:,1)<0.2 & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
YMask = bwmorph(YMask,"close",inf);
YMask = bwareaopen(YMask,50);
YMask = imfill(YMask,"holes");
YImg = YMask.*Argb;

GMask = (Ahsv(:,:,1)>0.25 & Ahsv(:,:,1)<0.4 & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
GMask = bwmorph(GMask,"close",inf);
GMask = bwareaopen(GMask,50);
GMask = imfill(GMask,"holes");
GImg = GMask.*Argb;

BMask = (Ahsv(:,:,1)>0.55 & Ahsv(:,:,1)<0.75 & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
BMask = bwmorph(BMask,"close",inf);
BMask = bwareaopen(BMask,50);
BMask = imfill(BMask,"holes");
BImg = BMask.*Argb;

subplot(1,4,1)
imshow(RImg)

subplot(1,4,2)
imshow(YImg)

subplot(1,4,3)
imshow(GImg)

subplot(1,4,4)
imshow(BImg)


end

if ismember('ex8',exlist)
%% Ex8
clearvars -except exlist
figure(8)

Argb = im2double(imread("morangos.jpg"));
subplot(2,1,1)
imshow(Argb)

Ahsv= rgb2hsv(Argb);

minS = 0.4;
minV = 0.2;
                % >0.9
RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
RMask = bwmorph(RMask,"close",inf);
RMask = bwareaopen(RMask,100);
RMask = imfill(RMask,"holes");

subplot(2,1,2)
imshow(RMask)

end

if ismember('ex9',exlist)
%% Ex9
clearvars -except exlist
figure(9)

Argb = im2double(imread("morangos7.jpg"));

imshow(Argb)
hold on

Ahsv= rgb2hsv(Argb);

minS = 0.75;
minV = 0.5;

RMask = ((Ahsv(:,:,1)<0.1 |Ahsv(:,:,1)>0.9) & Ahsv(:,:,2)>minS & Ahsv(:,:,3)>minV);
RMask = bwmorph(RMask,"close",inf);
RMask = bwareaopen(RMask,150);
RMask = imfill(RMask,"holes");

points = bwmorph(RMask,"shrink",inf);

[r,c] = find(points);
plot(c,r,"bo",'MarkerSize',30)

end


function [cR,cG,cB,x] = rgbhist(A)

    [cR,x] = imhist(A(:,:,1));
    [cG,~] = imhist(A(:,:,2));
    [cB,~] = imhist(A(:,:,3));

    plot(x,cR,'r-',x,cG,'g-',x,cB,'b-')

end

