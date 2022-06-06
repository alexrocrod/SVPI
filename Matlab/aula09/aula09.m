% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Aula 09

exlist = {'ex1a','ex1b','ex1c','ex2','ex3a','ex3b','ex4'};

if ismember('ex1a',exlist)
%% ex1a
figure(1)

A = im2double(imread("TP2_img_01_01b.png"));

Tes = im2double(imread("tesoura_org_template.png"));
Pa = im2double(imread("pa_org_template.png"));


subplot(1,2,1)
corrTes = normxcorr2(Tes,A);
imshow(corrTes)
hold on
[r,c] = find(corrTes>0.9);
plot(c,r,'ro')

subplot(1,2,2)
corrPa = normxcorr2(Pa,A);
imshow(corrPa)
hold on
[r,c] = find(corrPa>0.9);
plot(c,r,'bo')

end
if ismember('ex1b',exlist)
%% ex1b
clearvars -except exlist
figure(2)


A = im2double(imread("TP2_img_01_01b.png"));

Tes = im2double(imread("tesoura_org_template.png"));

% subplot(1,2,1)
corrTes = normxcorr2(Tes,A);
imshow(A)
hold on
[r,c] = find(corrTes==max(corrTes(:)));
plot(c,r,'b*')

r = r - size(Tes,1)/2;
c = c - size(Tes,2)/2;
plot(c,r,'r*')


end
if ismember('ex1c',exlist)
%% ex1c
clearvars -except exlist
figure(3)

A = im2double(imread("TP2_img_01_01b.png"));

Tes = im2double(imread("tesoura_org_template.png"));

corrTes = normxcorr2(Tes,A);
imshow(A)
hold on
[r,c] = find(corrTes==max(corrTes(:)));
plot(c,r,'b*')

r = r - size(Tes,1)/2;
c = c - size(Tes,2)/2;
plot(c,r,'r*')

[L,Nb] = bwlabel(A); %obter matriz de 'labels'
s = regionprops(L,'Centroid'); %obter lista das propriedades todas

ff = reshape([s.Centroid],[2 Nb]);

cent = repmat([c;r],[1 Nb]);

diff = sum(abs(ff - cent),1);
idxL = find(diff==min(diff));

disp(idxL)

end
if ismember('ex2',exlist)
%% ex2
clearvars -except exlist
figure(4)

A = im2double(imread("TP2_img_01_01b.png"));

Pa = im2double(imread("pa_org_template.png"));

oriPa = regionprops(Pa,"Orientation").Orientation;

T = imrotate(Pa,-oriPa);
imshow(T)

% oriPa2 = regionprops(T,"Orientation").Orientation;


[L,Nb] = bwlabel(A); %obter matriz de 'labels'
s = regionprops(L,'Orientation','Image','BoundingBox'); %obter lista das propriedades todas

imshow(A)
hold on
for n=1:numel(s)
    im1 = imrotate(s(n).Image,-s(n).Orientation);

    if size(im1,1) < size(T,1)
        im1 = [zeros(size(T,1)-size(im1,1),size(im1,2));im1];
    end

    if size(im1,2) < size(T,2)
        im1 = [zeros(size(im1,1),size(T,2)-size(im1,2)),im1];
    end

    if (max(normxcorr2(T,im1),[],'all')>0.8)
        rectangle('position', s(n).BoundingBox,'EdgeColor','r');
        pause(0.5)
    end
end




end
if ismember('ex3a',exlist)
%% ex3a
clearvars -except exlist
figure(5)

A = im2double(imread("TP2_img_01_01b.png"));

[L,Nb] = bwlabel(A); %obter matriz de 'labels'
s = regionprops(L,'Area','Centroid','Eccentricity','Solidity','Perimeter','Circularity'); %obter lista das propriedades todas

ffa = [s.Circularity]';
sol = [s.Solidity]';
ecc = [s.Eccentricity]';
Patts = [ffa sol ecc];

imshow(A)
hold on

for n=1:numel(s)
    mstr = num2str(Patts(n,:)',3);
    text(s(n).Centroid(1)+10, s(n).Centroid(2),mstr,'Color','y',BackgroundColor='k');
    text(s(n).Centroid(1)-10, s(n).Centroid(2),num2str(n),'Color','g',BackgroundColor='k');
end


end
if ismember('ex3b',exlist)
%% ex3b
clearvars -except exlist
figure(6)

A = im2double(imread("TP2_img_01_01b.png"));

[L,Nb] = bwlabel(A); %obter matriz de 'labels'
s = regionprops(L,'Area','Centroid','Eccentricity','Solidity','Perimeter','Circularity','BoundingBox'); %obter lista das propriedades todas

ffa = [s.Circularity]';
sol = [s.Solidity]';
ecc = [s.Eccentricity]';
Patts = [ffa sol ecc];

pA = Patts(1,:);

pB = Patts(4,:);

dA = zeros(Nb,1);
dB = zeros(Nb,1);

imshow(A)
hold on

for n=1:Nb
    dA(n) = norm(Patts(n,:)-pA);
    dB(n) = norm(Patts(n,:)-pB);

    if (dA(n)<0.02)
        rectangle('position', s(n).BoundingBox,'EdgeColor','r');
    end

    if (dB(n)<0.02)
        rectangle('position', s(n).BoundingBox,'EdgeColor','g');
    end
end

end

if ismember('ex4',exlist)
%% ex4
clearvars -except exlist
figure(7)

A = im2double(imread("TP2_img_01_01b.png"));

[L,Nb] = bwlabel(A); %obter matriz de 'labels'
s = regionprops(L,'Area','Centroid','Eccentricity','Solidity','Perimeter','Circularity','BoundingBox'); %obter lista das propriedades todas

ffa = [s.Circularity]';
sol = [s.Solidity]';
ecc = [s.Eccentricity]';
Patts = [ffa sol ecc];

pA = Patts(1,:);

pB = Patts(4,:);

PattsA=Patts([1 12 14 16 17 18],:);
PattsB=Patts([4 6 19],:);

% dA = zeros(Nb,1);
% dB = zeros(Nb,1);

PattAMaha = mahal(Patts,PattsA); 
PattBMaha = mahal(Patts,PattsB); 

PattAMaha = PattAMaha/max(PattAMaha);
PattBMaha = PattBMaha/max(PattBMaha);

imshow(A)
hold on

for n=1:Nb
    if (PattAMaha(n)<0.0002)
        rectangle('position', s(n).BoundingBox,'EdgeColor','r');
    end

    if (PattBMaha(n)<0.0002)
        rectangle('position', s(n).BoundingBox,'EdgeColor','g');
    end
end

end

