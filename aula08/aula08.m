% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Aula 08

exlist = {'ex1','ex2','ex3','ex4a','ex4b','ex5','ex6','ex7','ex8','ex9'};

if ismember('ex1',exlist)
%% Ex1
figure(1)

A = im2double(rgb2gray(imread("nuts2a.jpg")));
B = ~imbinarize(A);
B = imfill(B,"holes");
C = imclearborder(B);

subplot(1,3,1)
imshow(A)
subplot(1,3,2)
imshow(B)
subplot(1,3,3)
imshow(C)

figure(101)
C = bwmorph(C,"close");
[L,Nb] = bwlabel(C);

idx = 1;
for k=1:Nb
    D=(L==k);
    subplot(2,6,idx)
    imshow(D)
    idx = idx + 1;
end


end
if ismember('ex2',exlist)
%% Ex2
close all
clearvars -except exlist
figure(2)

Argb = im2double(imread("traffic_signs.jpg"));
A = rgb2gray(Argb);

B = ~imbinarize(A);

B = imfill(B,"holes");


B = bwmorph(B,"close");
[L,Nb] = bwlabel(B);

imshow(B)
hold on

idx = 1;
for k=1:Nb
    D=(L==k);

    Rprops = regionprops(D,"Centroid","Circularity");

    r = Rprops.Centroid(1);
    c = Rprops.Centroid(2);
  
    text(r-5,c-7,num2str(idx),"FontSize",10,"Color","r",FontWeight="bold")
    text(r-17,c+7,num2str(Rprops.Circularity,4),"FontSize",10)
    idx = idx + 1;
end


figure(102)

trLim = 0.7; % 0.676 min
cirLim = 0.9; % 0.94 min

s = regionprops(L,"Circularity");
ff = [s.Circularity];

triidx = find(ff<trLim);
ciridx = find(ff>cirLim);
squidx = find(ff>trLim & ff<cirLim);

TRI = ismember(L,triidx);
SQUA = ismember(L,squidx);
CIRC = ismember(L,ciridx);

subplot(1,3,1)
imshow(TRI)
title('Triangles')

subplot(1,3,2)
imshow(CIRC)
title('Circles')

subplot(1,3,3)
imshow(SQUA)
title('Squares')

end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)

Argb = im2double(imread("traffic_signs_jam1.jpg"));
A = rgb2gray(Argb);

B = ~imbinarize(A);
B = imfill(B,"holes");
B = bwmorph(B,"close");

[L,Nb] = bwlabel(B);


trLim = 0.7; 
cirLim = 0.9; 

s = regionprops(L,"Circularity");
ff = [s.Circularity];

triidx = find(ff<trLim);
ciridx = find(ff>cirLim);
squidx = find(ff>trLim & ff<cirLim);

TRI = ismember(L,triidx);
SQUA = ismember(L,squidx);
CIRC = ismember(L,ciridx);

subplot(1,3,1)
imshow(TRI)
title('Triangles')

subplot(1,3,2)
imshow(CIRC)
title('Circles')

subplot(1,3,3)
imshow(SQUA)
title('Squares')

figure(103)

TRI = TRI.*Argb;
CIRC = CIRC.*Argb;
SQUA = SQUA.*Argb;

subplot(1,3,1)
imshow(TRI)
title('Triangles')

subplot(1,3,2)
imshow(CIRC)
title('Circles')

subplot(1,3,3)
imshow(SQUA)
title('Squares')

end
if ismember('ex4a',exlist)
%% Ex4a
clearvars -except exlist
figure(4)

A = im2double(imread("talheres_individuais.jpg"));

B = ~imbinarize(A);

[L,Nb] = bwlabel(B);

idx = 1;
for k=1:Nb
    D=(L==k);
    subplot(1,3,idx)
    imshow(D)

    Rprops = regionprops(D,"Solidity","Circularity");

    xlabel(sprintf("Solidity=%.4f\nForm Factor=%.4f",Rprops.Solidity,Rprops.Circularity))
    idx = idx + 1;
end



end
if ismember('ex4b',exlist)
%% Ex4b   FALHA AINDA
clearvars -except exlist
figure(204)

SolGarfo = 0.6935;
SolFaca = 0.7561;
SolColher = 0.7220;

FormGarfo = 0.1098;
FormFaca = 0.2468;
FormColher = 0.2433;

A = im2double(imread("talheres.jpg"));
B = ~imbinarize(A);
B = imclearborder(B);

[L,Nb] = bwlabel(B);

StatsGarfo = repmat([SolGarfo;FormGarfo],[1 Nb]);
StatsFaca = repmat([SolFaca;FormFaca],[1 Nb]);
StatsColher = repmat([SolColher;FormColher],[1 Nb]);

s = regionprops(L,"Solidity","Circularity");
ff = [s.Solidity; s.Circularity];

erroGarfo = abs(ff-StatsGarfo)./StatsGarfo;
erroFaca = abs(ff-StatsFaca)./StatsFaca;
erroColher = abs(ff-StatsColher)./StatsColher;


Garfoidx = find(all(erroGarfo<0.05));
Facaidx = find(all(erroFaca<0.05));
Colheridx = find(all(erroColher<0.05));

Garfos = ismember(L,Garfoidx);
Facas = ismember(L,Facaidx);
Colheres = ismember(L,Colheridx);


subplot(1,3,1)
imshow(Garfos)
title('Garfos')

subplot(1,3,2)
imshow(Facas)
title('Facas')

subplot(1,3,3)
imshow(Colheres)
title('Colheres')


end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)

Argb = im2double(imread("Seq1\TP2_img_01_01.png"));
A = rgb2gray(Argb);
B = imbinarize(A,0.01);
B = imclearborder(B);
B = bwareaopen(B,100);
C = imfill(B,"holes");

subplot(1,2,1)
imshow(Argb)

subplot(1,2,2)
imshow(B)


end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)

Argb = im2double(imread("Seq1\TP2_img_01_01.png"));
A = rgb2gray(Argb);
B = imbinarize(A,0.01);
B = imclearborder(B);
B = bwareaopen(B,100);
% C = imfill(B,"holes");
B = bwmorph(B,"close",inf);

[L,Nb] = bwlabel(B);
s = regionprops(L,"EulerNumber");
ff = [s.EulerNumber];

idx2 = find(ff == -1);
euler2 = (ismember( L,idx2));
subplot(1,3,1), imshow(euler2);

idx1 = find( ff == 0);
euler1 = (ismember( L,idx1));
subplot(1,3,2), imshow(euler1);

idx0 = find( ff == 1);
euler0 = (ismember( L,idx0));
subplot(1,3,3), imshow(euler0);



end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

Argb = im2double(imread("Seq1\TP2_img_01_01.png"));
A = rgb2gray(Argb);
B = imbinarize(A,0.01);
B = imclearborder(B);
B = bwareaopen(B,100);
% C = imfill(B,"holes");
B = bwmorph(B,"close",inf);


L=bwlabel(B); %obter matriz de 'labels'
s=regionprops(L,'All'); %obter lista das propriedades todas
soli=[0 0.5 0.6 0.7 1]; %limites dos intervalos
lins=2; cols=2; %medidas para o subplot

for i=2:numel(soli)
    idx=find([s.Solidity]>soli(i-1)&[s.Solidity]<=soli(i));
    m=ismember(L,idx); %imagem binaria dos objetos detetados
    subplot(lins,cols,i-1); imshow(m);
    str=sprintf('Sol>%0.2f&Sol<=%0.2f',soli(i-1),soli(i));
    title(str);
end

end

if ismember('ex8',exlist)
%% Ex8
clearvars -except exlist
figure(8)

Argb = im2double(imread("Seq1\TP2_img_01_01.png"));
A = rgb2gray(Argb);
B = imbinarize(A,0.01);
B = imclearborder(B);
B = bwareaopen(B,100);
% C = imfill(B,"holes");
B = bwmorph(B,"close",inf);


L=bwlabel(B); %obter matriz de 'labels'
s=regionprops(L,'All'); %obter lista das propriedades todas

inters=[0 0.94 0.96 0.98 1]; %limites dos intervalos
lins=2; cols=2; %medidas para o subplot

for i=2:numel(inters)
    idx=find([s.Eccentricity]>inters(i-1)&[s.Eccentricity]<=inters(i));
    m=ismember(L,idx); %imagem binaria dos objetos detetados
    subplot(lins,cols,i-1); imshow(m);
    str=sprintf('ecce>%0.2f&ecce<=%0.2f',inters(i-1),inters(i));
    title(str);
end

end

if ismember('ex9',exlist)
%% Ex9
clearvars -except exlist
figure(9)

Argb = im2double(imread("Seq1\TP2_img_01_01.png"));
A = rgb2gray(Argb);
B = imbinarize(A,0.01);
B = imclearborder(B);
B = bwareaopen(B,100);
% C = imfill(B,"holes");
B = bwmorph(B,"close",inf);


L=bwlabel(B); %obter matriz de 'labels'
s=regionprops(L,'All'); %obter lista das propriedades todas

inters=[0 0.15 0.2 0.3 1]; %limites dos intervalos
lins=2; cols=2; %medidas para o subplot

ff = [s.Circularity];

for i=2:numel(inters)
    idx=find(ff>inters(i-1)&ff<=inters(i));
    m=ismember(L,idx); %imagem binaria dos objetos detetados
    subplot(lins,cols,i-1); imshow(m);
    str=sprintf('Cir>%0.2f&Cir<=%0.2f',inters(i-1),inters(i));
    title(str);
end


end

