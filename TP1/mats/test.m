close 
clear


addpath('../sequencias/Seq160')
listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');


idxImg = 11;
imName = listaF(idxImg).name;

% ss = 3;

A = im2double(imread(imName));

regions=vs_getsubimages(A); %extract all regions
regions2=vs_getsubimages(A); %working
N=numel(regions);
SS=ceil(sqrt(N));

figure(1)
for k=1:N 
    subplot( SS, SS, k);
    imshow(regions{k})
    xlabel(k)
end


figure(2)
for k=1:N 
    subplot( SS, SS, k);
    regions2{k} = medfilt2(filter2(fspecial('average',3),regions{k}));
    imshow(regions2{k})
    xlabel(k)
end

