close all
clear all


addpath('../sequencias/Seq160')
listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

% addpath('../sequencias/Seq530')
% listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');


idxImg = 1;
imName = listaF(idxImg).name;


A = im2double(imread(imName));

regions=vs_getsubimages(A); %extract all regions
regions2=vs_getsubimages(A); %working
N=numel(regions);
SS=ceil(sqrt(N));

for k=1:N 
    regions2{k} = medfilt2(filter2(fspecial('average',3),regions{k}));
end



figure(3)
cartas1k = [];
cartas2k = [];
perc = 0.15; % 0.15 
perc0 = 0.1; % 0.1
px = 0.15; % 0.14
py = 0.25; % 0.25


for k=1:N 
    subplot( SS, SS, k);
    cut = 2; % 5
%     B = autobin(imadjust(regions{k}(cut:end-cut,cut:end-cut)));
    B = autobin(imadjust(regions2{k}(cut:end-cut,cut:end-cut)));
%     B = edging(B);
    
    sx = size(B,1);
    sy = size(B,2);

    % Test Noise
    C = bwmorph(B,'erode',2);
    if nnz(C) < 0.01*nnz(B)
        fprintf("nnz=%d, m= %d, noise: %d\n",nnz(C),0.01*nnz(B) ,k)
        imshow(B)
        xlabel("noise")
    

    elseif sx ~= sy
        if sx>sy
            B = rot90(B);
            regions{k} = rot90(regions{k});
        end
        
        [res,B] = sepCartas(B,perc,perc0,px,py);
        if res == 0
            imshow(B)
            xlabel("carta NA")
        elseif res ==1
            imshow(B)
            xlabel("carta tipo1")
            cartas1k = [cartas1k k];
        elseif res == 2
            imshow(B)
            xlabel("carta tipo2")
            cartas2k = [cartas2k k];
        end
    
    else
        imshow(B)
    end

    regions2{k} = double(B);

end

cartas = [cartas1k cartas2k];


cut = 2; % 5
nrows = 2;

copa = getCopaMatrix();
ouro = strel('diamond',250).Neighborhood;
tolOuros = 0.2;
tolCopas = 0.2;


figure(4)
means = zeros(length(cartas),1);
meansCopa = zeros(length(cartas),1);

idx = 1;
for k=cartas
    
%     B = regions{k}(cut:end-cut,cut:end-cut);
    B = regions{k}(cut:end-cut,cut:end-cut);

    dx = round(px*size(B,1)); 
    dy = round(py*size(B,2));

    if ismember(k,cartas1k)
        tipo=1;
        CantoSup = rot90(B(1:dx,end-dy:end));
    else
        tipo=2;
        CantoSup = rot90(rot90(rot90(B(1:dx,1:dy))));
    end

    subplot(nrows,length(cartas),idx)
    imshow(CantoSup)

    subplot(nrows,length(cartas),idx + length(cartas))
    imshow(getNaipe(B,tipo,px,py))
    
    [res,means(idx)] = classNaipe(B,tipo,ouro,px,py,tolOuros);
    if res
        xlabel(sprintf("Ouros tp%d",tipo))
    else
        [res,meansCopa(idx)] = classNaipe(B,tipo,copa,px,py,tolCopas);
        if res
            xlabel(sprintf("Copas tp%d",tipo))
        else
            xlabel(sprintf("T%d,O:%.2f,C:%.2f",tipo,means(idx),meansCopa(idx)))
        end
    end

    idx = idx + 1;

end


% deck = ["♠","♥","♦","♣"] + ["A"; (2:10)';'J';'Q';'K'];


function B = edging(A)
    B = A;
%     B = medfilt2(B);
    B = edge(B,'roberts');
%     B = bwareaopen(B,round(0.1*size(B,1)));
    B = bwmorph(B,'remove');
    B = bwmorph(B,'bridge');
%     B = bwmorph(B,'fill');
    
%     B = bwmorph(B,'remove'); % remove
%     B = bwareaopen(B,round(0.2*size(B,1)));
end

function Ibin= autobin(I) 
    Ibin = double(imbinarize(I));
    
    if nnz(Ibin)>0.5*(size(Ibin,1)*size(Ibin,2))
        Ibin = not(Ibin);
    end
end

function [res,B] = sepCartas(B,perc,perc0,px,py)
    
    dx = round(px*size(B,1));
    dy = round(py*size(B,2));
    area = dx*dy;
    nnzSupDir = nnz(B(1:dx,end-dy:end));
    nnzInfDir = nnz(B(end-dx:end,end-dy:end));
    nnzSupEsq = nnz(B(1:dx,1:dy));
    nnzInfEsq = nnz(B(end-dx:end,1:dy));

    if (nnzInfEsq > perc*area && nnzSupDir > perc*area  && ...
            nnzInfDir < perc0*area && nnzSupEsq < perc0*area)
        B(1:dx,1:dy) = 0;
        B(end-dx:end,end-dy:end) = 0;

        res=1;
        
    elseif (nnzInfDir > perc*area && nnzSupEsq > perc*area && ...
            nnzInfEsq < perc0*area && nnzSupDir < perc0*area)
        B(end-dx:end,1:dy) = 0;
        B(1:dx,end-dy:end) = 0;
        
        res = 2;
    else
        res = 0;
    end
end

function copa = getCopaMatrix()
    A = false(501,501);
    idx = 1;
    for x=-250:250
        idy = 1;
        for y = -250:250
            if (x^2 + y^2 - 1e4)^3 < 200*x^2*y^3
                A(end-idy,idx) = true;
            end
            idy = idy + 1;
        end
        idx = idx +1;
    end
    
    copa = A(any(A,2),:);
    copa = copa(:,any(copa,1));
end

function [res,meanC] = classNaipe(carta, tipo,naipe,px,py,tol)
    res = false;
    sc = 10;

    clean0s = getNaipe(carta,tipo,px,py);

    meanC = mean(imresize(clean0s,sc)~=imresize(naipe,sc*size(clean0s)),'all');

    if meanC < tol
        res = true;
    end

end

function res = getNaipe(carta, tipo,px, py)

    B = carta;
    dx = round(px*size(B,1)); % 0.14
    dy = round(py*size(B,2)); % 0.25??

    if tipo == 1
        CantoSup = rot90(B(1:dx,end-dy:end));
    elseif tipo == 2
        CantoSup = rot90(rot90(rot90(B(1:dx,1:dy))));
    end

    CSbin = autobin(imadjust(CantoSup));

    dx2 = round(0.55*size(CSbin,1)); 
    NaipeSD = CSbin(dx2:end,:);
    
%     clean0s = NaipeSD(any(NaipeSD,2),:);
%     res = clean0s(:,any(clean0s,1));

    % clean rows/cols with only one nnz
%     res = res(:,(sum(res,1)-1)>0);
%     res = res((sum(res,2)-1)>0,:);

    res = NaipeSD;

    res = bwareaopen(res,3);

    % clean corner noise
%     sx = round(0.1*size(res,1));
%     sy = round(0.1*size(res,2));
%     res(1:sx,1:sy) = 0;
%     res(1:sx,end-sy:end) = 0;
%     res(end-sx:end,end-sy:end) = 0;
%     res(end-sx:end,1:sy) = 0;
    


    % clean all zeros rows/cols
    res = res(:,any(res,1));
    res = res(any(res,2),:);
end



 

