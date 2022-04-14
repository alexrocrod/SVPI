close all
clear all


addpath('../sequencias/Seq160')
listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');


idxImg = 2;
imName = listaF(idxImg).name;


A = im2double(imread(imName));

regions=vs_getsubimages(A); %extract all regions
regions2=vs_getsubimages(A); %working
N=numel(regions);
SS=ceil(sqrt(N));

% figure(1)
% for k=1:N 
%     subplot( SS, SS, k);
%     imshow(regions{k})
%     xlabel(k)
% end


% figure(2)
for k=1:N 
%     subplot( SS, SS, k);
    regions2{k} = medfilt2(filter2(fspecial('average',3),regions{k}));
%     imshow(regions2{k})
%     xlabel(k)
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
    B = autobin(imadjust(regions2{k}(cut:end-cut,cut:end-cut)));
%     B = edging(B);
    
    sx = size(B,1);
    sy = size(B,2);

    % Test Noise
    C = bwmorph(B,'erode',2);
    if nnz(C) < 0.01*sx*sy
        fprintf("nnz=%d, m= %d, noise: %d\n",nnz(C),0.01*sx*sy ,k)
        imshow(B)
        xlabel("noise")
    

    elseif sx ~= sy
        if sx>sy
            B = rot90(B);
            regions{k} = rot90(regions{k});
        end
        
        dx = round(px*size(B,1));
        dy = round(py*size(B,2));
        area = dx*dy;
        CantoSupDir = B(1:dx,end-dy:end);
        nnzSupDir = nnz(CantoSupDir);
        CantoInfDir = B(end-dx:end,end-dy:end);
        nnzInfDir = nnz(CantoInfDir);
        CantoSupEsq = B(1:dx,1:dy);
        nnzSupEsq = nnz(CantoSupEsq);
        CantoInfEsq = B(end-dx:end,1:dy);
        nnzInfEsq = nnz(CantoInfEsq);

        if (nnzInfEsq > perc*area && nnzSupDir > perc*area  && ...
                nnzInfDir < perc0*area && nnzSupEsq < perc0*area)
% %             Tapar os q nao tem nada
%             B(1:dx,1:dy) = 1;
%             B(end-dx:end,end-dy:end) = 1;
            B(1:dx,1:dy) = 0;
            B(end-dx:end,end-dy:end) = 0;

            imshow(B)
            xlabel("carta 1")
            cartas1k = [cartas1k k];
            
        elseif (nnzInfDir > perc*area && nnzSupEsq > perc*area && ...
                nnzInfEsq < perc0*area && nnzSupDir < perc0*area)
            B(end-dx:end,1:dy) = 0;
            B(1:dx,end-dy:end) = 0;
            imshow(B)
            xlabel("carta 2")
            cartas2k = [cartas2k k];
        else
            imshow(B)
        end
    
    else
        imshow(B)
    end

    regions2{k} = B;

end


figure(4)
idx = 1;
cut = 2; % 5
nrows = 4;
means = zeros(length(cartas1k),1);
for k=cartas1k
    
%     B = regions2{k};
%     B = regions{k};
    
    B = regions{k}(cut:end-cut,cut:end-cut);
    
    sx = size(B,1);
    sy = size(B,2);

    dx = round(px*size(B,1)); % 0.14
    dy = round(py*size(B,2)); % 0.25??
    CantoSupDir = rot90(B(1:dx,end-dy:end));
    CantoInfEsq = rot90(B(end-dx:end,1:dy));


    subplot(nrows, length(cartas1k), idx);
    imshow(CantoSupDir)

    CSDbin = autobin(imadjust(CantoSupDir));
    CIEbin = autobin(imadjust(CantoInfEsq));

    dx2 = round(0.55*size(CSDbin,1)); 
    NaipeSD = CSDbin(dx2:end,:);
    
    subplot(nrows, length(cartas1k), idx+length(cartas1k));
    imshow(NaipeSD)

    clean0s = NaipeSD(any(NaipeSD,2),:);
    clean0s = clean0s(:,any(clean0s,1));
    subplot(nrows, length(cartas1k), idx+length(cartas1k)*2);
    imshow(clean0s)
   
    ouro1 = strel('diamond',ceil(0.5*size(clean0s,1))).Neighborhood;
    ouro1 = imresize(ouro1,size(clean0s));
    subplot(nrows, length(cartas1k), idx+length(cartas1k)*3);
    imshow(ouro1)
    
    means(idx) = mean(imresize(clean0s,10)~=imresize(ouro1,10),'all');
    if means(idx) < 1e-1
        xlabel("Ouros")
    else
        xlabel(sprintf("Ndif:%.2f",means(idx)))
    end
    

    idx = idx + 1;

end


if not(isempty(cartas2k))
    cartas1k = cartas2k;
    figure(5)
    idx = 1;
    for k=cartas2k
    
    %     B = regions2{k};
    %     B = regions{k};
        B = regions{k}(cut:end-cut,cut:end-cut);
        
        sx = size(B,1);
        sy = size(B,2);
    
        dx = round(px*size(B,1)); % 0.14
        dy = round(py*size(B,2)); % 0.25??
        
        CantoInf = B(end-dx:end,end-dy:end); % dir
        CantoSup = B(1:dx,1:dy); % esq

        subplot(nrows, length(cartas1k), idx);
        imshow(CantoSup)
    
        CSDbin = autobin(imadjust(CantoSup));
        CIEbin = autobin(imadjust(CantoInf));
    
        dx2 = round(0.55*size(CSDbin,1)); 
        NaipeSD = CSDbin(dx2:end,:);
        
        subplot(nrows, length(cartas1k), idx+length(cartas1k));
        imshow(NaipeSD)
    
        clean0s = NaipeSD(any(NaipeSD,2),:);
        clean0s = clean0s(:,any(clean0s,1));
        subplot(nrows, length(cartas1k), idx+length(cartas1k)*2);
        imshow(clean0s)
       
        ouro1 = strel('diamond',ceil(0.5*size(clean0s,1))).Neighborhood;
        ouro1 = imresize(ouro1,size(clean0s));
        subplot(nrows, length(cartas1k), idx+length(cartas1k)*3);
        imshow(ouro1)
        
        means(idx) = mean(imresize(clean0s,10)~=imresize(ouro1,10),'all');
        if means(idx) < 1e-1
            xlabel("Ouros")
        else
            xlabel(sprintf("Ndif:%.2f",means(idx)))
        end
   
        idx = idx + 1;

    end
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


