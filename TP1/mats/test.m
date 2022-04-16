close all
clear all


% addpath('../sequencias/Seq160')
% listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

addpath('../sequencias/Seq530')
listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');


idxImg = 1;
imName = listaF(idxImg).name;


A = im2double(imread(imName));

regions=vs_getsubimages(A); %extract all regions
regions2=vs_getsubimages(A); %working
N=numel(regions);
SS=ceil(sqrt(N));

for k=1:N 
%     regions2{k} = medfilt2(filter2(fspecial('average',3),regions{k}));
    regions2{k} = medfilt2(regions{k});
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

%         if k==37
%             figure(100)
%             imshow(B)
%             pause(10)
%         end
        
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

% cartas = [cartas1k cartas2k];
% 
% 
% cut = 2; % 5
% nrows = 3;
% 
% copa = getCopaMatrix();
% ouro = strel('diamond',250).Neighborhood;
% tolOuros = 0.12;
% tolCopas = 0.12;
% 
% 
% figure(4)
% means = zeros(length(cartas),1);
% meansCopa = zeros(length(cartas),1);
% 
% idx = 1;
% for k=cartas
%     
%     B = regions{k}(cut:end-cut,cut:end-cut);
% 
%     dx = round(px*size(B,1)); 
%     dy = round(py*size(B,2));
% 
%     if ismember(k,cartas1k)
%         tipo=1;
%         CantoSup = rot90(B(1:dx,end-dy:end));
%     else
%         tipo=2;
%         CantoSup = rot90(rot90(rot90(B(1:dx,1:dy))));
%     end
% 
%     subplot(nrows,length(cartas),idx)
%     imshow(CantoSup)
% 
%     Naipe0 = getNaipe0(B,tipo,px,py);
% 
% 
%     subplot(nrows,length(cartas),idx + length(cartas))
% %     imshow(getNaipe(B,tipo,px,py))
%     imshow(Naipe0)
% 
% %     per = bwperim(Naipe0);
% %     Ar = bwarea(Naipe0);
% 
% %     subplot(nrows,length(cartas),idx + length(cartas)*2)
% %     imshow(per)
% 
% %     P = nnz(per);
% 
% %     T = size(Naipe0,1)*size(Naipe0,2);
% 
% %     [L,Nb] = bwlabel(Naipe0);
% 
% %     xlabel(sprintf("T%d,P:%.2f,A:%.2f \n P/A:%.2f,Nb:%d\nP/T:%.2f,A/T:%.2f",tipo,P,Ar,P/Ar,Nb,P/T,Ar/T))
%     
%     [res,means(idx)] = classNaipe(B,tipo,ouro,px,py,tolOuros);
%     if res
%         xlabel(sprintf("T%d,O:%.2f,C:%.2f\nOuros",tipo,means(idx),meansCopa(idx)))
%     else
%         [res,meansCopa(idx)] = classNaipe(B,tipo,copa,px,py,tolCopas);
%         if res
%             xlabel(sprintf("T%d,O:%.2f,C:%.2f\nCopas",tipo,means(idx),meansCopa(idx)))
%         else
%             xlabel(sprintf("T%d,O:%.2f,C:%.2f",tipo,means(idx),meansCopa(idx)))
%         end
%     end
% 
%     idx = idx + 1;
% 
% end


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
%         B(1:dx,1:dy) = 0;
%         B(end-dx:end,end-dy:end) = 0;

        res=1;
        
    elseif (nnzInfDir > perc*area && nnzSupEsq > perc*area && ...
            nnzInfEsq < perc0*area && nnzSupDir < perc0*area)
%         B(end-dx:end,1:dy) = 0;
%         B(1:dx,end-dy:end) = 0;
        
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

%     clean0s = getNaipe(carta,tipo,px,py);
    clean0s = getNaipe0(carta,tipo,px,py);


%     naipe = bwmorph(naipe,'remove'); % usar so a border
%     clean0s = imresize(clean0s,sc);
%     clean0s = bwmorph(clean0s,'remove');
% %     clean0s = bwmorph(clean0s,'dilate');
%     naipe = imdilate(naipe,ones(1,3));
%     meanC = mean(clean0s~=imresize(naipe,size(clean0s)),'all');




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
    res = CSbin(dx2:end,:);

    % clean rows/cols with only one nnz
%     res = res(:,(sum(res,1)-1)>0);
%     res = res((sum(res,2)-1)>0,:);

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

function res = getNaipe0(carta, tipo,px, py)

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
    res = CSbin(dx2:end,:);

    [~,Nb] = bwlabel(res);

    ola = bwmorph(res,'shrink', inf);
    ppi = filter2([1 1 1; 1 -8 1; 1 1 1], ola);
%     marker = (abs(ppi)==8);
%     acept = 7;
%     while nnz(marker)==0
%         marker = (abs(ppi)>acept);
%         acept = acept - 1;
%     end
    marker = (abs(ppi)>5);
    indexes = find(marker);
    prev = res;
    curArea = 0;

    for i=1:length(indexes)
        mk2 = false(size(marker));
        mk2(indexes(i)) = true;
        temp = imreconstruct(mk2, prev);
        Ar = bwarea(temp);
        if  Ar > curArea
            curArea = Ar;
            res = temp;
        end
    end
    

    % clean all zeros rows/cols
    res = res(:,any(res,1));
    res = res(any(res,2),:);

end

 

