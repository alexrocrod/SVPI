close 
clear


addpath('../sequencias/Seq160')
listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');


idxImg = 1;
imName = listaF(idxImg).name;


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



figure(3)
for k=1:N 
    subplot( SS, SS, k);
    cut = 2; % 5
    B = autobin(imadjust(regions2{k}(cut:end-cut,cut:end-cut)));
%     B = edging(B);
    
    sx = size(B,1);
    sy = size(B,2);

    if sx ~= sy
        if sx>sy
            B = rot90(B);
        end
        
        dx = round(0.14*size(B,1)); % 0.12
        dy = round(0.25*size(B,2)); % 0.12??
        area = dx*dy;
        CantoSupDir = B(1:dx,end-dy:end);
        nnzSupDir = nnz(CantoSupDir);
        CantoInfDir = B(end-dx:end,end-dy:end);
        nnzInfDir = nnz(CantoInfDir);
        CantoSupEsq = B(1:dx,1:dy);
        nnzSupEsq = nnz(CantoSupEsq);
        CantoInfEsq = B(end-dx:end,1:dy);
        nnzInfEsq = nnz(CantoInfEsq);

%         cIESD = ;
        perc = 0.15;
        perc0 = 0.1;
        if (nnzInfEsq > perc*area && nnzSupDir > perc*area  && ...
                nnzInfDir < perc0*area && nnzSupEsq < perc0*area)
%             Tapar os q nao tem nada
            B(1:dx,1:dy) = 1;
            B(end-dx:end,end-dy:end) = 1;
            imshow(B)
            xlabel("carta 1")
            
        elseif (nnzInfDir > perc*area && nnzSupEsq > perc*area && ...
                nnzInfEsq < perc0*area && nnzSupDir < perc0*area)
            B(end-dx:end,1:dy) = 1;
            B(1:dx,end-dy:end) = 1;
            imshow(B)
            xlabel("carta 2")
        else
            imshow(B)
        end
        
        

        
    
        
    else
        imshow(B)
    end

end

function B = edging(A)
    B = A;
%     B = medfilt2(B);
    B = edge(B,'roberts');
    B = bwareaopen(B,round(0.5*size(B,1)));
    B = bwmorph(B,'close');
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


