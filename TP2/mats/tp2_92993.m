% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Trabalho Pratico 2

function NumMec = tp2_92993()

    %% Init Vars
    NumMec = 92993;
    
    %% Open Image
    
%     addpath('../')
%     listaF=dir('../svpi2022_TP2_img_*.png');
%     lista1=dir('../svpi2022_TP2_img_*1_*.png');
%     lista2=dir('../svpi2022_TP2_img_*2_*.png');


    addpath('../Seq29x')

    listaF=dir('../Seq29x/svpi2022_TP2_img_*.png');
    fileExact = fopen("svpi2022_tp2_seq_ALL.txt","r"); nLineExact = 0;

%     lista1=dir('../Seq29x/svpi2022_TP2_img_*1_*.png');
%     fileExact1 = fopen("svpi2022_tp2_seq_291.txt","r"); nLineExact = 0;
% 
%     lista2=dir('../Seq29x/svpi2022_TP2_img_*2_*.png');
%     fileExact2 = fopen("svpi2022_tp2_seq_292.txt","r"); nLineExact = 0;

    MaxImg = size(listaF,1);

%     showplot = false;

    idxImg = 5; showplot = true;
   
%     for idxImg = 1:MaxImg

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));
        
        A0 = im2double(imread(imName));

        if showplot
            figure(1)
            imshow(A0)
        end

        A = im2double(rgb2gray(imread(imName)));

        if showplot
            figure(2)
            imshow(A)
        end
        
        %% SubImages
        
        minSize = 0.1; % 0.2  min nnz for aceptable boundary (percentage)
        minWidth = 0.01; % 0.04 min width of subimage (percentage)

        fmaskRot = zeros(size(A));
        cutx = -1; 
        cuty = -1; 
        extend = false; %true;
        relSizes = 5; %3

        % Find other subimages
        [regionsNormal,~] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot);

        regions = regionsNormal;
        N = numel(regions);
        SS = ceil(sqrt(N));

        if showplot
            figure(3)
            for k=1:N
                subplot(SS, SS, k);
                imshow(regions{k})
                xlabel(k)
            end
        end
        
        
        %% Vars
        ObjBord = 0;
        ObjPart = 0;
        ObjOK = 0;
        beurre = 0;
        choco = 0;
        confit = 0;
        craker = 0;
        fan = 0;
        ginger = 0;
        lotus = 0;
        maria = 0;
        oreo = 0;
        palmier = 0;
        parijse = 0;
        sugar = 0;
        wafer = 0;
        zebra = 0;
        
        %% Normal
    
        for k = 1:N
            
            B = imadjust(regions{k});
            B = medfilt2(B);
            B = imadjust(B);
            B = autobin2th(B);
            B = bwmorph(B,'close',inf);
        
        
            sx = size(B,1);
            sy = size(B,2);
        
            % Test Noise
            C = bwmorph(B,'erode',2);
            minNNZ =  0.01 * nnz(B) +1;
            if nnz(C) < minNNZ 
                str = "noise";
            else
                if sx > sy % rotate to horizontal
                    regions{k} = rot90(regions{k});
                end
        
        
                B = double(regions{k});
                
                B = autobin(imadjust(B));
                
                B = edge(B,'roberts');
                B = bwmorph(B,'bridge');
                B = bwareaopen(B,round(0.5*size(B,1)));
    
    
                [~,Nb] = bwlabel(B);
    
                if (Nb>9 || Nb==0)
                    fprintf("Remove Carta:k=%d,Nb=%d\n",k,Nb);
                    str = sprintf("Remove Carta:k=%d,Nb=%d\n",k,Nb);
                else
                
                    str = sprintf("k=%d\n Desc.,Nb=%d",k,Nb);
                end
            end 

            if showplot
                subplot(SS,SS,k);
                imshow(B)
                xlabel(str);
            end
        
        end
  
        
        %% Save Vars
                
        %

        %% Show vars

        if showplot
            fprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d," + ...
                "%d,%d,%d,%d\n", NumMec, NumSeq, NumImg, ObjBord, ObjPart, ...
                ObjOK, beurre, choco, confit, craker, fan, ginger, lotus, ...
                maria, oreo , palmier, parijse, sugar, wafer, zebra);
            fprintf("Esperado:\n");
            while nLineExact < idxImg
                strExact = fgets(fileExact);
                nLineExact = nLineExact + 1;
            end
            fprintf("%s\n",strExact(2:end));            
        end
       
        
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, ObjBord, ObjPart, ObjOK, beurre, ...
                choco, confit, craker, fan, ginger, lotus, maria, oreo , ...
                palmier, parijse, sugar, wafer, zebra);

        writetable(T,'tp2_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')

%     end

        if showplot
            save
        end
    

end


function Ibin = autobin(I)
    Ibin = double(imbinarize(I));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end

function Ibin = autobin2th(I) % autobin but for 2 thresholds
    warning ('off','all');
    [ts,met] = multithresh(I,2);
    warning ('on','all');

    if met==0 % invalid 2nd threshold
        Ibin = double(imbinarize(I));
    else
        T = (ts(1)+ts(2))/2;
        Ibin = double(imbinarize(I,T));
    end  
    
    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end


function B = maskNormal(A)
    % mask for all other subimages
    
%     B = edge(A,'roberts');
%     B = bwmorph(B,'bridge');

    B = edge(A,'roberts') | edge(A,'sobel');
%     B = bwmorph(B,'bridge',inf);
    B = bwmorph(B,'close',inf);
end

function [regions,fullMask] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev)
    % get all subimages(regions)

    B = maskNormal(A);
    
    B = bwareaopen(B,round(minSize*size(B,1)));
    
    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

    figure(20)
    imshow(B)
    hold on
    
    for k = Nb+1:length(Bx) % use only interior boundaries
        boundary = Bx{k};
    
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
        if (nnz(mask) < minSize*sx), continue, end
    
        % remove already found
        if nnz(mask.*fmaskPrev), continue, end
    
        % clean all zeros cols and rows
        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);
        if (mean(mask0s,'all') < 0.4), continue, end % very sparse image
    
        % remove weird shapes
        sizesT = sort(size(mask0s));
        if sizesT(2) > relSizes * sizesT(1) || sizesT(1) < minWidth * sx, continue, end
    
        % estender a quadrilateros
        if extend
            idxs = find(max(mask,[],2));
            minx = max(idxs(1)+cutx,1);
            maxx = min(idxs(end)-cutx,sx);
            idxs = find(max(mask));
            miny = max(idxs(1)+cuty,1);
            maxy = min(idxs(end)-cuty,sy);
            mask = zeros(size(A));
            mask(minx:maxx,miny:maxy) = 1;

            % remove already found
            if nnz(mask.*fmaskPrev), continue, end

            % clean all zeros cols and rows
            mask0s = mask(:,any(mask,1));
            mask0s = mask0s(any(mask0s,2),:);

            % remove weird shapes
            sizesT = sort(size(mask0s));
            if sizesT(2) > relSizes * sizesT(1) || sizesT(1) < minWidth * sx , continue, end
        end

        plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
        pause(0.01)
        
        selected = A.*mask;

        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
    
        % guardar regiao
        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);
    
        sizesT = sort(size(selected));
        if sizesT(2) < 1.05 * sizesT(1) % guarantee that dices are squares
            selected = selected(1:sizesT(1),1:sizesT(1));
        end
    
        regions{count} = selected;
        count = count + 1;
    
    end
end
