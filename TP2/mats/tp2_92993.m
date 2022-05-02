% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Trabalho Pratico 2

%% 

% • Criar uma base dos objetos de referência (a partir das imagens de referência fornecidas).
% • Identificar e estabelecer os descritores relevantes para distinguir os diversos objetos.
% • Nas imagens a processar separar os objetos do fundo (binarização, deteção de contornos, ou
% outros, são técnicas esperadas para o fazer).
% • Eliminar os objetos em contacto com o bordo da imagem.
% • Identificar e eliminar os objetos partidos.
% • Obter os descritores dos objetos restantes (na representação em máscara binária e/ou na
% representação completa a cores).
% • Classificar os objetos por comparação de descritores com os objetos de referência mediante
% critérios de distância, ou outros.
% • Contar as ocorrências de cada classe/tipo de objetos e atualizar o ficheiro de resultados.


%% 


function NumMec = tp2_92993()

    close all
    clear all
    clc

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

    imgRef1 = im2double(imread("../svpi2022_TP2_img_001_01.png"));
    imgRef2 = im2double(imread("../svpi2022_TP2_img_002_01.png"));

%     lista1=dir('../Seq29x/svpi2022_TP2_img_*1_*.png');
%     fileExact1 = fopen("svpi2022_tp2_seq_291.txt","r"); nLineExact = 0;
% 
%     lista2=dir('../Seq29x/svpi2022_TP2_img_*2_*.png');
%     fileExact2 = fopen("svpi2022_tp2_seq_292.txt","r"); nLineExact = 0;

    MaxImg = size(listaF,1);

%     showplot = false;

    idxImg = 1; showplot = true;
   
%     for idxImg = 1:MaxImg

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));
        
        A0 = im2double(imread(imName));

        A = im2double(rgb2gray(imread(imName)));

        if showplot
            figure(1)
            imshow(A0)
            figure(2)
            imshow(A)
        end

        %% Reference subimages

        [regionsRef,regionsRGBRef] = getRefImages(1);

        %% Vars
        ObjBord = 0; % numero de objs a tocar o bordo (nao para classificar)
        ObjPart = 0; % numero de objs partidos (nao para classificar)
        ObjOK = 0; % numero de objs para classificar (migalhas nao contam (0.05% do obj inicial))
        
        % Bolachas;
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

        %% 

        B = imadjust(imclearborder(A));
        D = imadjust(A.*not(B));

        minTh = 0.01;

        E = (A>minTh);
%         E = bwmorph(E,"close",inf);
%         E = bwareaopen(E,50);

        E = double(E);

%         F = (B>minTh);
%         F = bwmorph(F,"close",inf);
%         F = bwareaopen(F,100);
%         F = bwmorph(F,"fill",100);

%         F = imadjust(imclearborder(E));
        F = imclearborder(E);

        [Bx,L,Nb,~] = bwboundaries(F,'noholes');

%         figure(100)
%         imshow(label2rgb(L, @jet, [.5 .5 .5]))
%         hold on

        ObjOK = 0;
        sx = size(A,1);
        sy = size(A,2);

        for k = 1:Nb
            boundary = Bx{k};
            mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
            if nnz(mask) < 200, continue, end 
            
            ObjOK = ObjOK + 1;

%             plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
%             pause(1)
        end

%         G = (D>minTh);
%         G = bwmorph(G,"close",inf);
%         G = bwareaopen(G,50);
%         G = bwmorph(G,"close",inf);
        G = imadjust(E.*not(F));

%         [~,Nb] = bwlabel(G);
        [Bx,L,Nb,~] = bwboundaries(G,'noholes');

%         figure(100)
%         imshow(label2rgb(L, @jet, [.5 .5 .5]))
%         hold on

        ObjBord = 0;
        sx = size(A,1);
        sy = size(A,2);

        for k = 1:Nb
            boundary = Bx{k};
            mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
            if nnz(mask) < 100, continue, end 
            
            ObjBord = ObjBord + 1;

%             plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
%             pause(1)
        end

        if showplot
            figure(11)
            subplot(1,3,1)
            imshow(E)

            subplot(1,3,2)
            imshow(F)
            xlabel(ObjOK)
            
            subplot(1,3,3)
            imshow(G)
            xlabel(ObjBord)
        end
        
        
        
        
        %% SubImages
        
%         minSize = 0.1; % 0.2  min nnz for aceptable boundary (percentage)
%         minWidth = 0.01; % 0.04 min width of subimage (percentage)
% 
%         fmaskRot = zeros(size(A));
%         cutx = -1; 
%         cuty = -1; 
%         extend = false; %true;
%         relSizes = 5; %3
% 
%         % Find other subimages
%         [regionsNormal,~] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot);
% 
%         regions = regionsNormal;
%         N = numel(regions);
%         SS = ceil(sqrt(N));
% 
%         if showplot
%             figure(3)
%             for k=1:N
%                 subplot(SS, SS, k);
%                 imshow(regions{k})
%                 xlabel(k)
%             end
%         end
        
        
        %% Normal
    
%         for k = 1:N
%             
%             B = imadjust(regions{k});
%             B = medfilt2(B);
%             B = imadjust(B);
%             B = autobin2th(B);
%             B = bwmorph(B,'close',inf);
%         
%         
%             sx = size(B,1);
%             sy = size(B,2);
%         
%             % Test Noise
%             C = bwmorph(B,'erode',2);
%             minNNZ =  0.01 * nnz(B) +1;
%             if nnz(C) < minNNZ 
%                 str = "noise";
%             else
%                 if sx > sy % rotate to horizontal
%                     regions{k} = rot90(regions{k});
%                 end
%         
%         
%                 B = double(regions{k});
%                 
%                 B = autobin(imadjust(B));
%                 
%                 B = edge(B,'roberts');
%                 B = bwmorph(B,'bridge');
%                 B = bwareaopen(B,round(0.5*size(B,1)));
%     
%     
%                 [~,Nb] = bwlabel(B);
%     
%                 if (Nb>9 || Nb==0)
%                     fprintf("Remove Carta:k=%d,Nb=%d\n",k,Nb);
%                     str = sprintf("Remove Carta:k=%d,Nb=%d\n",k,Nb);
%                 else
%                 
%                     str = sprintf("k=%d\n Desc.,Nb=%d",k,Nb);
%                 end
%             end 
% 
%             if showplot
%                 subplot(SS,SS,k);
%                 imshow(B)
%                 xlabel(str);
%             end
%         
%         end
  
        
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


function [regions,regionsRGB] = getRefImages(classe)

    if classe == 1
        imgRef1 = im2double(imread("../svpi2022_TP2_img_001_01.png"));
    else
        imgRef1 = im2double(imread("../svpi2022_TP2_img_002_01.png"));
    end
    
    A = rgb2gray(imgRef1);
    minSize = 0.1;
    cutx = -1;
    cuty = -1;
    relSizes = 3;
    minWidth = 0.05;
    extend = false;
    fmaskPrev = zeros(size(A));
    fromRef = true;
    
    [regions,regionsRGB,~] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef1,fromRef);

end

function B = maskNormal(A)
    A = A<1;
    % mask for all other subimages
    
    B = edge(A,'roberts') | edge(A,'sobel');
%     B = bwmorph(B,'bridge',inf);
    B = bwmorph(B,'close',inf);
end


function [regions,regionsRGB,fullMask] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef,fromRef)

    
    % get all subimages(regions)

    B = maskNormal(A);
    
    B = bwareaopen(B,round(minSize*size(B,1)));
    
    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

%     figure(20)
%     imshow(B)
%     hold on

    
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

%         plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
%         pause(0.01)
        
        selected = A.*mask;
        selectedRGB = imgRef.*repmat(mask,[1 1 3]);

        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
    
        % guardar regiao
        selectedRGB = selectedRGB(:,any(selected,1),:);
        selectedRGB = selectedRGB(any(selected,2),:,:);

        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);
        
        regions{count} = selected;

        % zona branca da palmier passa a preto

        for i = 1:size(selectedRGB,1)
            for j = 1:size(selectedRGB,2)
                if (sum(selectedRGB(i,j,:)) > 2.98)
                     % White pixel - do what you want to original image
                     selectedRGB(i,j,:) = [0 0 0]; % make it black, for example
                end
            end
        end

        regionsRGB{count} = selectedRGB;

        if fromRef % compute better order for cookies
            [y, x] = ndgrid(1:size(mask, 1), 1:size(mask, 2));
            centroid = round(mean([x(logical(mask)), y(logical(mask))]));
            
            divx = int16(round(sx/10));
            divy = int16(round(sy/10));
            locals(count) = (int16(centroid(1))/divy) + (int16(centroid(2))/divx)*10 + 2;
        end
        
        count = count + 1;
    
    end

    if fromRef % compute better order for cookies
        [~,sortedIdx] = sort(locals);
        regionsOld = regions;
        regionsRGB_old = regionsRGB;
    
        for i = 1:count-1
            regions{i} = regionsOld{sortedIdx(i)};
            regionsRGB{i} = regionsRGB_old{sortedIdx(i)};
        end
    end
end

