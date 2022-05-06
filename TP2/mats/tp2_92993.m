% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Trabalho Pratico 2

%% 

% • Criar uma base dos objetos de referência (a partir das imagens de
% referência fornecidas). << done
% >>> • Identificar e estabelecer os descritores relevantes para distinguir os
% diversos objetos. << falha nalgumas com invmoments
% >>>> • Nas imagens a processar separar os objetos do fundo (binarização, deteção de contornos, ou
% outros, são técnicas esperadas para o fazer). << (alterar para fundos
% complexos)
% • Eliminar os objetos em contacto com o bordo da imagem. << done
% >>>>>> • Identificar e eliminar os objetos partidos. << ?? area e perimetro?? <<<
% • Obter os descritores dos objetos restantes (na representação em máscara binária e/ou na
% representação completa a cores). << done com invmoments
% • Classificar os objetos por comparação de descritores com os objetos de referência mediante
% critérios de distância, ou outros. << done com invmoments
% • Contar as ocorrências de cada classe/tipo de objetos e atualizar o
% ficheiro de resultados. << done




%%

% Oreos pretas, bolacha vermelha
% encontrar zona oreo mal binarizada e tentar tratar so dessa zona
% estender a circulos as bolachas para perceber se sao partidas


%% 


function NumMec = tp2_92993()

    close all
    clear all
    clc

    %% Init Vars
    NumMec = 92993;
    
    %% Open Image
    
%     addpath('../')

    addpath('../Seq29x')

    listaF=dir('../Seq29x/svpi2022_TP2_img_*.png');
    fileExact = fopen("svpi2022_tp2_seq_ALL.txt","r"); nLineExact = 0;

%     imgRef1 = im2double(imread("../svpi2022_TP2_img_001_01.png"));
%     lista1=dir('../Seq29x/svpi2022_TP2_img_*1_*.png');
%     fileExact1 = fopen("svpi2022_tp2_seq_291.txt","r"); nLineExact = 0;
%     classe = 1;

%     imgRef2 = im2double(imread("../svpi2022_TP2_img_002_01.png"));
%     listaF=dir('../Seq29x/svpi2022_TP2_img_*2_*.png');
%     fileExact = fopen("svpi2022_tp2_seq_292.txt","r"); nLineExact = 0;
%     classe = 2;

    MaxImg = size(listaF,1);

    showplot = false;

    idxImg = 3; showplot = true;
   
%     for idxImg = 1:MaxImg

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        classe = str2double(imName(20));
        NumImg = str2double(imName(22:23));
        
        A0 = im2double(imread(imName));

        A = im2double(rgb2gray(imread(imName)));

%         A = medfilt2(filter2(fspecial("average",3),A));

%         showplot = false;

        if showplot
            figure(1)
            imshow(A0)
            figure(2)
            imshow(A)
        end

        %% Reference subimages
        
        [regionsRef,regionsRGBRef,bigRefArea] = getRefImages(classe);

        N = numel(regionsRef);
        SS = ceil(sqrt(N));
        
        invMRef = zeros(7,N);
        for k=1:N
            invMRef(:,k) = invmoments(regionsRef{k});
        end

        
        if showplot
            figure(20)
            for k=1:N
                subplot(SS,SS,k)
                imshow(regionsRef{k})
                xlabel(k)
            end
        end
        
        if showplot
            figure(21)
            for k=1:N
                subplot(SS,SS,k)
                imshow(regionsRGBRef{k})
                xlabel(k)
            end
        end

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

        %% Binarizar imagem

        minTh = 0.01; % 0.01;

        
%         E = double(A>minTh);

%         B = medfilt2(filter2(fspecial("average",3),A));
        B=A;
        T = adaptthresh(B);

        E = imbinarize(A,T);

        F = imclearborder(E);

        F = bwareaopen(F,100);
        F = bwmorph(F,"close",inf);
        F = imfill(F,"holes");


        [Bx,L,Nb,~] = bwboundaries(F);

        figure(100)
        imshow(F)
        hold on

        ObjOK = 0;
        sx = size(A,1);
        sy = size(A,2);

        for k = 1:Nb
            boundary = Bx{k};
            mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
            if nnz(mask) < 300, continue, end 
            
            ObjOK = ObjOK + 1;

%             plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%             pause(0.1)
        end

%         return

%         G = (D>minTh);
%         G = bwmorph(G,"close",inf);
%         G = bwareaopen(G,50);
%         G = bwmorph(G,"close",inf);
        G = imadjust(E.*not(F));

%         [~,Nb] = bwlabel(G);
        [Bx,L,Nb,~] = bwboundaries(G,'noholes');

%         figure(101)
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
        
%         A = F;
        minSize = 0.1; % 0.2  min nnz for aceptable boundary (percentage)
        minWidth = 0.01; % 0.04 min width of subimage (percentage)

        
        minAreaMigalha = 0.05 * bigRefArea;

        fmaskRot = zeros(size(A));
        cutx = -3; 
        cuty = -3; 
        extend = false; %true;
        relSizes = 5; %3

        % Find other subimages
%         try 
        [regions,regionsRGB] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,A0,minAreaMigalha);
%         catch
%             fprintf(">>>>>>>>>>>>>>>>fail binarization %d\n",idxImg)
% %             continue
%         end

        return

        N = numel(regions);
        SS = ceil(sqrt(N));

        ObjPart = 0;
        ObjOK = 0;
        
        count = 1;
        for k=1:N
            rBin = regions{k} > 0;
            if mean(rBin) < 0.9  % Detetar Partidas <<< MUDAR
                str{k} = sprintf("partida,k:%d\n",k);
%                 fprintf(str{k})
%                 ObjPart = ObjPart + 1;
            else
                str{k} = sprintf("OK,k:%d\n",k);
%                 ObjOK = ObjOK + 1;
                regionsOK{count} = regions{k};
                regionsOKRGB{count} = regionsRGB{k};
                count = count + 1;
            end
        end

        invM = zeros(7,N);
        if showplot
            figure(31)
            for k=1:N
                subplot(SS, SS, k);
                imshow(regions{k})
                
                xlabel(str{k})
            end
        end

        if showplot
            figure(32)
            for k=1:N
                subplot(SS, SS, k);
                imshow(regionsRGB{k})
                xlabel(str{k})
            end
        end

        %% Classificar bolachas inteiras

        N = numel(regions);
        SS = ceil(sqrt(N));

        if showplot
            figure(41)
            for k=1:N
                subplot(SS, SS, k);
                imshow(regions{k})
                xlabel(k)
            end
        end

        if showplot
            figure(42)
            for k=1:N
                subplot(SS, SS, k);
                imshow(regionsRGB{k})
                xlabel(k)
            end
        end

        return 

        %% Compare

        
        matchs = zeros(k,1);
        resx = zeros(k,1);
        partidas = [];

        for k=1:N
            invM(:,k) = invmoments(regions{k});
%             [kRef,res] = getBestMatch(invM(:,k),invMRef);
            [kRef,res,part] = getBestMatchFull(regionsRGB{k},regionsRGBRef);

            if  showplot
                figure(300)
                subplot(1,2,1)
                imshow(regionsRGB{k})
                if part
                    xlabel(sprintf("part%d",k))
                else
                    xlabel(k)
                end
    
                subplot(1,2,2)
                imshow(regionsRGBRef{kRef})
                xlabel(sprintf("Kref:%d,res:%f",kRef,res))

                pause(1)
            end

            matchs(k) = kRef;
            resx(k) = res;

            if part
                fprintf("Partida:%d\n",k)
                ObjPart = ObjPart + 1;
                partidas = [partidas k];
            else
                ObjOK = ObjOK + 1;
            end
        end

        %% Descartar partidas

        if showplot
            resx
            partidas

        end


        %% Contar
        
        for k=1:N
            if ismember(k,partidas), continue, end
            if classe == 1
                if matchs(k) < 3
                    beurre = beurre + 1;
                elseif matchs(k) < 5
                    choco = choco + 1;
                elseif matchs(k) < 7
                    confit = confit + 1;
                elseif matchs(k) < 9
                    craker = craker + 1;
                elseif matchs(k) < 11
                    fan = fan + 1;
                elseif matchs(k) < 13
                    ginger = ginger + 1;
                elseif matchs(k) < 15
                    lotus = lotus + 1;
                elseif matchs(k) < 17
                    maria = maria + 1;
                elseif matchs(k) < 19
                    oreo = oreo + 1;
                elseif matchs(k) < 21
                    palmier = palmier + 1;
                elseif matchs(k) < 23
                    parijse = parijse + 1;
                elseif matchs(k) < 25
                    sugar = sugar + 1;
                elseif matchs(k) < 27
                    wafer = wafer + 1;
                else
                    zebra = zebra + 1;
                end
            else
                if matchs(k) == 1
                    beurre = beurre + 1;
                elseif matchs(k) == 2
                    choco = choco + 1;
                elseif matchs(k) == 3
                    confit = confit + 1;
                elseif matchs(k) == 4
                    craker = craker + 1;
                elseif matchs(k) == 5
                    fan = fan + 1;
                elseif matchs(k) == 6
                    ginger = ginger + 1;
                elseif matchs(k) == 7
                    lotus = lotus + 1;
                elseif matchs(k) == 8
                    maria = maria + 1;
                elseif matchs(k) == 9
                    oreo = oreo + 1;
                elseif matchs(k) == 10
                    palmier = palmier + 1;
                elseif matchs(k) == 11
                    parijse = parijse + 1;
                elseif matchs(k) == 12
                    sugar = sugar + 1;
                elseif matchs(k) == 13
                    wafer = wafer + 1;
                else
                    zebra = zebra + 1;
                end
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

function [kRef,minres] = getBestMatch(invM,invMRef)

    minres = inf;
    kRef = 1;
    for k = 1:size(invMRef,2)

        elem = invMRef(:,k)./sum(invMRef(:,k));
        invM = invM./sum(invM);

%         elem = invMRef(:,k);

%         res = sum(abs((elem-invM)./elem));

        res = sqrt(sum((elem-invM).^2)); % Euclidean Distance

        if res < minres 
            minres = res;
            kRef = k;
        end
        
    end
end

function res = getDiffRGB(img1,img2)
    ola = imresize(img1,'OutputSize',size(img2,[1 2]));
    res = mean(img2 - ola,'all');
end

function [kRef,minres,part] = getBestMatchFull(img1,regionsRGBRef)
    minresT = zeros(length(regionsRGBRef),1);
    
%     for k=1:length(regionsRGBRef)
%         ola = imresize(img1,'OutputSize',size(regionsRGBRef{k},[1 2]));
% %         ola2 = sum(regionsRGBRef{k} - ola,'all');
%         minresT(k) = sum(abs(regionsRGBRef{k} - ola),'all');
%         fprintf("k:%d,Totaldiff:%.3f\n",k,minresT(k))
%     end
% 
%     [minres,kRef] = min(minresT); 
%     part = false;
%     
%     return

    % RGB Images
    for i=1:3
        invM = invmoments(img1(:,:,i));
        for k = 1:length(regionsRGBRef)
            img2 = regionsRGBRef{k};
            invMRef = invmoments(img2(:,:,i));

%             imgB = imresize(img1(:,:,i),size(img2(:,:,i)));
%             invM = invmoments(imgB);
    
            elem = invMRef./sum(invMRef);
            invM = invM./sum(invM);
    
       %         res = sum(abs((elem-invM)./elem));
    
            res = sqrt(sum((elem-invM).^2)); % Euclidean Distance
    
            minresT(k) = minresT(k) + res;            
        end
    end

    % GrayScale Images
    img1 = rgb2gray(img1);
    invM = invmoments(img1);

    for k = 1:length(regionsRGBRef)
        img2 = rgb2gray(regionsRGBRef{k});
        invMRef = invmoments(img2);

%         imgB = imresize(img1,size(img2));
%         invM = invmoments(imgB);

        elem = invMRef./sum(invMRef);
        invM = invM./sum(invM);

   %         res = sum(abs((elem-invM)./elem));

        res = sqrt(sum((elem-invM).^2)); % Euclidean Distance

        minresT(k) = minresT(k) + res;            
    end
    
    [minres,kRef] = min(minresT); 

    imgPrev = img1;

    img1 = img1 > 0.01;
    img1 = bwmorph(img1,'close',inf);
    img1 = imfill(img1,'holes');
    
    img2 = rgb2gray(regionsRGBRef{kRef}) > 0.01;
    img2 = bwmorph(img2,'close',inf);
    img2 = imfill(img2,'holes');

    img1 = imresize(img1,size(img2));

%     figure(27)
%     subplot(1,2,1)
%     imshow(img1)
%     subplot(1,2,2)
%     imshow(img2)
%     pause(2)

    A = zeros(2,1);
    P = zeros(2,1);
    Ap = zeros(2,1);

    % Areas
    A(1) = nnz(img1);
    A(2) = nnz(img2);

    % Perimeter
    P(1) = nnz(bwperim(img1));
    P(2) = nnz(bwperim(img2));
    

    isRound(1) = 4*pi*A(1)/P(1)^2;
    isRound(2) = 4*pi*A(2)/P(2)^2;

    isRoundRef = ismember(kRef,[3,4,6,7,11,12,15,16,17,18,23,24,27,28]);

    % P/A
    Ap(1) = P(1)/A(1);
    Ap(2) = P(2)/A(2);

%     ApNorm = P1*A2/A1 / A1 * A2 P(2)
    ApNorm = P(1)/P(2) / A(1)^2 * A(1) ^ 2;

    Ap = sort(Ap);
    A = sort(A);
    P = sort(P);

    part = false;

%     ths = [1.1 1.3 1.1 0.35];
%     ths = [1.2 100 1.2 0.35 0.5 0.94];
    ths = [1.2 100 100 0.8 0.7 0.8 0.16 0.95];

    ths(6) = 1-ths(6);
    isRound = abs(1-isRound);

    temp1 = imgPrev;
    temp1(imgPrev<0.01) = nan;
    avgColor(1) = mean(temp1,"all","omitnan");

    temp2 = regionsRGBRef{kRef};
    temp2(regionsRGBRef{kRef}<0.01) = nan;
    avgColor(2) = mean(temp2,"all","omitnan");

%     if (Ap(2) > ths(1) * Ap(1) ...
    if (ApNorm > ths(1)  ...
            || A(2) > ths(2) * A(1) ...
            || P(2) > ths(3) * P(1) ...
            || minres > ths(4) ...
            || mean(img1,'all') < ths(5) * mean(img2,'all')  ...
            || (isRound(1) > ths(6) && isRoundRef) ... % && isRound(2) < ths(6)) invert <
            || getDiffRGB(imgPrev,regionsRGBRef{kRef}) > ths(7) ...
            || avgColor(1) < ths(8) * avgColor(2))
%     if (Ap(2) > ths(1) * Ap(1) && A(2) > ths(2) * A(1) &&  P(2) > ths(3) * P(1)) ||  minres > ths(4)
        part = true;
        fprintf("fail ")


        if ApNorm > ths(1) %Ap(2) > ths(1) * Ap(1) 
%             fprintf("Ap;%.2f ",Ap(2)/Ap(1))   
            fprintf("Ap;%.2f ",ApNorm)   
        end
        if A(2) > ths(2) * A(1) 
            fprintf("A;%.2f ",A(2)/A(1))   
        end
        if P(2) > ths(3) * P(1)
            fprintf("P:%.2f ",P(2)/P(1))       
        end

        if minres > ths(4)
            fprintf("MR:%.2f ",minres)       
        end

        if mean(img1,'all') < ths(5) * mean(img2,'all')
            fprintf("mean:%.2f ",mean(img1,'all')/mean(img2,'all'))
        end

        if (isRound(1) > ths(6) && isRoundRef)
            fprintf("round: %.2f %2f ",isRound(1), isRound(2))
        end

        if getDiffRGB(imgPrev,regionsRGBRef{kRef}) > ths(7)
           fprintf("diffRGB: %.2f ",getDiffRGB(imgPrev,regionsRGBRef{kRef}))
        end

        if avgColor(1) < ths(8) * avgColor(2)
            fprintf("diffColor: %.2f ",avgColor(1)/avgColor(2))
        end

        fprintf("\n")
    end
    
end

function [regions,regionsRGB,bigRefArea] = getRefImages(classe)

    if classe == 1
        imgRef = im2double(imread("../svpi2022_TP2_img_001_01.png"));
    else
        imgRef = im2double(imread("../svpi2022_TP2_img_002_01.png"));
    end
    
    A = rgb2gray(imgRef);
    minSize = 0.1;
    relSizes = 2.5;
    minWidth = 0.05;
    fmaskPrev = zeros(size(A));
    
    A = A <1;

    B = maskNormal(A);
    
    B = bwareaopen(B,round(minSize*size(B,1)));
    
    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

    
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

        % compute better order for cookies
        [y, x] = ndgrid(1:size(mask, 1), 1:size(mask, 2));
        centroid = round(mean([x(logical(mask)), y(logical(mask))]));
        
        divx = int16(round(sx/10));
        divy = int16(round(sy/10));
        locals(count) = (int16(centroid(1))/divy) + (int16(centroid(2))/divx)*10 + 2;
        
        
        count = count + 1;
    
    end

    % compute better order for cookies
    [~,sortedIdx] = sort(locals);
    regionsOld = regions;
    regionsRGB_old = regionsRGB;

    for i = 1:count-1
        regions{i} = regionsOld{sortedIdx(i)};
        regionsRGB{i} = regionsRGB_old{sortedIdx(i)};
    end

    bigRefArea = inf;
    for k=1:length(regions)
        area = bwarea(regions{k});
        if area < bigRefArea
            bigRefArea = area;
        end
    end

end

function B = maskNormal(A)
    % mask for all other subimages
    
    B = edge(A,'roberts') | edge(A,'sobel');
    B = bwmorph(B,'close',inf);
end


function [regions,regionsRGB,fullMask] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef,minAreaMigalha)

    % get all subimages(regions)

%     B = maskNormal(A);
%     
%     B = bwareaopen(B,round(minSize*size(B,1)));

%         B = medfilt2(filter2(fspecial("average",3),A));
    B = A;
    T = adaptthresh(B);

    E = imbinarize(A,T);

    F = imclearborder(E);

    F = bwareaopen(F,100);
    F = bwmorph(F,"close",inf);
    F = imfill(F,"holes");
    B = F;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

    figure(751)
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
        selectedRGB = imgRef.*repmat(mask,[1 1 3]);

        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
    
        % guardar regiao
        selectedRGB = selectedRGB(:,any(selected,1),:);
        selectedRGB = selectedRGB(any(selected,2),:,:);

        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);

        if (nnz(mask) < minAreaMigalha)
            figure(300)
            imshow(selectedRGB)
            fprintf("migalha\n")
            pause(2)
            continue
        end
        
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

        
        count = count + 1;
    
    end
end

