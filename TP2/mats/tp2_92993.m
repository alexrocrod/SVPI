% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Trabalho Pratico 2


%% Ideias

% Oreos pretas, bolacha vermelha
% encontrar zona oreo mal binarizada e tentar tratar so dessa zona
% estender a circulos as bolachas para perceber se sao partidas


%% Falhas

% Falhas bin? 11a13,24a27 >> fundos diferentes cores
% falha imgidx = 2 nas bolachas iref=5
% falhas imgIdx=4 para oreos e vermelhas

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

    global showplot;
    showplot = false;

    idxImg = 5; showplot = true;
   
%     for idxImg = 1:MaxImg
        fprintf("idxImg:%d\n",idxImg);

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        classe = str2double(imName(20));
        NumImg = str2double(imName(22:23));
        
        A0 = im2double(imread(imName));

        A = im2double(rgb2gray(imread(imName)));

%         A = medfilt2(filter2(fspecial("average",3),A));

%         showplot = false;

        if showplot
            figure;
            title("A0 RGB")
            imshow(A0)
        end

        %% Reference subimages
        %%% SUBSTITUIR POR SO PARAMETROS
        [regionsRef,regionsRGBRef,bigRefArea] = getRefImages(classe);

        regionsGray = regionsRGBRef;

        N = numel(regionsRef);
        
        
        invMRef = zeros(7,N);
        oriRefs = zeros(N,1);
        sizesRefs = zeros(N,2);
        for k=1:N
            invMRef(:,k) = invmoments(regionsRef{k});
            regionsRef{k} = logical(regionsRef{k});
            regionsGray{k} = rgb2gray(regionsRGBRef{k});
            sizesRefs(k,:) = size(regionsRef{k});
        end

        
        if showplot
            SS = ceil(sqrt(N));
            figure;
            title("Regions Ref")
            for k=1:N
                subplot(SS,SS,k)
                imshow(regionsRef{k})
                xlabel(k)
            end
            figure;
            title("Regions RGB Ref")
            for k=1:N
                subplot(SS,SS,k)
                imshow(regionsRGBRef{k})
                xlabel(k)
            end
        end

        nFeats = 13;
        AllFeatsRef = getFeatures(regionsRef,regionsGray,regionsRGBRef,nFeats);


        %% Vars
%         ObjBord = 0; % numero de objs a tocar o bordo (nao para classificar)
        ObjPart = 0; % numero de objs partidos (nao para classificar)
        ObjOK = 0; % numero de objs para classificar (migalhas nao contam (5% do obj inicial))
        
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

        %% SubImages
        
%         A = F;
        minSize = 0.1; % 0.2  min nnz for aceptable boundary (percentage)
        minWidth = 0.01; % 0.04 min width of subimage (percentage)

        
%         minAreaMigalha = 0.05 * bigRefArea;
        minAreaMigalha = 0.03 * bigRefArea;

        fmaskRot = zeros(size(A));
        cutx = -3; 
        cuty = -3; 
        extend = false; %true;
        relSizes = 5; %3
        minSpare = 0.4; %0.2 da melhor na img4, melhor binarizacao das bolachas vermelhas??

        % Find other subimages
%         try 
        [regions,regionsRGB,~,ObjBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,A0,minAreaMigalha,minSpare);
%         catch
%             fprintf(">>>>>>>>>>>>>>>>fail binarization %d\n",idxImg)
%             continue
%         end

%         return

%         
%         SS = ceil(sqrt(N));

        N = numel(regions);
        
%         count = 1;
%         for k=1:N
%             rBin = regions{k} > 0;
%             if mean(rBin) < 0.9  % Detetar Partidas <<< MUDAR
% %                 str{k} = sprintf("partida,k:%d\n",k);
% %                 fprintf(str{k})
% %                 ObjPart = ObjPart + 1;
%             else
% %                 str{k} = sprintf("OK,k:%d\n",k);
% %                 ObjOK = ObjOK + 1;
%                 regionsOK{count} = regions{k};
%                 regionsOKRGB{count} = regionsRGB{k};
%                 count = count + 1;
%             end
%         end

%         invM = zeros(7,N);
   

        if showplot
            SS = ceil(sqrt(N));
            figure;
            title("Regions Bin")
            for k=1:N
                subplot(SS, SS, k);
                imshow(regions{k})
                xlabel(k)
            end
            figure;
            title("Regions RGB")
            for k=1:N
                subplot(SS, SS, k);
                imshow(regionsRGB{k})
                xlabel(k)
            end
        end

        %% Compare

        
        matchs = zeros(k,1);
        resx = zeros(k,1);
        partidas = [];

        fanKs = [];
        if classe==1
            fanKs = [9 10];
        elseif classe==2
            fanKs = 5;
        end

        for k=1:N
%             invM(:,k) = invmoments(regions{k});
%             [kRef,res] = getBestMatch(invM(:,k),invMRef);
%             [kRef,res,part] = getBestMatchFull(regionsRGB{k},regionsRGBRef);
            
            [kRef,res,part,str] = getBestMatchv2(regionsRGB{k}, AllFeatsRef, oriRefs, sizesRefs, nFeats, fanKs);
            
            if  showplot %&& kRef == 19
                figure;
                subplot(1,2,1)
                imshow(regionsRGB{k})
                if part
                    xlabel(sprintf("part%d",k))
                else
                    xlabel(k)
                end
    
                subplot(1,2,2)
                imshow(regionsRGBRef{kRef})
                xlabel(sprintf("Kref:%d \n %s",kRef,str))

                pause(0.001)
            end

            matchs(k) = kRef;
            resx(k) = res;

            if part
%                 fprintf("Partida:%d\n",k)
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
        
       

        %% Show vars

%         if showplot
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
%         end
       
        
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

function B = maskComplex(A0,minAreaMigalha)
    global showplot;
%     A = rgb2gray(A0);

    minAreaMigalha = round(minAreaMigalha);

%     tol = 0.2;
    rgbImg = A0;
    [idx,map] = rgb2ind(rgbImg, 0.03, 'nodither'); %// consider changing tolerance here
    m = mode( idx );
    frequentRGB = mode(map(m, : ));
    [~,freqChanel] = max(frequentRGB);
    
    Abin = A0;
    for i = 1:3
        Abin(:,:,i) = autobin(Abin(:,:,i));
        Abin(:,:,i) = bwmorph(Abin(:,:,i),"close",inf);
        Abin(:,:,i) = imfill(Abin(:,:,i),"holes");
        Abin(:,:,i) = bwareaopen(Abin(:,:,i),100);
    end
    
    Abin(:,:,freqChanel) = 0;
    
    B = sum(Abin,3)>0;
    B = bwmorph(B,"close",inf);
    B = bwmorph(B,"bridge",inf);
    B = imfill(B,"holes");
    B = bwareaopen(B,minAreaMigalha); % 1000
   
    saveB=B;

    if showplot
        figure;
        imshow(B)
        title("maskComplex 1")
    end
    
    %% clean most common
%     RGB = B.*A0;
%     
%     tol = 0.1;
    
%     A2R = RGB(:,:,1);
%     A2R(abs(A2R-frequentRGB(1))<tol) = 0;
%     A2G = RGB(:,:,2);
%     A2G(abs(A2G-frequentRGB(2))<tol) = 0;
%     A2B = RGB(:,:,3);
%     A2B(abs(A2B-frequentRGB(3))<tol) = 0;
%     
%     A2 = cat(3,A2R,A2G,A2B);
    A2 = A0;
    %%
    
    Abin = ~saveB.*A2;
    A2R = Abin(:,:,1);
    A2R(B) = frequentRGB(1);
    A2G = Abin(:,:,2);
    A2G(B) = frequentRGB(2);
    A2B = Abin(:,:,3);
    A2B(B) = frequentRGB(3);
    Abin = cat(3,A2R,A2G,A2B);
    
    for i = 1:3
        Abin(:,:,i) = autobin(Abin(:,:,i));
        Abin(:,:,i) = bwmorph(Abin(:,:,i),"close",inf);
%         Abin(:,:,i) = imfill(Abin(:,:,i),"holes");
        Abin(:,:,i) = bwareaopen(Abin(:,:,i),minAreaMigalha);%2000
    end
    
    Abin(:,:,freqChanel) = 0;
    
    B = sum(Abin,3)>0;
    B = autobinBW(double(B));
    B = bwareaopen(B,minAreaMigalha);%1000
    B = bwmorph(B,"open",inf);
    B = bwareaopen(B,minAreaMigalha);%1000
    
    
    saveB2=B|saveB;

    if showplot
        figure;
        imshow(saveB2)
        title("maskComplex 2")
    end

    %% HSV
    Abin = ~saveB2.*A2;
    A2R = Abin(:,:,1);
    A2R(saveB2) = frequentRGB(1);
    A2G = Abin(:,:,2);
    A2G(saveB2) = frequentRGB(2);
    A2B = Abin(:,:,3);
    A2B(saveB2) = frequentRGB(3);
    Abin = cat(3,A2R,A2G,A2B);
        
    Abin = rgb2hsv(Abin);
           
    B = autobin(Abin(:,:,1));
    B = bwareaopen(B,minAreaMigalha);%1000
    B = bwmorph(B,"open",inf);
    B = bwareaopen(B,minAreaMigalha);%1000
    B = bwmorph(B,"bridge",inf);
    B = imfill(B,"holes");
   
    B = (B|saveB2);

    if showplot
        figure;
        imshow(B)
        title("maskComplex 3")
    end
end

function [regions,regionsRGB,fullMask,countBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef,minAreaMigalha,minSparse)
    % get all subimages(regions)

%     B = A;
%     T = adaptthresh(B);
% 
%     E = imbinarize(A,T);
% 
%     if mean(E,'all') > 0.3
%         E = not(E);
%     end

    global showplot;

    
    E = maskComplex(imgRef,minAreaMigalha);
%     E = E(2:end-1,2:end-1);

%     E = bwareaopen(E,10);
    E = bwmorph(E,"bridge",inf);
    E = bwmorph(E,"close",inf);
    E = imfill(E,"holes");
    E = bwareaopen(E,100);

    if showplot
        figure;
        imshow(E)
        title("Resultado MaskComplex")
    end

%     F = imclearborder(E);
    F = imclearborder(E(2:end-1,2:end-1));
    F = padarray(F,[1 1],0,"both");

%     F = bwareaopen(F,100);
%     F = bwmorph(F,"close",inf);
%     F = imfill(F,"holes");

    

    %% Bolachas Normais e Partidas
    B = F;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

    if showplot
        figure;
        imshow(B)
        title("Bolachas sem border")
        hold on
    end
    
%     for k = Nb+1:length(Bx) % use only interior boundaries
    for k = 1:Nb % use only exterior boundaries
        boundary = Bx{k};
    
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
        if (nnz(mask) < minSize*sx), continue, end

        % remove already found
        if nnz(mask.*fmaskPrev), continue, end
    
        % clean all zeros cols and rows
        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);
        if (mean(mask0s,'all') < minSparse), continue, end % very sparse image % 0.4 ou 0.2??
    
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

        mask = bwmorph(mask,"dilate");
        mask = bwmorph(mask,"close",inf);
        mask = imfill(mask,"holes");

        if showplot
            plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
            pause(0.001)
        end
        
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
%             figure(300)
%             imshow(selectedRGB)
            fprintf("migalha\n")
%             pause(2)
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


    %% Borders


    G = imadjust(E.*not(F));

    G = bwareaopen(G,100);
    G = bwmorph(G,"close",inf);
    G = imfill(G,"holes");

    B = G;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);

    if showplot
        figure;
        title("Bolachas no border")
        imshow(B)
        hold on
    end

    countBord = 0;
    
%     for k = Nb+1:length(Bx) % use only interior boundaries
    for k = 1:Nb % use only exterior boundaries
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
    
        if showplot
            plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
            pause(0.001)
        end
      
        fmaskPrev = fmaskPrev | mask;

        if (nnz(mask) < minAreaMigalha)
            fprintf("migalha border\n")
            continue
        end

        countBord = countBord + 1;
    
    end

end

function features = getFeatures(regions,regionsGray,regionsRGB,nFeats)
    N = numel(regions);
    features = zeros(nFeats,N);
    for k=1:N
        A = regionsGray{k};
        Argb = regionsRGB{k};
        Abin = regions{k};
        features(:,k) = getFeats(Argb,A,Abin,nFeats);
    end
    
end

function [kRef,minres,part,str] = getBestMatchv2(img1, AllFeatsRef, oriRefs, sizesRefs, nFeats, fanKs)

    global showplot;

    part = false;
    solRefs = AllFeatsRef(end,:);

    Nref = length(oriRefs);
    partidaMean = zeros(Nref,1);
    partidaDiffY = zeros(Nref,1);
    dist = zeros(Nref,1);
    
    tolPartidasMean = 0.7;
    tolPartidasMinVal = 3.5e-1;% 3e-1;
    tolPartidasDiffY = 0.095; %0.1 falha 1 ou 2x no img3

    B = rgb2gray(img1);
    Brgb = img1;
    Bbin = B;
    Bbin = bwareaopen(Bbin,10);

    oriB = regionprops(Bbin,'Orientation').Orientation;
    eulerN = 1;
    for iRef=1:Nref
        oriRef = oriRefs(iRef);
        sxRef = sizesRefs(iRef,1);
        syRef = sizesRefs(iRef,2);

        Brgb2 = imrotate(Brgb,oriRef-oriB);
        Bbin2 = imrotate(Bbin,oriRef-oriB);
        B2 = imrotate(B,oriRef-oriB);

        Bbin2 = bwareaopen(Bbin2,sxRef);
        Bbin2 = Bbin2(:,any(Bbin2,1));
        Bbin2 = Bbin2(any(Bbin2,2),:);
        
        Brgb2 = Brgb2(:,any(B2,1),:);
        Brgb2 = Brgb2(any(B2,2),:,:);

        B2 = B2(:,any(B2,1));
        B2 = B2(any(B2,2),:);

        Brgb2 = imresize(Brgb2,[sxRef NaN]);
        Bbin2 = imresize(Bbin2,[sxRef NaN]);
        B2 = imresize(B2,[sxRef NaN]);

        Bbin2 = bwmorph(Bbin2,"close",3);
%         Bbin2 = bwmorph(Bbin2,"dilate",1);
%         Bbin2 = bwmorph(Bbin2,"fill",10);
        Bbin2 = bwareaopen(Bbin2, sxRef);
        

        [~,Nb] = bwlabel(Bbin2);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb2 = Brgb;
            Bbin2 = Bbin;
            B2 = B;
        end

        featsIm = getFeats(Brgb2,B2,Bbin2,nFeats);
        dists(1) = norm(featsIm - AllFeatsRef(:,iRef));

        Brgb3 = imrotate(Brgb2,180);
        Bbin3 = imrotate(Bbin2,180);
        B3 = imrotate(B2,180);

        [~,Nb] = bwlabel(Bbin3);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb3 = Brgb;
            Bbin3 = Bbin;
            B3 = B;
        end

        featsIm2 = getFeats(Brgb3,B3,Bbin3,nFeats);
        dists(2) = norm(featsIm2 - AllFeatsRef(:,iRef));
        dist(iRef) = min(dists);
        
        if dists(2)<dists(1)
            partidaMean(iRef) =  mean(Bbin3,'all')/solRefs(iRef);
            partidaDiffY(iRef) = size(Bbin3,2)/syRef;
            minres = dists(2);
            eulerN = regionprops(Bbin3,'EulerNumber').EulerNumber;
            
            if iRef == 8 && size(Bbin3,2)/sxRef < partidaDiffY(iRef)
                partidaDiffY(iRef) = size(Bbin3,2)/sxRef;
            end

        else
            partidaMean(iRef) =  mean(Bbin2,'all')/solRefs(iRef);
            partidaDiffY(iRef) = size(Bbin2,2)/syRef;
            minres = dists(1);
            eulerN = regionprops(Bbin2,'EulerNumber').EulerNumber;

            if iRef == 8 && size(Bbin2,2)/sxRef < partidaDiffY(iRef)
                partidaDiffY(iRef) = size(Bbin2,2)/sxRef;
            end
        end
    end

    [minVal,kRef] = min(dist);

    if ismember(kRef,fanKs)
        tolPartidasMean = 0.5;
    end

    partidaDiffY = abs(1-partidaDiffY);
    
%     str = "fine";
    if partidaMean(kRef)<tolPartidasMean || minVal > tolPartidasMinVal || partidaDiffY(kRef) > tolPartidasDiffY 
%         str = sprintf("Partida\n meanRel=%.2f\n minVal=%d\n DiffY:%d",partidaMean(kRef),minVal,partidaDiffY(kRef));
        part = true;
    end
    if kRef == 19 && eulerN~=0
        part = true;
    end
    str = sprintf("meanRel=%.2f\n minVal=%d\n DiffY:%d",partidaMean(kRef),minVal,partidaDiffY(kRef));

    if false %showplot && kRef==8
        iRef = kRef;

        oriRef = oriRefs(iRef);
        sxRef = sizesRefs(iRef,1);

        Brgb2 = imrotate(Brgb,oriRef-oriB);
        Bbin2 = imrotate(Bbin,oriRef-oriB);
        B2 = imrotate(B,oriRef-oriB);

        Bbin2 = bwareaopen(Bbin2,sxRef);
        Bbin2 = Bbin2(:,any(Bbin2,1));
        Bbin2 = Bbin2(any(Bbin2,2),:);
        
        Brgb2 = Brgb2(:,any(B2,1),:);
        Brgb2 = Brgb2(any(B2,2),:,:);

        B2 = B2(:,any(B2,1));
        B2 = B2(any(B2,2),:);

        Brgb2 = imresize(Brgb2,[sxRef NaN]);
        Bbin2 = imresize(Bbin2,[sxRef NaN]);
        B2 = imresize(B2,[sxRef NaN]);

        Bbin2 = bwmorph(Bbin2,"close",3);
%         Bbin2 = bwmorph(Bbin2,"dilate",1);
%         Bbin2 = bwmorph(Bbin2,"fill",10);
        Bbin2 = bwareaopen(Bbin2, sxRef);
        

        [~,Nb] = bwlabel(Bbin2);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb2 = Brgb;
            Bbin2 = Bbin;
            B2 = B;
        end

        featsIm = getFeats(Brgb2,B2,Bbin2,nFeats);
        dists(1) = norm(featsIm - AllFeatsRef(:,iRef));

        Brgb3 = imrotate(Brgb2,180);
        Bbin3 = imrotate(Bbin2,180);
        B3 = imrotate(B2,180);

        [~,Nb] = bwlabel(Bbin3);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb3 = Brgb;
            Bbin3 = Bbin;
            B3 = B;
        end
        
        figure;
        sgtitle("Brgb")
        subplot(1,3,1)
        imshow(Brgb)
        subplot(1,3,2)
        imshow(Brgb2)
        subplot(1,3,3)
        imshow(Brgb3)

        figure;
        sgtitle("B")
        subplot(1,3,1)
        imshow(B)
        subplot(1,3,2)
        imshow(B2)
        subplot(1,3,3)
        imshow(B3)

        figure;
        sgtitle("Bbin")
        subplot(1,3,1)
        imshow(Bbin)
        subplot(1,3,2)
        imshow(Bbin2)
        subplot(1,3,3)
        imshow(Bbin3)
    end
    
    
end

function feats = getFeats(ARGB,Agray,Abin,nFeats)
    s = regionprops(Abin,'Eccentricity','Solidity');
    meanR = mean(ARGB(:,:,1),'all');
    meanG = mean(ARGB(:,:,2),'all');
    meanB = mean(ARGB(:,:,3),'all');
    Ahsv = rgb2hsv(ARGB);
    meanV = mean(Ahsv(:,:,3),'all');
    ola = -real(log(invmoments(Agray)))/20;
    feats = [meanR meanG meanB meanV ola s.Eccentricity s.Solidity]';
end

function Ibin = autobin(I)
%     Ibin = double(imbinarize(I));

%     T = adaptthresh(I,0.2,'ForegroundPolarity','dark');
    [counts,x] = imhist(I,16);
    T = otsuthresh(counts);
    Ibin = double(imbinarize(I,T));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end

function Ibin = autobinBW(I)
    Ibin = double(imbinarize(I));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end
