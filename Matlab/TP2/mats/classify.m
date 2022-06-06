close all
clear all
clc

classe = 1 ;
[regionsRef,regionsRGBRef,bigRefArea] = getRefImages(classe);


N = numel(regionsRef);
Nref=N;
SS = ceil(sqrt(N));

invMRef = zeros(7,N);
for k=1:N
    invMRef(:,k) = invmoments(regionsRef{k});
end

regionsGray = regionsRGBRef;

figure;
for k=1:N
    subplot(SS,SS,k)
    imshow(regionsRef{k})
    regionsRef{k} = logical(regionsRef{k});
    regionsGray{k} = rgb2gray(regionsRGBRef{k});
    xlabel(k)
end

nFeats = 13;

AllFeatures = getFeaturesRot(regionsRef,regionsGray,regionsRGBRef,nFeats);

% return
%%

imName = "../Seq29x/svpi2022_TP2_img_291_01.png";
A0 = im2double(imread(imName));

A = im2double(rgb2gray(imread(imName)));

figure;
imshow(A0)

figure;
imshow(A)

minSize = 0.1; % 0.2  min nnz for aceptable boundary (percentage)
minWidth = 0.01; % 0.04 min width of subimage (percentage)

minAreaMigalha = 0.05 * bigRefArea;

fmaskRot = zeros(size(A));
cutx = -3; 
cuty = -3; 
extend = false; %true;
relSizes = 5; %3
minSpare = 0.4; %0.2 da melhor na img4, melhor binarizacao das bolachas vermelhas??

FundoLims = zeros(9,3,2);

FundoLims(:,:,1)=[  0.112	0.076	0.911
                    0.514	0.268	0.188
                    0.516	0	    0
                    0.614	0.132	0.019
                    0.588	0	    0
                    0.206	0.146	0.519
                    0.194	0	    0
                    0.995	0	    0
                    0.040	0	    0];


FundoLims(:,:,2)=[  0.185	0.163	1
                    0.602	1	    1
                    0.569	1	    1
                    0.704	1	    0.493
                    0.929	1	    1
                    0.274	1	    1
                    1	    1	    0.181
                    0.008	0.014	0.190
                    0.185	1   	0.241];

minSizesFundos = [100 10 100 10 100 10 1000 10 20]; 

minAcceptFundo = 0.2;
maxAcceptFundo = 0.4;

% [regions,regionsRGB,~,ObjBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,A0,minAreaMigalha);
[regions,regionsRGB,~,ObjBord] = getSubImagesV2(A,minSize,relSizes,minWidth,fmaskRot,A0,minAreaMigalha,minSpare,FundoLims,minSizesFundos,minAcceptFundo,maxAcceptFundo);

N = numel(regions);
regionsBin = regions;
for k=1:N
    regionsBin{k} = regions{k}>0;
end

AllFeatsCookies = getFeaturesRot(regionsBin,regions,regionsRGB,nFeats)

% AllFeats = zeros(N,Nref,nFeats);
partidaMean = zeros(N,Nref);
partidaDiffY = zeros(N,Nref);
partsMBbin = [];
dist = zeros(N,Nref);

for k=1:N
    Bbin = regionsBin{k};
%     B = rgb2gray(regionsRGB{k});
%     Brgb = regionsRGB{k};
% %     Bbin = B;
% %     Bbin = regions{k};
% %     Bbin = bwareaopen(Bbin,10);
%     Bbin = B>0;
% %     figure;
% %     subplot(1,4,1)
% %     imshow(Brgb)
    oriB = regionprops(Bbin,'Orientation').Orientation;
    for iRef=1:Nref
%         oriRef = regionprops(regionsRef{iRef},'Orientation').Orientation;
%         sxRef = size(regionsRef{iRef},1);
% %         subplot(1,4,2)
% %         imshow(regionsRGBRef{iRef})
% 
%         Brgb2 = imrotate(Brgb,oriRef-oriB);
        Bbin2 = imrotate(Bbin,-oriB);
%         B2 = imrotate(B,oriRef-oriB);

        Bbin2 = bwareaopen(Bbin2,100);
        Bbin2 = Bbin2(:,any(Bbin2,1));
        Bbin2 = Bbin2(any(Bbin2,2),:);
%         
%         Brgb2 = Brgb2(:,any(B2,1),:);
%         Brgb2 = Brgb2(any(B2,2),:,:);
% 
% %         B2 = B2(:,any(B2,1));
% %         B2 = B2(any(B2,2),:);
% 
%         Brgb2 = imresize(Brgb2,[sxRef NaN]);
% %         Bbin2 = imresize(Bbin2,[sxRef NaN]);
% %         B2 = imresize(B2,[sxRef NaN]);
%         B2 = rgb2gray(Brgb2);
%         Bbin2 = B2>0;
%         Bbin2 = bwareaopen(Bbin2, 100); %sxRef
% 
%         [~,Nb] = bwlabel(Bbin2);
%         if Nb~=1
%             fprintf("fail k%d iRef%d\n",k,iRef)
%             Brgb2 = Brgb;
%             Bbin2 = Bbin;
%             B2 = B;
%         end

%         featsIm = getFeats(Brgb2,B2,Bbin2);
        featsIm = AllFeatsCookies(:,k);
%         dists(1) = norm(featsIm - AllFeatsRef(:,iRef));
        dist(k,iRef) = norm(featsIm - AllFeatures(:,iRef));
        
        partidaMean(k,iRef) =  mean(Bbin2,'all')/mean(regionsRef{iRef},'all');
%         partidaDiffY(k,iRef) = size(Bbin2,2)/size(regionsRef{iRef},2);

        sz1 = sort(size(Bbin2));
        szRa = sz1(1)/sz1(2);

        sz1 = sort(size(regionsRef{iRef}));
        szRef = sz1(1)/sz1(2);
        partidaDiffY(k,iRef) = szRa/szRef;
        
    end
end


%%
[minVal,minIdx] = min(dist,[],2);


%%
tolPartidasMean = 0.7;
tolPartidasMinVal = 3e-1;
tolPartidasDiffY = 0.05;

partidaDiffY = abs(1-partidaDiffY);

for k=1:N
    figure;
    sgtitle(sprintf("Bolacha k=%d",k))
    subplot(1,2,1)
    imshow(regionsRGB{k})
    if partidaMean(k,minIdx(k))<tolPartidasMean || minVal(k) > tolPartidasMinVal || partidaDiffY(k,minIdx(k)) > tolPartidasDiffY 
%     if minVal(k) > tolPartidasMinVal
        fprintf("Partida %d\n",k);
        xlabel(sprintf("Partida\n meanRel=%.2f\n minVal=%d\n DiffY:%d",partidaMean(k,minIdx(k)),minVal(k),partidaDiffY(k,minIdx(k))))
        partsMBbin = [partsMBbin k];
    end
%     xlabel(sprintf("%.3f %.3f %.3f\n %.3f\n %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n %.3f %.3f",AllFeats(k,minIdx(k),:)))
    subplot(1,2,2)
    imshow(regionsRGBRef{minIdx(k)})
    title(sprintf("Corr:%d",minIdx(k)))
    xlabel(sprintf("Diff:%d",minVal(k)))
    
%     xlabel(sprintf("%.3f %.3f %.3f\n %.3f\n %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n %.3f %.3f",AllFeatures(:,minIdx(k))))
%     feats = [meanR meanG meanB meanV ola s.Eccentricity s.Solidity]';
   
    pause(0.1)
end

return


%% 
clc
tolPartidasMean = 0.95;
tolPartidasMinVal = 2.15e-1; %2.12 a 2.18
tolPartidasDiffY = 0.04; 


parts2 = [];
for k=1:N
    fprintf("Partida %d meanRel=%.2f minVal=%d DiffY:%d\n",k,partidaMean(k,minIdx(k)),minVal(k),partidaDiffY(k,minIdx(k)));
    if partidaMean(k,minIdx(k))< tolPartidasMean || minVal(k) > tolPartidasMinVal || partidaDiffY(k,minIdx(k)) > tolPartidasDiffY 
        parts2 = [parts2 k];
    end
end
partsTH = find(minVal>tolPartidasMinVal)'
parts2
partidas = sort([2 5 8 12 19 16 18 24 25 28 38 39 41 45]) 



%%
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

function features = getFeaturesRot(regions,regionsGray,regionsRGB,nFeats)
    
    N = numel(regions);
    features = zeros(nFeats,N);
    for k=1:N
        Brgb = regionsRGB{k};
        B = regionsGray{k};
        Bbin = regions{k};
        ori = regionprops(Bbin,"Orientation").Orientation;

        Brgb2 = imrotate(Brgb,-ori);
        B2 = rgb2gray(Brgb2);
        Bbin2 = B2>0;
        Bbin2 = bwareaopen(Bbin2, 100);         

        [~,Nb] = bwlabel(Bbin2);
        if Nb~=1
            fprintf(">>>>>>>>> fail Nb=%d iref=%d\n",Nb,iRef)
            Brgb2 = Brgb;
            Bbin2 = Bbin;
            B2 = B;
        end
        features(:,k) = getFeats(Brgb2,B2,Bbin2,nFeats);
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

%     s = regionprops(Abin,'Eccentricity','Solidity');
% %     meanR = mean(ARGB(:,:,1),'all');
% %     meanG = mean(ARGB(:,:,2),'all');
% %     meanB = mean(ARGB(:,:,3),'all');
%     Ahsv = rgb2hsv(ARGB);
%     meanH = mean(Ahsv(:,:,1),'all');
%     meanV = mean(Ahsv(:,:,3),'all');
%     ola = -real(log(invmoments(Agray)))/20;
%     feats = [meanH meanV ola s.Eccentricity s.Solidity]';
end

function [B,mask] = removeFundoDado(A,FundoLims,minS)
    HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);

    if FundoLims(:,1,1) > FundoLims(:,1,2) 
        mask = (H >= FundoLims(:,1,1) | H <= FundoLims(:,1,2)) & (S >= FundoLims(:,2,1) & S <= FundoLims(:,2,2)) & (V >= FundoLims(:,3,1) & V <= FundoLims(:,3,2)); %add a condition for value
    else
        mask = (H >= FundoLims(:,1,1) & H <= FundoLims(:,1,2)) & (S >= FundoLims(:,2,1) & S <= FundoLims(:,2,2)) & (V >= FundoLims(:,3,1) & V <= FundoLims(:,3,2)); %add a condition for value
    end

    mask=bwareaopen(mask,minS);

    mask=~mask; %mask for objects (negation of background)

    mask=bwareaopen(mask,minS); %in case we need some cleaning of "small" areas.

    %%%% Sempre??
    mask = bwmorph(mask,"close",inf);
    mask = imfill(mask,"holes");
    %%%%

    B = mask.*A;
end

function [regions,regionsRGB,fullMask,countBord] = getSubImagesV2(A,minSize,relSizes,minWidth,fmaskPrev,imgRef,minAreaMigalha, ...
    minSparse,FundosLims,minSizesFundos,minAcceptFundo,maxAcceptFundo)
    % get all subimages(regions)


    maxAccept = maxAcceptFundo;
    fundoUsed = 0;
    imgRefOld = imgRef;
    maskEnd = ones(size(A));

    for i=1:length(FundosLims)
        [AnoF,mask] = removeFundoDado(imgRefOld,FundosLims(i,:,:),minSizesFundos(i));
        nnzMask = mean(mask,"all");
        fprintf("fundo n%d, mean%.2f \n",i, nnzMask)
        if nnzMask < maxAccept && nnzMask > minAcceptFundo
            maxAccept = nnzMask;
            A = rgb2gray(AnoF);
            imgRef = AnoF;
            maskEnd = mask;
            fprintf("Usado fundo n%d, mean%.2f \n",i, nnzMask)
            fundoUsed = i;
        end
    end

    if ~fundoUsed
%     if ~fundoUsed || fundoUsed==7
        fprintf("Not using a fundo\n")
        E = maskComplex(imgRef,minAreaMigalha);

%         E = bwmorph(E,"bridge",inf);
%         E = bwmorph(E,"close",inf);
%         E = imfill(E,"holes");
%         E = bwareaopen(E,100);
    
    else
        fprintf("Usou fundo n%d, mean%.2f \n",fundoUsed, maxAccept)
%         E = bwareaopen(mask,minAreaMigalha);
        E = maskEnd;
    end

%     F = imclearborder(E);
    F = imclearborder(E(2:end-1,2:end-1));
    F = padarray(F,[1 1],0,"both");


    %% Bolachas Normais e Partidas
    B = F;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

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
    
        mask = bwmorph(mask,"dilate");
        mask = bwmorph(mask,"close",inf);
        mask = imfill(mask,"holes");

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
            fprintf("migalha\n")
            continue
        end
        
        regions{count} = selected;

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
      
        fmaskPrev = fmaskPrev | mask;

        if (nnz(mask) < minAreaMigalha)
            fprintf("migalha border\n")
            continue
        end

        countBord = countBord + 1;
    
    end

end

function [regions,regionsRGB,fullMask,countBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef,minAreaMigalha)
    % get all subimages(regions)

    B = A;
    T = adaptthresh(B);

    E = imbinarize(A,T);

    if mean(E,'all') > 0.3
        E = not(E);
    end

    F = imclearborder(E);

    F = bwareaopen(F,100);
    F = bwmorph(F,"close",inf);
    F = imfill(F,"holes");

    

    %% Bolachas Normais e Partidas
    B = F;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;
% 
%     figure(751)
%     imshow(B)
%     hold on
    
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

%     figure(752)
%     imshow(B)
%     hold on

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
    
%         plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
%         pause(0.01)
      
        fmaskPrev = fmaskPrev | mask;

        if (nnz(mask) < minAreaMigalha)
            fprintf("migalha border\n")
            continue
        end

        countBord = countBord + 1;
    
    end

end

function B = maskNormal(A)
    % mask for all other subimages
    
    B = edge(A,'roberts') | edge(A,'sobel');
    B = bwmorph(B,'close',inf);
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