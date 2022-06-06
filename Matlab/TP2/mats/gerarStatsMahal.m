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
    regionsGray{k} = rgb2gray(regionsRGBRef{k});
    xlabel(k)
end

nFeats = 12; % 3, 10, 14

% AllFeatures = getFeatures(regionsRef,regionsRGBRef,nFeats);
AllFeatures = getFeatures(regionsRef,regionsGray,regionsRGBRef,nFeats);

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

[regions,regionsRGB,~,ObjBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,A0,minAreaMigalha);

N=numel(regions);

dist = zeros(N,Nref);
for k=1:N
    B = rgb2gray(regionsRGB{k});
    Brgb = regionsRGB{k};
    Bbin = B;
    Bbin = bwareaopen(Bbin,10);
    for iRef=1:Nref
%     for iRef=1:2:Nref-1
%         size(getFeats(B,Bbin,nFeats,1,0))
%         size(AllFeatures{iRef})
        dist(k,iRef) = mahal(getFeats(Brgb,B,Bbin,nFeats,1,0)',AllFeatures{iRef}');
%         ola = getFeatures({B},{Brgb},nFeats);
%         dist(k,iRef) = mean(mahal(ola{1}',AllFeatures{iRef}'));
%         dist(k,iRef) = min(mahal(ola{1}',AllFeatures{iRef}'));
%         dist(k,iRef+1) = dist(k,iRef);
%         pause(0.1)
    end
end
%%
% ola = getFeatures({B},{Brgb},nFeats);
% size(ola)
% size(ola{1}')
% size(AllFeatures{iRef}')


%%
[minVal,minIdx] = min(dist,[],2);

thPartida = 1e4;

%%
for k=1:N
    figure;
    sgtitle(sprintf("Bolacha k=%d",k))
    subplot(1,2,1)
    imshow(regionsRGB{k})
    if minVal(k) > thPartida
        title("Partida")
    end
    subplot(1,2,2)
    imshow(regionsRGBRef{minIdx(k)})
    xlabel(sprintf("Corresponde a %d \n minVal:%d",minIdx(k),minVal(k)))
    
    pause(0.1)
end
%%


%%
function AllFeatures = getFeatures(regions,regionsGray,regionsRGB,nFeats)
    N = numel(regions);
    features = zeros(nFeats,24);
%     features = zeros(nFeats,2);
    for k=1:N
        A = regionsGray{k};
        Argb = regionsRGB{k};
        Abin = regions{k};
        idx = 1;
        for rot=0:45:315
            for sc=0.5:0.5:1.5
                A2 = imresize(imrotate(Abin,rot),sc);
                A2 = bwareaopen(A2,10);
                Agray2 = imresize(imrotate(A,rot),sc);
                Argb2 = imresize(imrotate(Argb,rot),sc);
                features(:,idx) = getFeats(Argb2,Agray2,A2,nFeats,sc,rot);
                idx = idx + 1;
            end
        end
        AllFeatures{k} = features;
    end
    
end

function feats = getFeats(ARGB,Agray,Abin,nFeats,sc,angle)
    if nFeats == 14
        s = regionprops(Abin,'basic');
        feats = [invmoments(Agray) s.Area s.Centroid s.BoundingBox]';
%     elseif nFeats== 7
%         s = regionprops(Abin,'All');
%         feats = [s.Area/sc/sc s.MajorAxisLength/sc s.MinorAxisLength./sc s.Eccentricity mod(mod(s.Orientation,180)+angle,90)  s.Circularity s.EquivDiameter/sc  s.MaxFeretDiameter/sc s.MinFeretDiameter/sc]';
%     elseif nFeats == 10
%         s = regionprops(Abin,'Circularity','Eccentricity','EulerNumber');
%         feats = [invmoments(Agray) s.Circularity s.Eccentricity s.EulerNumber]';
    elseif nFeats == 10
        s = regionprops(Abin,'Circularity','Eccentricity','Solidity');
        ola = real(log(invmoments(Agray)));
%         ola = abs(ola)/max(abs(ola));
        feats = [ola s.Circularity s.Eccentricity s.Solidity]';
    elseif nFeats == 12
        s = regionprops(Abin,'Circularity','Eccentricity','Solidity');
        R = ARGB(:,:,1);
        meanR = mean(R(Abin),'all');
        G = ARGB(:,:,2);
        meanG = mean(G(Abin),'all');
        B = ARGB(:,:,3);
        meanB = mean(B(Abin),'all');
%         R = ARGB(:,:,1);
%         meanR = mean(R,'all');
%         G = ARGB(:,:,2);
%         meanG = mean(G,'all');
%         B = ARGB(:,:,3);
%         meanB = mean(B,'all');
        ola = real(log(invmoments(Agray)));
        feats = [meanR meanG meanB ola s.Eccentricity s.Solidity]';
    elseif false %nFeats == 12
        s = regionprops(Abin,'All');
        areaFig = size(Abin,1)*size(Abin,2)/1e3;
        maxS = max(size(Abin));
        minS = min(size(Abin));
        R = ARGB(:,:,1);
        meanR = mean(R(Abin),'all');
        G = ARGB(:,:,2);
        meanG = mean(G(Abin),'all');
        B = ARGB(:,:,3);
        meanB = mean(B(Abin),'all');
%         feats = [meanR meanG meanB s.Solidity s.Area./areaFig./1000 s.MajorAxisLength./areaFig s.MinorAxisLength./areaFig s.Eccentricity s.Circularity s.EquivDiameter/areaFig s.MaxFeretDiameter/areaFig s.MinFeretDiameter/areaFig]';
        feats = [meanR meanG meanB s.Solidity s.Area./areaFig./1000 s.MajorAxisLength./maxS s.MinorAxisLength./minS s.Eccentricity s.Circularity s.EquivDiameter/maxS s.MaxFeretDiameter/maxS s.MinFeretDiameter/minS]';
%     elseif nFeats == 10
%         s = regionprops(Abin,'All');
%         areaFig = size(Abin,1)*size(Abin,2)/1e3;
%         meanGray = mean(Agray(Abin),'all');
%         feats = [meanGray s.Solidity s.Area/areaFig/1000 s.MajorAxisLength/areaFig s.MinorAxisLength./areaFig s.Eccentricity s.Circularity s.EquivDiameter/areaFig s.MaxFeretDiameter/areaFig s.MinFeretDiameter/areaFig]';
    elseif nFeats == 2
        s = regionprops(Abin,'Circularity','Eccentricity');
        feats = [s.Circularity s.Eccentricity]';
    else
        s = regionprops(Abin,'All');
%         MeanFA = (s.MaxFeretAngle-s.MinFeretAngle)/sc??;
%         feats = [s.Area/sc/sc s.MajorAxisLength/sc s.MinorAxisLength./sc s.Eccentricity mod(mod(s.Orientation,180)+angle,90)  s.Circularity s.EquivDiameter/sc  s.MaxFeretDiameter/sc s.MinFeretDiameter/sc]'; %s.Perimeter/sc
        areaFig = size(Abin,1)*size(Abin,2)/1e3;
%         maxS = max(size(Abin));
%         minS = min(size(Abin));
%         feats = [s.Area/areaFig s.MajorAxisLength/areaFig s.MinorAxisLength./areaFig s.Eccentricity mod(mod(s.Orientation,180)+angle,90)/90  s.Circularity s.EquivDiameter/areaFig s.MaxFeretDiameter/areaFig s.MinFeretDiameter/areaFig]'; %s.Perimeter/sc       
%         feats = [s.Area/areaFig/1000 s.MajorAxisLength/areaFig s.MinorAxisLength./areaFig s.Eccentricity s.Circularity s.EquivDiameter/areaFig s.MaxFeretDiameter/areaFig s.MinFeretDiameter/areaFig]'; %s.Perimeter/sc       
        feats = [s.Solidity s.Area/areaFig/1000 s.MajorAxisLength/areaFig s.MinorAxisLength./areaFig s.Eccentricity s.Circularity s.EquivDiameter/areaFig s.MaxFeretDiameter/areaFig s.MinFeretDiameter/areaFig]'; %s.Perimeter/sc 
%         feats = [s.Solidity s.Area/areaFig s.MajorAxisLength/maxS s.MinorAxisLength./minS s.Eccentricity s.Circularity s.EquivDiameter/maxS s.MaxFeretDiameter/maxS s.MinFeretDiameter/minS]'; %s.Perimeter/sc 
        %algumas coisas nao escalam com areaFig so...
    end
end


function [regions,regionsRGB,fullMask,countBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef,minAreaMigalha)

    % get all subimages(regions)

%     B = maskNormal(A);
%     
%     B = bwareaopen(B,round(minSize*size(B,1)));
%         B = medfilt2(filter2(fspecial("average",3),A));

    B = A;
    T = adaptthresh(B);

    E = imbinarize(A,T);

%     mean(E,'all');


%     Eold = E;

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