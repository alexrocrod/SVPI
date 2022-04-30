close all
clear all
clc

imgRef1 = im2double(imread("../svpi2022_TP2_img_001_01.png"));
imgRef2 = im2double(imread("../svpi2022_TP2_img_002_01.png"));


figure(1)
imshow(imgRef1)

A = rgb2gray(imgRef1);
minSize = 0.1;
cutx = -1;
cuty = -1;
relSizes = 3;
minWidth = 0.05;
extend = false;
fmaskPrev = zeros(size(A));

[regions,regionsRGB,fullMask] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef1);

N = numel(regions);
SS = ceil(sqrt(N));

figure(2)
for k=1:N
    subplot(SS,SS,k)
    imshow(regions{k})
    xlabel(k)
end

figure(3)
for k=1:N
    subplot(SS,SS,k)
    imshow(regionsRGB{k})
    xlabel(k)
end




function B = maskNormal(A)
    A = A<1;
    % mask for all other subimages
    
    B = edge(A,'roberts') | edge(A,'sobel');
%     B = bwmorph(B,'bridge',inf);
    B = bwmorph(B,'close',inf);
end


function [regions,regionsRGB,fullMask] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,imgRef)
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
        selectedRGB = imgRef.*repmat(mask,[1 1 3]);

        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
    
        % guardar regiao
        selectedRGB = selectedRGB(:,any(selected,1),:);
        selectedRGB = selectedRGB(any(selected,2),:,:);

        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);
        
        regions{count} = selected;
        regionsRGB{count} = selectedRGB;
        
        count = count + 1;
    
    end
end