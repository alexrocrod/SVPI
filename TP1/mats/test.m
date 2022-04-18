close all
clear all

addpath('../sequencias/Seq160')
listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

% addpath('../sequencias/Seq530')
% listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');


idxImg = 40;
imName = listaF(idxImg).name;


A = im2double(imread(imName));

tic
regionsSor=vs_getsubimages(A);
toc

figure(1)
imshow(A)

minSize = 60; % 50
relSizes = 4; % 4
minWidth = 30; % 30

cutx = -3; % -3
cuty = -3; % -3
reductRoted = 2; % 2
rot = true;
extend = false;

tic
% [regionsRotated,masksRotated,fmaskRot] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,zeros(size(A)),reductRoted);
[regionsRotated,fmaskRot] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,zeros(size(A)),reductRoted);
toc


cutx = -1; % -1
cuty = -1; % -1
extend = true;
rot = false;
tic
% [regionsNormal,masksNormal,fmaskNorm] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,reductRoted);
[regionsNormal,fmaskNorm] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,reductRoted);
toc

fprintf("Roted:%d,Normal:%d,Sum:%d,Exact:%d\n",numel(regionsRotated),numel(regionsNormal),numel(regionsNormal)+numel(regionsRotated),numel(regionsSor))


N=numel(regionsNormal);
SS=ceil(sqrt(N));
figure(2)
for k=1:N 
    subplot( SS, SS, k);
    imshow(regionsNormal{k})
    xlabel(k)
end

N=numel(regionsRotated);
SS=ceil(sqrt(N));
figure(3)
for k=1:N 
    subplot( SS, SS, k);
    imshow(regionsRotated{k})
    xlabel(k)
end

figure(4)
imshow(fmaskNorm)

figure(5)
imshow(fmaskRot)


N=numel(regionsSor);
SS=ceil(sqrt(N));
figure(10)
for k=1:N 
    subplot( SS, SS, k);
    imshow(regionsSor{k})
    xlabel(k)
end


function B2 = maskRotated(B)
    B2 = edge(B,'sobel','horizontal');
    B2 = bwmorph(B2,'bridge');
    B2 = bwareaopen(B2,round(0.2*size(B2,1)));
end


function B2 = maskNormal(B)
    B = edge(B,'roberts');
%     B = bwareaopen(B,round(0.2*size(B,1)));
    B = bwmorph(B,'bridge');
    B2 = bwareaopen(B,round(0.2*size(B,1)));
end


% function [regions,masks,fullMask] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,reductRoted)
function [regions,fullMask] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,reductRoted)
    if rot
        B = maskRotated(A);
    else
        B = maskNormal(A);
    end

    fullMask = zeros(size(B));
    
    [Bx,L,Nb] = bwboundaries(B);

    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;
    for k=Nb+1:length(Bx)
        boundary = Bx{k};
        C = (L==k);
        if (nnz(C) < minSize), continue, end
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);

        if nnz(mask.*fmaskPrev) || nnz(mask.*fullMask)
            fprintf("exclui\n")
            continue
        end

        
        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);

        % remove weird shapes
        temp = mask0s;
        sizesT = sort(size(temp));
        if sizesT(1) > relSizes * sizesT(2) || sizesT(1) < minWidth 
            continue
        end

        % estender a quadrilateros
%         if mean(mask0s,"all")>0.99
%         if mean(autobin(mask.*A),"all")>0.7
%             disp(k)
        if extend
            idxs = find(max(mask,[],2));
            minx = max(idxs(1)+cutx,1);
            maxx = min(idxs(end)-cutx,sx);
            idxs = find(max(mask));
            miny = max(idxs(1)+cuty,1);
            maxy = min(idxs(end)-cuty,sy);
            mask = zeros(size(A));
            mask(minx:maxx,miny:maxy) = 1;

            selected = A.*mask;

        elseif rot
            selected = A.*mask;
            selected = selected(:,any(selected,1));
            selected = selected(any(selected,2),:);

            selected = rotateDice(selected,reductRoted);
        else
            selected = A.*mask;
        end

%         masks{count} = mask;
        fullMask = fullMask | mask;
        
        % guardar regiao
        selected = selected(:,any(selected,1));
        regions{count} = selected(any(selected,2),:);
        count = count + 1;
        
    end


%     figure(2)
%     imshow(B)
%     hold on
%     for k=1:length(Bx)
%         boundary = Bx{k};
%         C = (L==k);
%         if (nnz(C) < minSize), continue, end
%         if(k > Nb)
% %             plot(boundary(:,2), boundary(:,1), 'g','LineWidth',2);
%             mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
%     
%             mask0s = mask(:,any(mask,1));
%             mask0s = mask0s(any(mask0s,2),:);
%     
%             % remove weird shapes
%             temp = mask0s;
%             sizesT = sort(size(temp));
% %             fprintf("k=%d,s1=%d,s2=%d\n",count+1,sizesT(1),sizesT(2))
%             if sizesT(1) > relSizes * sizesT(2) || sizesT(1) < minWidth   % 4, 30
%                 continue
%             end
%     
%             % estender a quadrilateros
%     %         if mean(mask0s,"all")>0.99
%     %         if mean(autobin(mask.*A),"all")>0.7
%     %             disp(k)
%                 idxsx = find(max(mask));
%                 minx = idxsx(1);
%                 maxx = idxsx(end);
%                 idxsy = find(max(mask,[],2));
%                 miny = idxsy(1);
%                 maxy = idxsy(end);
%                 mask = zeros(size(A));
%                 mask(miny:maxy,minx:maxx) = 1;
%     %         end
%     
%             masks{count} = mask;
%             
%             % guardar regiao
%             selected = A.*mask;
%             selected = selected(:,any(selected,1));
%             selected = selected(any(selected,2),:);
%             regions{count} = selected(cutx+1:end-cutx,cuty+1:end-cuty);
%             count = count + 1;
%             
%     
%         else
% %             plot(boundary(:,2), boundary(:,1), 'r','LineWidth',2);  
%         end
%            pause(0.01)
%     end
end

function B = rotateDice(unaltered, reductRoted)

    % rodar
    A = imrotate(unaltered,45);

    % reduzir imagem ao dado
    x = size(unaltered,1);
    xmeio = round(size(A,1)/2);

    l = floor(x/sqrt(2));
    deltal = round(l/2)-reductRoted; % 6

    B = double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal));
%     B = autobin(imadjust(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal))));

end




 

