close all
clear all

% addpath('../sequencias/Seq160')
% listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

addpath('../sequencias/Seq530')
listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');


idxImg = 13;
imName = listaF(idxImg).name;

A = im2double(imread(imName));

tic
regionsSor = vs_getsubimages(A);
toc

% figure(1)
% imshow(A)

minSize = 0.2; % 60 -> 0.08, 152-> 0.2
minWidth = 0.04; % 30 -> 0.04

cutx = -3; % -3
cuty = -3; % -3
reductRoted = 2; % 2
rot = true;
extend = false;
relSizes = 1.2; % 1.2
tic
[regionsRotated,fmaskRot] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,zeros(size(A)),reductRoted);
toc


cutx = -1; % -1
cuty = -1; % -1
extend = true;
rot = false;
relSizes = 3; % 3
tic
[regionsNormal,fmaskNorm] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,reductRoted);
toc

fprintf("Img:%d,Roted:%d,Normal:%d,Sum:%d,Exact:%d\n",idxImg,numel(regionsRotated),numel(regionsNormal),numel(regionsNormal)+numel(regionsRotated),numel(regionsSor))


% N=numel(regionsNormal);
% SS=ceil(sqrt(N));
% figure(2)
% for k=1:N 
%     subplot(SS,SS,k);
%     imshow(regionsNormal{k})
%     xlabel(k)
% end

N=numel(regionsRotated);
SS=ceil(sqrt(N));
figure(2)
for k=1:N 
    subplot(SS,SS,k);
    imshow(regionsRotated{k})
    xlabel(k)
end

figure(3)
for k=1:N 
    subplot(SS,SS,k);
%     regionsRotated{k} =  medfilt2(filter2(fspecial('average',3),regionsRotated{k}));
% %     regionsRotated{k} =  medfilt2(regionsRotated{k});
% %     regionsRotated{k} = wiener2(regionsRotated{k},[5 5]);
%     regionsRotated{k} = imadjust(regionsRotated{k});
%     regionsRotated{k} = autobin(regionsRotated{k},true);
%     regionsRotated{k} = bwareaopen(regionsRotated{k},30);
% %     regionsRotated{k} = bwareafilt(regionsRotated{k},[30 300]);

%     regionsRotated{k} =  medfilt2(regionsRotated{k});
%     regionsRotated{k} = imadjust(regionsRotated{k});
%     regionsRotated{k} = autobin(regionsRotated{k},false);
%     regionsRotated{k} = bwareaopen(regionsRotated{k},30);



    imshow(regionsRotated{k})
    xlabel(nnz(regionsRotated{k}))
end

figure(20)
for k=1:N 
    subplot(SS,SS,k);
    B = regionsRotated{k};
    B =  medfilt2(B);
    B = imadjust(B);
    B = autobin(B,false);
    B = bwareaopen(B,30);

%     B = bwmorph(B,'bridge',inf);
%     B = edge(B,'roberts');
% %     B = edge(B,'sobel','horizontal');
% %     B = bwmorph(B,'bridge',inf);
% %     B = bwareaopen(B,round(0.5*size(B,1)));

%     B = edge(B,'roberts');
%     B = bwmorph(B,'close');
%     B = bwareaopen(B,round(0.5*size(B,1)));

    B = bwmorph(B,'remove');

    if nnz(medfilt2(B))>10
        disp(k)
        B = regionsRotated{k};
        B =  medfilt2(B);
        B = imadjust(B);
        B = autobin(B,true);

        B =  medfilt2(B);
        B = bwmorph(B,'remove');
        B = bwmorph(B,'close');
        B = bwareaopen(B,round(0.5*size(B,1)));

        [~,Nb] = bwlabel(B);
%         while Nb>6
%             B =  medfilt2(B);
%             [~,Nb] = bwlabel(B);
%         end
%         fprintf("0nnz=%d,100Nb=%d\n",nnz(B),100*Nb);
        while nnz(B)>100*Nb
%             fprintf("nnz=%d,100Nb=%d\n",nnz(B),100*Nb);
            B =  medfilt2(B);
            B = bwmorph(B,'remove');
            B = bwareaopen(B,round(0.5*size(B,1)));
            [~,Nb] = bwlabel(B);
        end
%         B = edge(B,'sobel');
%         B = bwmorph(B,'close');
    end

        
    [~,Nb] = bwlabel(B);
    imshow(B)
    xlabel(sprintf("k=%d,Nb=%d\n nnz/Nb=%.2f",k,Nb,nnz(B)/Nb))
end

% figure(4)
% 
% subplot(1,4,1)
% imshow(fmaskNorm)
% 
% subplot(1,4,2)
% imshow(fmaskRot)
% 
% subplot(1,4,3)
% imshow(fmaskNorm | fmaskRot)
% 
% subplot(1,4,4)
% imshow(fmaskNorm .* fmaskRot)


% N=numel(regionsSor);
% SS=ceil(sqrt(N));
% figure(10)
% for k=1:N 
%     subplot( SS, SS, k);
%     imshow(regionsSor{k})
%     xlabel(k)
% end



function Ibin = autobin(I,v2) % autobin but for 2 thresholds

    if v2
        warning ('off','all');
        [ts,met] = multithresh(I,2);
        warning ('on','all');
    
        if met==0 % invalid 2nd threshold
            Ibin = double(imbinarize(I));
        else
            T = (ts(1)+ts(2))/2;
    %         T = ts(2);
            Ibin = double(imbinarize(I,T));
        end
    else
            Ibin = double(imbinarize(I));
    end
    
    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end

function B2 = maskRotated(B)
%     B2 = edge(B,'sobel','horizontal');
%     B2 = bwmorph(B2,'bridge');

    SE1 = [0 0 1
           0 1 0
           1 0 0];
    SE2 = [1 0 0
           0 1 0
           0 0 1];
    
    B2 = edge(B,'sobel','vertical');
%     B2 = bwmorph(B2,'bridge',inf);
    B2 = imclose(B2,SE1);
    B2 = imclose(B2,SE2);
end


function B = maskNormal(A)
    B = edge(A,'roberts');
    B = bwmorph(B,'bridge');
    
%      SE1 = [0 1 0
%            0 1 0
%            0 1 0];
%     SE2 = [0 0 0
%            1 1 1
%            0 0 0];
% 
%     B2 = edge(B,'roberts');
%     B2 = imclose(B2,SE1);
%     B2 = imclose(B2,SE2);
%     B2 = bwareaopen(B2,round(minSize*size(B2,1)));
end


% function [regions,masks,fullMask] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,reductRoted)
function [regions,fullMask] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,reductRoted)
    if rot
        B = maskRotated(A);
    else
        B = maskNormal(A);
    end
    
    B = bwareaopen(B,round(minSize*size(B,1)));

    fullMask = zeros(size(B));
    
    [Bx,L,Nb] = bwboundaries(B);

    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;


%     figure(20)
%     imshow(B)
%     hold on

    for k=Nb+1:length(Bx)
        boundary = Bx{k};
        
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
        if (nnz(mask) < minSize*sx), continue, end
%         fprintf("pass1\n")

        if nnz(mask.*fmaskPrev)
            continue
        end
%         fprintf("pass2\n")
        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);

        % remove weird shapes
        sizesT = sort(size(mask0s));
        if sizesT(2) > relSizes * sizesT(1) || sizesT(1) < minWidth*sx
            continue
        end
%         fprintf("pass3\n")

         % remove already found
        if nnz(mask.*fmaskPrev)
            continue
        end
%         fprintf("pass4\n")

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
        end

        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);

        % remove weird shapes
        sizesT = sort(size(mask0s));
        if sizesT(2) > relSizes * sizesT(1) || sizesT(1) < minWidth*sx
            continue
        end
%         fprintf("pass5\n")

%         plot(boundary(:,2), boundary(:,1), 'g','LineWidth',2);
% %         text(boundary(1,2), boundary(1,1),num2str(k),'Color','r');
%         pause(0.1)

        selected = A.*mask;

        if rot && ~extend
            selected = selected(:,any(selected,1));
            selected = selected(any(selected,2),:);

            selected = rotateDice(selected,reductRoted);
        end

%         masks{count} = mask;
        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
        
        % guardar regiao
        selected = selected(:,any(selected,1));
        regions{count} = selected(any(selected,2),:);
        count = count + 1;
        
    end
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




 

