close all
clear all

% addpath('../sequencias/Seq160')
% listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

addpath('../sequencias/Seq530')
listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');


idxImg = 1;
imName = listaF(idxImg).name;


A = im2double(imread(imName));

regionsSor=vs_getsubimages(A);
% N=numel(regions);
% SS=ceil(sqrt(N));

figure(1)
imshow(A)


% B = A;
% % B = autobin(A);
% B = edge(B,'roberts');
% % B = bwareaopen(B,20);
% B = bwmorph(B,'close');
% B = bwareaopen(B,20);

B = edging(A);
% [L,Nb] = bwlabel(B);

% [Bx,L,Nb] = bwboundaries(B,'noholes');
[Bx,L,Nb] = bwboundaries(B);

figure(2)
imshow(B)
hold on

sx = size(B,1);
sy = size(B,2);
minSize = 50;

count = 1;
% cutx = 1; % 1
% cuty = 1; % 2
cutx = 0; % 1
cuty = 0; % 2

for k=1:length(Bx)
    boundary = Bx{k};
    C = (L==k);
    if (nnz(C) < minSize), continue, end
    if(k > Nb)
        plot(boundary(:,2), boundary(:,1), 'g','LineWidth',2);
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);

        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);
%         mask0s = mask0s(:,any(mask0s,1));

        % remove weird shapes
        temp = mask0s;
        sizesT = sort(size(temp));
        fprintf("k=%d,s1=%d,s2=%d\n",count+1,sizesT(1),sizesT(2))
        if sizesT(1) > 4 * sizesT(2) || sizesT(1) < 30
            continue
        end

        % estender a quadrilateros
%         if mean(mask0s,"all")>0.99
%         if mean(autobin(mask.*A),"all")>0.7
            disp(k)
            idxsx = find(max(mask));
            minx = idxsx(1);
            maxx = idxsx(end);
            idxsy = find(max(mask,[],2));
            miny = idxsy(1);
            maxy = idxsy(end);
            mask = zeros(size(A));
            mask(miny:maxy,minx:maxx) = 1;
%         end

        masks{count} = mask;
        
        % guardar regiao
        selected = A.*mask;
        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);
%         selected = selected(:,any(selected,1));
        regions{count} = selected(cutx+1:end-cutx,cuty+1:end-cuty);
        count = count + 1;
        

    else
        plot(boundary(:,2), boundary(:,1), 'r','LineWidth',2);  
    end
       pause(0.01)
end


N=numel(regions);
SS=ceil(sqrt(N));

figure(4)
for k=1:N 
    subplot( SS, SS, k);
    imshow(regions{k})
    xlabel(k)
end

figure(5)
imshow(zeros(size(A)))
temp = zeros(size(A));
hold on
for k=2:N 
    temp = temp | masks{k};
    imshow(temp)
%     drawnow
%     pause(0.2)
end

N=numel(regionsSor);
SS=ceil(sqrt(N));

figure(10)
for k=1:N 
    subplot( SS, SS, k);
    imshow(regionsSor{k})
    xlabel(k)
end


% pause(100)

% for x = 1:Nb % select each boundary
%     C = (L==x);
%     if (nnz(C) < minSize), continue, end
%     BB = bwboundaries(C,'noholes');
%     boundary = BB{1};
% 
%     plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
% 
% %         M = poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)); % from the boundary to a mask matrix (region)
% % 
% %         if median(boundary(:,1))>0.4*size(B,1) % invert lower card symbols
% %              M = rot90(rot90(poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)))); 
% %         end
% %         
%     
% %         M = bwmorph(M,"bridge");
% %         M = bwmorph(M,"fill");
% 
%     % remove all zeros rows and cols
% %         clean0s = M(:,any(M,1));
% %         clean0s = clean0s(any(clean0s,2),:);
% 
% %         if nnz(clean0s)<3*size(clean0s,1)
% % %             subplot(1,2,2);
% % %             imshow(clean0s)
% % %             xlabel("discard")
% % %             pause(1)
% %             continue
% %         end
% %         count = count + 1;
% 
% 
% %         subplot(1,2,2);
% %         imshow(clean0s)
%     
% %     meanO = meanO + mean(imresize(clean0s,scNaipe)~=imresize(ouro,scNaipe*size(clean0s)),'all');
% %     meanC = meanC + mean(imresize(clean0s,scNaipe)~=imresize(copa,scNaipe*size(clean0s)),'all');
% %     meanE = meanE + mean(imresize(clean0s,scNaipe)~=imresize(espada,scNaipe*size(clean0s)),'all');
% 
%         pause(0.1)
% 
% end

function B = edging(A)
    B = A;

%     B = edge(B,'roberts');
%     B = bwareaopen(B,round(0.1*size(B,1)));
%     B = bwmorph(B,'bridge');

%     B = edge(B,'sobel');
%     B = bwmorph(B,'bridge');
%     B = bwareaopen(B,round(0.2*size(B,1)));

% %     B = edge(B,'sobel');
%     B = bwareaopen(B,round(0.15*size(B,1)));
%     B = bwmorph(B,'close');
%     B = bwmorph(B,'bridge');
% %     B = bwmorph(B,'remove'); % remove
%     B = bwareaopen(B,round(0.2*size(B,1)));

%     B = bwmorph(B,'remove'); % remove
%     B = bwareaopen(B,round(0.3*size(B,1)));

    B = edge(B,'roberts');
%     B = bwareaopen(B,round(0.2*size(B,1)));
    B = bwmorph(B,'bridge');
    B = bwareaopen(B,round(0.2*size(B,1)));

    %rodados
    B2 = edge(A,'sobel','horizontal');
%     B = bwareaopen(B,round(0.2*size(B,1)));
    B2 = bwmorph(B2,'bridge');
    B2 = bwareaopen(B2,round(0.2*size(B2,1)));



    % da os rodados???
%     B2 = bwmorph(A,'bridge');
%     B2 = bwmorph(B2,'remove'); % remove
%     B2 = bwareaopen(B2,round(0.5*size(B2,1)));
%     B2 = edge(B2,'sobel');

    B = B | B2;
end

function Ibin = autobin(I) % autobin but for 2 thresholds

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

%     Ibin = double(imbinarize(I));
    
    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end







 

