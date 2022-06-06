% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Trabalho Pratico 1

function NumMec = tp1_92993()

    %% Init Vars
    NumMec = 92993;
    
    %% Open Image
    
%     addpath('../')
%     listaF=dir('../svpi2022_TP1_img_*.png');

    addpath('../sequencias/Seq160')
    listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

    MaxImg = size(listaF,1);
   
    for idxImg = 1:MaxImg

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));
        
        A = im2double(imread(imName));
        
        %% SubImages
        
        minSize = 0.2; % min nnz for aceptable boundary (percentage)
        minWidth = 0.04; % min width of subimage (percentage)
       
        cutx = -3; % increase area of boundary
        cuty = -3; % 
        reductRoted = 2; % reduce after rotation
        rot = true; % rotate
        extend = false; % extend to rectangular subimage
        relSizes = 1.2; % relative width to height

        % Find rotated dices
        [regionsRotated,fmaskRot] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,zeros(size(A)),reductRoted);
        
        RDO = numel(regionsRotated);
        
        cutx = -1; 
        cuty = -1; 
        extend = true;
        rot = false;
        relSizes = 3; 

        % Find other subimages
        [regionsNormal,~] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,reductRoted);

        % Merge all subimages
        regions = [regionsRotated, regionsNormal]; 
        N = numel(regions);
        
        
        %% Vars
        PntCartas = [];
        numDomsRoted = 0;
        numDadosRoted = 0;
        tDom = 0;   
        tDice = 0;
        tCard = 0;
        Ouros = 0;
        Copas = 0;
        tDuplas = 0;
        PntDom = 0;
        PntDad = 0;
        
        % definem parte da imagem que é o naipe e numero
        pxNN = 0.14; 
        pyNN = 0.25; 
             
        accept = 5; % acept as center points for marker in getNaipe0
        
        % Comparison symbols matrices
        copa = getCopaMatrix();
        ouro = strel('diamond',250).Neighborhood;
        
        % Tolerance to be a symbol (average different pixels)
        tolOuros = 0.12; 
        tolCopas = 0.20;
        
        % final value of average different pixels
        meansOuros = -ones(N,1);
        meansCopa = -ones(N,1);
        scNaipe = 10; % scaling for comparison
        
        % Dices
        percRotate = 0.2; % pecentage of area (border zone) -> rodado
        posDia = 2; % larger outside diamond
        negDia = -1; % inner diamond
        reductRoted = 6; % reduction in the image to get the final diamond
      
        % Domino
        percCenterDom = 0.04; % center vertical line percentage
        perAreaWhiteLine = 0.1; % percentage of area to be acepted as a line in the center
        percBordDom = 0.02; % cut borders of original domino subimage
        
        %% Rotated Dices
        
        for k = 1:RDO
            B = regions{k};
            
            [~,Nb] = edgeRotDice(B);    
        
            if (Nb>6 || Nb==0) , continue, end  % NOISE
            
            tDice = tDice + 1;
        
            numDadosRoted = numDadosRoted + 1;
            PntDad = PntDad + Nb;
           
        end
        
        %% Normal
    
        for k = RDO+1:N
            
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
            if nnz(C) < minNNZ , continue, end 
            
                
            % Rectangular (excludes dices)
            if sx ~= sy
                rotated = false;
        
                B = edgeDice(regions{k});    
                
                if sx > sy % rotate to horizontal
                    B = rot90(B);
                    regions{k} = rot90(regions{k});
                    rotated = true;
                    sy = size(B,2);
                    sx = size(B,1);
                end
        
                % Check Central Vertical Line
                t1 = round(sy*(0.5-percCenterDom/2));
                t2 = round(sy*(0.5+percCenterDom/2));
                area = round(percCenterDom*sy*sx);
    
                % Main boundary on central vertical line
                Bound = bwboundaries(B(:,t1:t2),'noholes');
                if numel(Bound) == 0 % no vertical line
                    vertlines = zeros(size(B(:,t1:t2)));
                else
                    bd = Bound{1};
                    vertlines = poly2mask(bd(:,2), bd(:,1),sx,t2-t1);
                    vertlines = bwmorph(vertlines,"fatten");
                    vertlines = imdilate(vertlines,ones(3,1));
                end
        
                if nnz(vertlines) > perAreaWhiteLine * area % Dominos
                           
                    % Pintas de cada lado
                    Nb1 = getHalfPintas(regions{k},1,percBordDom,t1,t2);

                    Nb2 = getHalfPintas(regions{k},2,percBordDom,t2,t2);

        
                    if (Nb1>6 || Nb2>6) , continue, end % invalid number of pintas
                    
                    
                    if (rotated)
                        numDomsRoted = numDomsRoted + 1;
                    end

                    tDom = tDom + 1;

                    
                    PntDom = PntDom + Nb1 + Nb2;

                    if Nb1 == Nb2
                        tDuplas = tDuplas + 1;
                    end
        
        
                else % cards
        
                    B = double(regions{k});
                    
                    B = autobin(imadjust(B));
                    B = double(cleanCorner(B,pxNN,pyNN)); % remove borders (with naipe and number)
                    B = autobin(imadjust(B));
                    
                    B = edge(B,'roberts');
                    B = bwmorph(B,'bridge');
                    B = bwareaopen(B,round(0.5*size(B,1)));
        
        
                    [~,Nb] = bwlabel(B);

                    if (Nb>9 || Nb==0) , continue, end 
                            
                    
                    D = autobin(imadjust(regions{k}));

                    % Clean irrelevant corners of the card
                    dx = round(pxNN*size(B,1));
                    dy = round(pyNN*size(B,2));
                    D(1:dx,1:dy) = 0;
                    D(end-dx:end,end-dy:end) = 0;
                    
                    PntCartas = [PntCartas Nb];
                    tCard = tCard + 1;
                    
                    % classify naipe from pintas
                    [resO,meansOuros(k),resC,meansCopa(k)] = classAllNaipe(D,ouro,copa,tolOuros,tolCopas,pxNN,pyNN,accept,scNaipe);

                    meansx = [meansOuros(k), meansCopa(k)];
                    resx = [resO,resC];


                    [~,sortedI] = sort(meansx); % use most probable valid naipe
                    for idx = sortedI
                        if resx(idx)
                            if idx==1
                                Ouros = Ouros + 1;
                            else
                                Copas = Copas + 1;
                            end
                            break
                        end
                    end
                    
        
        
                end
            else % DADOS
                
                % Perceber se estao a 45º
                dado1 = autobin(imadjust(regions{k}));

                [res,B2] = rotateDiceIf(dado1,regions{k},percRotate,posDia,negDia, reductRoted);

                if res % foi rodado
                    B = B2;
                end

                % Determinar pintas
                B = bwareaopen(B,round(0.3*size(B,1)));
        
        
                [~,Nb] = bwlabel(B);
                if (Nb>6 || Nb==0) ,continue, end % NOISE

                if res
                    numDadosRoted = numDadosRoted + 1;
                end

                tDice = tDice + 1;
                PntDad = PntDad + Nb;
            end
        
        end
  
        
        %% Save Vars
                
        PntCartas = sort(PntCartas);
        StringPT = strjoin(string(PntCartas),'');
        
        RDO = tDom - numDomsRoted;
        RFO = tDice - numDadosRoted;
        
        CopOuros = Ouros + Copas;
        EspPaus = tCard - CopOuros;
       
        
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, ...
            RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);

        writetable(T,'tp1_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')

    end
    

end

function Nb = getHalfPintas(A,idx,perc,t1,t2) 
    % get number of pintas in each half of the domino
    
    sy = round(size(A,1)*perc)*2;
    sx = round(size(A,2)*perc);

    if idx == 1
        B = A(sy:end-sy,sx:t1);
    else
        B = A(sy:end-sy,t2:end-sx);
    end
    
    B = imadjust(B);
    B = medfilt2(B,[5 5]);
    B = imadjust(B);
    B = autobin(B);
    B = bwmorph(B,'close',inf);

    B = edge(B,'sobel');
    B = bwmorph(B,'close',inf);
    B = bwmorph(B,'remove');
    B = bwareaopen(B,round(0.4*size(B,1)));

    [L,Nprev] = bwlabel(B);

    Nb = 0;
    for x = 1:Nprev % select each boundary
        D = (L==x);
        BB = bwboundaries(D,'noholes');
        boundary = BB{1};
    
        M = poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)); % from the boundary to a mask matrix (region)
        
        % remove all zeros rows and cols
        clean0s = M(:,any(M,1));
        clean0s = clean0s(any(clean0s,2),:);
    
        sizesS = sort(size(clean0s));
        if sizesS(1) < 0.7 * sizesS(2), continue, end % too far from square
        if isempty(clean0s),continue,end
    
        Nb = Nb + 1;    
    
    end
end



function [res,B] = rotateDiceIf(dado1,unaltered,percRotate,posDia,negDia,reductRoted)
    % check if dice is rotated 45º, rotate it to 0º 

    res = false;

    % diamond exterior
    A = strel('diamond',floor(size(dado1,1)/2)+posDia); %+2
    dia = A.Neighborhood;

    % diamond interior
    C = strel('diamond',floor(size(dado1,1)/2)+negDia); %-1
    diamin = C.Neighborhood;
    deltas = round((size(dia,1)-size(diamin,1))/2);
    d2 = zeros(size(dia));
    d2(deltas+1:end-deltas,deltas+1:end-deltas) = diamin;

    % zona esperada para a edge
    zona = dia & not(d2);
    area = nnz(zona);

    % edges
    edges = maskRotated(dado1);
    B = dado1;

    if nnz(edges(zona(1:size(edges,1),1:size(edges,1)))) > percRotate * area % is rotated 45º
        res = true;
        B = rotateDice(unaltered,reductRoted);
    end

end

function B = edgeDice(ori) % edging algorithm for Dices
    B =  medfilt2(ori);
    B = imadjust(B);
    B = autobin(B);
    B = bwareaopen(B,round(0.3*size(B,1)));
    B = bwmorph(B,'remove');
    B = bwareaopen(B,round(0.3*size(B,1)));
end


function [B,Nb] = edgeRotDice(original) % edging algorithm for Rotated Dices

        B = edgeDice(original);
    
        if nnz(medfilt2(B)) > 10 % if there is stil noise
            B = original;
            B =  medfilt2(B);
            B = imadjust(B);
            B = autobin2th(B);
    
            B =  medfilt2(B);
            B = bwmorph(B,'remove');
            B = bwmorph(B,'close');
            B = bwareaopen(B,round(0.5*size(B,1)));
    
            [~,Nb] = bwlabel(B);

            while nnz(B) > 100 * Nb % iteratively reduce noise
                B =  medfilt2(B);
                B = bwmorph(B,'remove');
                B = bwareaopen(B,round(0.5*size(B,1)));
                [~,Nb] = bwlabel(B);
            end
        end
    
        [~,Nb] = bwlabel(B);
end

function B = edging(A) % general edging algorithm
    B = A;
    %     B = medfilt2(B);
    B = edge(B,'roberts');
    B = bwareaopen(B,round(0.5*size(B,1)));
    B = bwmorph(B,'close');
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


function B = cleanCorner(B,px,py)
    dx = round(px*size(B,1));
    dy = round(py*size(B,2));

    B(1:dx,end-dy:end) = 0;
    B(end-dx:end,end-dy:end) = 0;
    B(1:dx,1:dy) = 0;
    B(end-dx:end,1:dy) = 0;
end

function copa = getCopaMatrix() % Generate Matrix of Copa symbol
    N = 501;
    maxR = round((N-1)/2);
    scH = 1e4;
    scIn = 200;
    
    A = false(N,N);
    idx = 1;
    for x=-maxR:maxR
        idy = 1;
        for y = -maxR:maxR
            if (x^2 + y^2 - scH)^3 < scIn * x^2 * y^3
                A(end-idy,idx) = true;
            end
            idy = idy + 1;
        end
        idx = idx +1;
    end
    
    % clean zero rows and cols
    copa = A(any(A,2),:);
    copa = copa(:,any(copa,1));
end
    
function [res,meanC] = classNaipe(carta,naipe,px,py,tol,acept,scNaipe)
    % classify each naipe from 1 pinta

    carta = double(carta);
    clean0s = getNaipe(carta,px,py,acept);
    
    if nnz(clean0s) == 0
        meanC = -1;
        res = false;
        return
    end
    
    meanC = mean(imresize(clean0s,scNaipe) ~= imresize(naipe,scNaipe * size(clean0s)),'all');
    
    res = meanC < tol;

end


function res = getNaipe(carta,px, py,acept)
    
    B = carta;
    
    % number and naipe zone
    dx = round(px*size(B,1)); % 0.14
    dy = round(py*size(B,2)); % 0.25??
    
    CantoSup = rot90(B(1:dx,end-dy:end));
        
    % only naipe zone
    dx2 = round(0.55*size(CantoSup,1));
    res = autobin2th(imadjust(CantoSup(dx2:end,:)));
    
    
    % centroid points of the relevant region
    temp = bwmorph(res,'shrink', inf);
    ppi = filter2([1 1 1; 1 -8 1; 1 1 1], temp);
    marker = (abs(ppi)>acept);
    indexes = find(marker);
    prev = logical(res);
    curArea = 0;
    
    % select the region with largest error and use it as image
    for i=1:length(indexes)
        mk2 = false(size(marker));
        mk2(indexes(i)) = true;
        temp = imreconstruct(mk2, prev);
        Ar = bwarea(temp);
        if  Ar > curArea
            curArea = Ar;
            res = temp;
        end
    end
    
    % clean all zeros rows/cols
    res = res(:,any(res,1));
    res = res(any(res,2),:);

end

function [resO,meanO,resC,meanC] = class1Naipe(B,ouro,copa,tolO,tolC,px,py,acept,scNaipe)
    % classify naipe next to the number
    [resO,meanO] = classNaipe(B,ouro,px,py,tolO,acept,scNaipe);
    [resC,meanC] = classNaipe(B,copa,px,py,tolC,acept,scNaipe);

end

function [resO,meanO,resC,meanC] = classAllNaipe(carta,ouro,copa,tolO,tolC,px,py,acept,scNaipe)
    % classify all symbols except the one next to the number
    
    meanC = 0;
    meanO = 0;
    
    B = carta; % working image
    dx = round(px*size(B,1)); % 0.14
    
    % remove all zones with number
    B(1:dx,:)=0;
    B(end-dx:end,:)=0;
    
    B = double(rot90(B)); % vertical is better

    B = edging(B);
    
    [L,Nb] = bwlabel(B);
    
    count = 0;
    for x =1:Nb % select each boundary
        C = (L==x);
        BB = bwboundaries(C,'noholes');
        boundary = BB{1};
    
        M = poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)); % from the boundary to a mask matrix (region)
    
        if median(boundary(:,1))>0.4*size(B,1) % invert lower card symbols
            M = rot90(rot90(poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2))));
        end
    
    
        % remove all zeros rows and cols
        clean0s = M(:,any(M,1));
        clean0s = clean0s(any(clean0s,2),:);
    
        if nnz(clean0s)< 3 * size(clean0s,1) , continue, end
        if isempty(clean0s),continue,end
    
        count = count + 1;
    
    
        % diferences to symbol matrix
        meanO = meanO + mean(imresize(clean0s,scNaipe)~=imresize(ouro,scNaipe*size(clean0s)),'all');
        meanC = meanC + mean(imresize(clean0s,scNaipe)~=imresize(copa,scNaipe*size(clean0s)),'all');
    
    
    end

    meanC = meanC/count;
    resC = meanC < tolC;

    meanO = meanO/count;
    resO = meanO < tolO;
    
    if count == 0 % did not find any valid middle card symbol
        [resO,meanO,resC,meanC] = class1Naipe(carta,ouro,copa,tolO,tolC,px,py,acept,scNaipe);
        return
    end
    

end


function B2 = maskRotated(B)  
    % mask for rotated dices in main image

    SE1 = [0 0 1
        0 1 0
        1 0 0];
    SE2 = [1 0 0
        0 1 0
        0 0 1];
    
    B2 = edge(B,'sobel','vertical');
    B2 = imclose(B2,SE1);
    B2 = imclose(B2,SE2);

    B2 = bwmorph(B2,'bridge',2);
end


function B = maskNormal(A)
    % mask for all other subimages
    
    B = edge(A,'roberts');
    B = bwmorph(B,'bridge');
end

function [regions,fullMask] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskPrev,reductRoted)
    % get all subimages(regions)

    if rot
        B = maskRotated(A);
    else
        B = maskNormal(A);
    end
    
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
        
        selected = A.*mask;
    
        if rot && ~extend % rotate
            selected = selected(:,any(selected,1));
            selected = selected(any(selected,2),:);
    
            selected = rotateDice(selected,reductRoted);
        end
    
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

function B = rotateDice(unaltered, reductRoted)

    % rodar
    A = imrotate(unaltered,45);
    
    % reduzir imagem ao dado
    x = size(unaltered,1);
    xmeio = round(size(A,1)/2);
    
    l = floor(x/sqrt(2));
    deltal = round(l/2)- reductRoted; 
    
    B = double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal));

end
