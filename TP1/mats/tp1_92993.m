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

    addpath('../sequencias/Seq530')
    listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');

    MaxImg = size(listaF,1);
   
    for idxImg = 1:MaxImg

        imName = listaF(idxImg).name;
        
        tDuplas = 0;
        PntDom = 0;
        PntDad = 0;
        
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
        domKs = [];
        diceKs = [];
        cardKs = [];
        rodados = [];
        PntCartas = [];
        noiseKs = [];
        ourosk = [];
        copask = [];
        numDomsRoted = 0;
        
        % definem parte da imagem que é o naipe e numero
        pxNN = 0.14; 
        pyNN = 0.25; 
        
        % definem percentagem de nnz para separar tipos de cartas
        percWhiteCorner = 0.10; 
        percBlackCorner = 0.05; 
        
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
        strRes = ["Ouros","Copas"];
        
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
        
            if (Nb>6 || Nb==0) % NOISE
                noiseKs = [noiseKs k];
                continue
            end
        
            diceKs = [diceKs k];
            rodados = [rodados k];
            PntDad = PntDad + Nb;
           
        end
        
        %% Normal
    
        for k = RDO+1:N
        
    %         B = medfilt2(regions{k});
    %         B = imadjust(B);
    %         B = autobin(B,true);
    %         B = bwareaopen(B,round(0.5*size(B,1)));
    
            B = imadjust(regions{k});
            B = medfilt2(B);
            B = imadjust(B);
            B = autobin(B,true);
            B = bwmorph(B,'close',inf);
    %         B = bwareaopen(B,round(0.5*size(B,1)));
        
        
            sx = size(B,1);
            sy = size(B,2);
        
            % Test Noise
            C = bwmorph(B,'erode',2);
            minNNZ =  0.01 * nnz(B) +1;
    
            if nnz(C) < minNNZ
                noiseKs = [noiseKs k];
                continue
            end
        
        
            % Rectangular (excludes dices)
            if sx ~= sy
                rotated = false;
        
                Bold = B;
                B = edgeDice(regions{k});

%                 B =  medfilt2(regions{k});
%                 B = imadjust(B);
%                 B = autobin(B,false);
%                 B = edge(B,'roberts');
%                 B = bwmorph(B,'bridge');
    
                
                if sx > sy % rotate to horizontal
                    B = rot90(B);
                    Bold = rot90(Bold);
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
                    B = Bold;
                    B(:,t1:t2) = 0; % remove line
        
                    % clean borders
                    B(1:round(sy*percBordDom ),:)= 0;
                    B(end-round(sy*percBordDom ):end,:)= 0;
                    B(:,1:round(sx*percBordDom ))= 0;
                    B(:,end-round(sx*percBordDom * 2):end)= 0; % <<<<<

        
                    % Detect Pintas
                    B = edge(B,'roberts');
                    B = bwmorph(B,'bridge',2);
                    B = bwmorph(B,'close');
                    B = bwareaopen(B,round(0.5*size(B,1)));
                    [~,Nb] = bwlabel(B);
        
                    % Pintas de cada lado
                    Nb1 = getHalfPintas(regions{k},1,percBordDom,t1,t2);

                    Nb2 = getHalfPintas(regions{k},2,percBordDom,t2,t2);

        
                    if (Nb1>6 || Nb2>6) % invalid number of pintas
                        noiseKs = [noiseKs k];
                        continue
                    end
                    
                    if (rotated)
                        numDomsRoted = numDomsRoted + 1;
                    end

                    domKs = [domKs k];

                    
                    PntDom = PntDom + Nb1 + Nb2;

                    if Nb1==Nb2
                        tDuplas = tDuplas + 1;
                    end
        
        
                else % cards
        
                    B = double(regions{k});
                    
                    B = autobin(imadjust(B),false);
                    B = double(cleanCorner(B,pxNN,pyNN)); % remove borders (with naipe and number)
                    B = autobin(imadjust(B),false);
                    
                    B = edge(B,'roberts');
                    B = bwmorph(B,'bridge');
                    B = bwareaopen(B,round(0.5*size(B,1)));
        
        
                    [~,Nb] = bwlabel(B);

                    if (Nb>9 || Nb==0)
                        noiseKs = [noiseKs k];
                        continue
                    end
        
                    
                    D = autobin(imadjust(regions{k}),false);

                    [tipo,D] = sepCartas(D,percWhiteCorner,percBlackCorner,pxNN,pyNN); % classify orientation of a card

                    if tipo == 0 % invalid card (does nothing)
                        noiseKs = [noiseKs k];
                    else 
                        PntCartas = [PntCartas Nb];
                        cardKs = [cardKs k];
    
                        [resO,meansOuros(k),resC,meansCopa(k)] = classAllNaipe(D,ouro,copa,tolOuros,tolCopas,pxNN,pyNN,accept,tipo,scNaipe);

                        meansx = [meansOuros(k), meansCopa(k)];
                        resx = [resO,resC];
    
    
                        [~,sortedI] = sort(meansx);
                        for idx = sortedI
                            if resx(idx)
                                if idx==1
                                    ourosk = [ourosk k];
                                else
                                    copask = [copask k];
                                end
                                break
                            end
                        end
                    end
        
        
                end
            else % DADOS
                
                % Perceber se estao a 45º
                dado1 = autobin(imadjust(regions{k}),false);

                [res,B2] = rotateDiceIf(dado1,regions{k},percRotate,posDia,negDia, reductRoted);

                if res
                    fprintf("Rodado %d\n",k)
                    B = B2;
                    rodados = [rodados k];
                end

                % Determinar pintas
        
    %             B = bwareaopen(B,round(0.5*size(B,1)));
    %             B = bwmorph(B,'remove');
                B = bwareaopen(B,round(0.3*size(B,1)));
        
        
                [~,Nb] = bwlabel(B);
                if (Nb>6 || Nb==0) % NOISE
                    noiseKs = [noiseKs k];
                    if ismember(k,rodados)
                        rodados(rodados==k) = [];
                    end
                    continue
                end

                diceKs = [diceKs k];
                PntDad = PntDad + Nb;
        
        
            end
        
        end
  
        
        %% Save Vars
                
        PntCartas = sort(PntCartas);
        
        StringPT = strjoin(string(PntCartas),'');
        
        tDom = length(domKs);
        RDO = tDom - numDomsRoted;
        
        tDice = length(diceKs);
        RFO = tDice - length(rodados);
        
        tCard = length(cardKs);
        
        Ouros = length(ourosk);
        Copas = length(copask);
        CopOuros = Ouros + Copas;
        
        EspPaus = tCard - CopOuros;
       
        
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, ...
            RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);

        writetable(T,'tp1_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')

    end
    

end

function Nb = getHalfPintas(A,idx,perc,t1,t2)
    
    sy = round(size(A,1)*perc)*2;
    sx = round(size(A,2)*perc);

    if idx == 1
        B1 = A(sy:end-sy,sx:t1);
    else
        B1 = A(sy:end-sy,t2:end-sx);
    end
    
    B = B1;

    B = imadjust(B);
    B = medfilt2(B,[5 5]);
    B = imadjust(B);
    B = autobin(B,false);
    B = bwmorph(B,'close',inf);
%     B = bwmorph(B,'remove');


    B = edge(B,'sobel');
%     B = bwmorph(B,'bridge',inf);
    B = bwmorph(B,'close',inf);
    B = bwmorph(B,'remove');
    B = bwareaopen(B,round(0.4*size(B,1)));

    [L,Nb] = bwlabel(B);

    count = 0;
    for x =1:Nb % select each boundary
        D = (L==x);
        BB = bwboundaries(D,'noholes');
        boundary = BB{1};
    
        M = poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)); % from the boundary to a mask matrix (region)
        
        % remove all zeros rows and cols
        clean0s = M(:,any(M,1));
        clean0s = clean0s(any(clean0s,2),:);
    
        sizesS = sort(size(clean0s));
        if sizesS(1) < 0.7 * sizesS(2) , continue, end
        if isempty(clean0s),continue,end
    
        count = count + 1;    
    
    end

    Nb = count;   
    
end


function [res,B] = rotateDiceIf(dado1,unaltered,percRotate,posDia,negDia, reductRoted)
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

    if nnz(edges(zona(1:size(edges,1),1:size(edges,1)))) > percRotate * area %.2

        res = true;

        % rodar
        A = imrotate(unaltered,45);

        % reduzir imagem ao dado
        x = size(dado1,1);
        xmeio = round(size(A,1)/2);

        l = floor(x/sqrt(2));
        deltal = round(l/2)-reductRoted; % 6


%         B = autobin(imadjust(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal))),true);
        B = edgeDice(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal)));

    end

end

function B = edgeDice(ori)
    B =  medfilt2(ori);
    B = imadjust(B);
    B = autobin(B,false);
%     B = bwareaopen(B,30);
%     B = bwmorph(B,'remove');
    B = bwareaopen(B,round(0.3*size(B,1)));
    B = bwmorph(B,'remove');
    B = bwareaopen(B,round(0.3*size(B,1)));
end

function [B,Nb] = edgeRotDice(original)

        B = edgeDice(original);
    
        if nnz(medfilt2(B))>10
            B = original;
            B =  medfilt2(B);
            B = imadjust(B);
            B = autobin(B,true);
    
            B =  medfilt2(B);
            B = bwmorph(B,'remove');
            B = bwmorph(B,'close');
            B = bwareaopen(B,round(0.5*size(B,1)));
    
            [~,Nb] = bwlabel(B);
            while nnz(B)>100*Nb
                B =  medfilt2(B);
                B = bwmorph(B,'remove');
                B = bwareaopen(B,round(0.5*size(B,1)));
                [~,Nb] = bwlabel(B);
            end
        end
    
        [~,Nb] = bwlabel(B);
end

function B = edging(A)
    B = A;
    %     B = medfilt2(B);
    B = edge(B,'roberts');
    B = bwareaopen(B,round(0.5*size(B,1)));
    B = bwmorph(B,'close');
    
    %     B = bwmorph(B,'remove'); % remove
    %     B = bwareaopen(B,round(0.2*size(B,1)));
end

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


function B = cleanCorner(B,px,py)
    dx = round(px*size(B,1));
    dy = round(py*size(B,2));
    B(1:dx,end-dy:end) = 0;
    B(end-dx:end,end-dy:end) = 0;
    B(1:dx,1:dy) = 0;
    B(end-dx:end,1:dy) = 0;
end

function [res,B] = sepCartas(B,perc,perc0,px,py)

    % define areas of relevance
    dx = round(px*size(B,1));
    dy = round(py*size(B,2));
    area = dx*dy;
    nnzSupDir = nnz(B(1:dx,end-dy:end));
    nnzInfDir = nnz(B(end-dx:end,end-dy:end));
    nnzSupEsq = nnz(B(1:dx,1:dy));
    nnzInfEsq = nnz(B(end-dx:end,1:dy));
    
    if (nnzInfEsq > perc*area && nnzSupDir > perc*area  && nnzInfDir < perc0*area && nnzSupEsq < perc0*area)
        % clean not relevant corners
        B(1:dx,1:dy) = 0;
        B(end-dx:end,end-dy:end) = 0;
        % return as type 1
        res = 1;
    
    elseif (nnzInfDir > perc*area && nnzSupEsq > perc*area && nnzInfEsq < perc0*area && nnzSupDir < perc0*area)
        % clean not relevant corners
        B(end-dx:end,1:dy) = 0;
        B(1:dx,end-dy:end) = 0;
        % return as type 2
        res = 2;
    else
        res = 0; % error: invalid card or badly binarized
    end
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
    
    function [res,meanC] = classNaipe(carta,tipo,naipe,px,py,tol,acept,scNaipe)
    carta = double(carta);
    clean0s = getNaipe(carta,tipo,px,py,acept);
    
    if nnz(clean0s) == 0
        fprintf("clean0s vazio\n")
        meanC = -1;
        res = false;
        return
    end
    
    %     naipe = bwmorph(naipe,'remove'); % usar so a border
    %     clean0s = imresize(clean0s,sc);
    %     clean0s = bwmorph(clean0s,'remove');
    % %     clean0s = bwmorph(clean0s,'dilate');
    %     naipe = imdilate(naipe,ones(1,3));
    %     meanC = mean(clean0s~=imresize(naipe,size(clean0s)),'all');
    
    meanC = mean(imresize(clean0s,scNaipe) ~= imresize(naipe,scNaipe * size(clean0s)),'all');
    
    res = meanC < tol;

end


function res = getNaipe(carta, tipo,px, py,acept)
    
    B = carta;
    
    % number and naipe zone
    dx = round(px*size(B,1)); % 0.14
    dy = round(py*size(B,2)); % 0.25??
    
    if tipo == 1
        CantoSup = rot90(B(1:dx,end-dy:end));
    elseif tipo == 2
        CantoSup = rot90(rot90(rot90(B(1:dx,1:dy))));
    end
    
    % only naipe zone
    dx2 = round(0.55*size(CantoSup,1));
    res = autobin(imadjust(CantoSup(dx2:end,:)),true);
    
    
    % centroid points of the relevant region
    ola = bwmorph(res,'shrink', inf);
    ppi = filter2([1 1 1; 1 -8 1; 1 1 1], ola);
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

% function [resO,meanO,resC,meanC,resE,meanE] = class1Naipe(B,ouro,copa,espada,tolO,tolC,tolE,px,py,acept,tipo,scNaipe)
function [resO,meanO,resC,meanC] = class1Naipe(B,ouro,copa,tolO,tolC,px,py,acept,tipo,scNaipe)
    % classify naipe next to the number
    [resO,meanO] = classNaipe(B,tipo,ouro,px,py,tolO,acept,scNaipe);
    [resC,meanC] = classNaipe(B,tipo,copa,px,py,tolC,acept,scNaipe);
%     [resE,meanE] = classNaipe(B,tipo,espada,px,py,tolE,acept,scNaipe);

end

% function [resO,meanO,resC,meanC,resE,meanE] = classAllNaipe(carta,ouro,copa,espada,tolO,tolC,tolE,px,py,acept,tipo,scNaipe)
function [resO,meanO,resC,meanC] = classAllNaipe(carta,ouro,copa,tolO,tolC,px,py,acept,tipo,scNaipe)
    % classify all symbols except the one next to the number (tipo not
    % relevant )
    
    meanC = 0;
    meanO = 0;
%     meanE = 0;
    
    B = carta; % working image
    dx = round(px*size(B,1)); % 0.14
    
    % remove all zones with number
    B(1:dx,:)=0;
    B(end-dx:end,:)=0;
    
    B = double(rot90(B)); % vertical is better
    if tipo == 2
        B = rot90(rot90(B)); % needed?? <<<<<<<<<<<<<<<<<<<<<<<
    end
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
    
    
        meanO = meanO + mean(imresize(clean0s,scNaipe)~=imresize(ouro,scNaipe*size(clean0s)),'all');
        meanC = meanC + mean(imresize(clean0s,scNaipe)~=imresize(copa,scNaipe*size(clean0s)),'all');
%         meanE = meanE + mean(imresize(clean0s,scNaipe)~=imresize(espada,scNaipe*size(clean0s)),'all');
    
    
    end

    meanC = meanC/count;
    resC = meanC < tolC;
    
%     meanE = meanE/count;
%     resE = meanE < tolE;

    meanO = meanO/count;
    resO = meanO < tolO;
    
    if count == 0 % did not find any valid middle card symbol
%         [resO,meanO,resC,meanC,resE,meanE] = class1Naipe(carta,ouro,copa,espada,tolO,tolC,tolE,px,py,acept,tipo,scNaipe);
        [resO,meanO,resC,meanC] = class1Naipe(carta,ouro,copa,tolO,tolC,px,py,acept,tipo,scNaipe);
        return
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

    B2 = bwmorph(B2,'bridge',2);
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
    
    [Bx,~,Nb] = bwboundaries(B);
    
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
    
        if nnz(mask.*fmaskPrev), continue, end
    
        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);
        if (mean(mask0s,'all') < 0.4), continue, end
    
        % remove weird shapes
        sizesT = sort(size(mask0s));
        if sizesT(2) > relSizes * sizesT(1) || sizesT(1) < minWidth * sx, continue, end
    
        % remove already found
        if nnz(mask.*fmaskPrev), continue, end
    
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
        if sizesT(2) > relSizes * sizesT(1) || sizesT(1) < minWidth * sx , continue, end
    
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
        selected = selected(any(selected,2),:);
    
        sizesT = sort(size(selected));
        if sizesT(2) < 1.05 * sizesT(1) 
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
    deltal = round(l/2)-reductRoted; % 6
    
    B = double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal));
    %     B = autobin(imadjust(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal))));

end


function A = getEspadaMatrix()
    A=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
        0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
        0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
        0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
        0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
        0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
        0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
        0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
        0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
        0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
        0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
        0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
        0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
        0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
        0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
        0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0];


end

