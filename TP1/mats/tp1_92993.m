% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Trabalho Pratico 1

%% A FAZER
% Verificar semelhanca dos naipes so no interior do esperado
% Edge diferente no rotate dice
% poly2mask em mais sitios



%%



function NumMec = tp1_92993()
%%

close all
clear
clc

%% Init Vars
NumMec = 92993;

%% Open Image

%     addpath('../')
%     listaF=dir('../svpi2022_TP1_img_*.png');
% 
% addpath('../sequencias/Seq160')
% listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

    addpath('../sequencias/Seq530')
    listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');

MaxImg = size(listaF,1);
showplot = false;
    for idxImg = 1:MaxImg
    % idxImg = 2; showplot = true;
    fprintf("idxImg=%d\n",idxImg);
    
    tDuplas = 0;
    PntDom = 0;
    PntDad = 0;
    
    imName = listaF(idxImg).name;
    NumSeq = str2double(imName(18:20));
    NumImg = str2double(imName(22:23));
    
    A = im2double(imread(imName));
    %         imshow(A)
    
    %% SubImages
    
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
    
    N=numel(regionsRotated);
    RDO = N;
    
    if showplot
        SS=ceil(sqrt(N));
        figure(1)
        for k=1:N
            subplot( SS, SS, k);
            imshow(regionsRotated{k})
            xlabel(k)
        end
    end
    
    cutx = -1; % -1
    cuty = -1; % -1
    extend = true;
    rot = false;
    relSizes = 3; % 3
    tic
    [regionsNormal,~] = getSubImages(A,rot,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,reductRoted);
    toc
    
    if showplot
        N=numel(regionsNormal);
        SS=ceil(sqrt(N));
        figure(1)
        for k=1:N
            subplot( SS, SS, k);
            imshow(regionsNormal{k})
            xlabel(k)
        end
    end
    
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
    %         cartas1k = [];
    %         cartas2k = [];
    
    % definem parte da imagem que é o naipe e numero
    pxNN = 0.15; % 0.14
    pyNN = 0.25; % 0.25
    pxCutCenter = 0.12;
    
    % definem percentagem de nnz para separar tipos de cartas
    percWhiteCorner = 0.10; % 0.10    0.15
    percBlackCorner = 0.05; % 0.05    0.10
    
    accept = 5; % acept as center points for marker in getNaipe0
    
    % Comparison symbols matrices
    copa = getCopaMatrix();
    ouro = strel('diamond',250).Neighborhood;
    espada = getEspadaMatrix();
    
    % Tolerance to be a symbol (average different pixels)
    tolOuros = 0.12; % 0.12     0.12 0.2
    tolCopas = 0.20; % 0.2    0.12 0.2
    tolEspadas = 0.3; % 0.3    0.41    0.12 0.2
    
    % final value of average different pixels
    meansOuros = -ones(N,1);
    meansCopa = -ones(N,1);
    meansEspadas = -ones(N,1);
    scNaipe = 10; % scaling for comparison
    
    strRes = ["Ouros","Espadas","Copas"];
    
    % Dices
    % percRotate = 0.2; % pecentage of area (border zone) -> rodado
    % posDia = 2; % larger outside diamond
    % negDia = -1; % inner diamond
    % edgeGrad = 1; % gradient that defines an edge
    % reductRoted = 6; % reduction in the image to get the final diamond
    
    %% Rotated Dices
    
    if showplot
        figure(3)
    end
    
    regions = [regionsRotated, regionsNormal];
    for k=1:RDO
    
        rodados = [rodados k];
    
        B = regions{k};
    
        sx = size(B,1);
        sy = size(B,2);
    
        B =  medfilt2(B);
        B = imadjust(B);
        B = autobin(B,false);
        B = bwareaopen(B,30);
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
            while nnz(B)>100*Nb
                B =  medfilt2(B);
                B = bwmorph(B,'remove');
                B = bwareaopen(B,round(0.5*size(B,1)));
                [~,Nb] = bwlabel(B);
            end
        end
    
        [~,Nb] = bwlabel(B);
    
        if (Nb>6 || Nb==0) % NOISE
            noiseKs = [noiseKs k];
            rodados(rodados==k) = [];
            fprintf("IdxImg:%d,Removeu rodado %d, Nb:%d\n",idxImg,k,Nb)
            B = ones(size(B));
        else
            diceKs = [diceKs k];
            PntDad = PntDad + Nb;
        end
    
        if showplot
            SS = ceil(sqrt(RDO));
            subplot(SS,SS,k);
            imshow(B)
            str = sprintf('DR.%d,N=%d',k,Nb);
            xlabel(str);
        end
    
        %             regions{k} = double(B);
    
    
    end
    
    %% Normal
    
    N=numel(regions);
    for k=RDO+1:N
    
        B = medfilt2(regions{k});
        B = imadjust(B);
        %             B = autobin(B,false);
        B = autobin(B,true);
        B = bwareaopen(B,round(0.5*size(B,1)));
    
        %             B = autobin(imadjust(regions{k}));
    
        sx = size(B,1);
        sy = size(B,2);
    
        % Test Noise
        C = bwmorph(B,'erode',2);
        minNNZ =  0.01*nnz(B);
    %     fprintf("nnz=%d, m= %d, noise: %d\n",nnz(C),minNNZ ,k)
        if nnz(C) < minNNZ + 1
            noiseKs = [noiseKs k];
            if showplot
                SS = ceil(sqrt(N));
                subplot(SS,SS,k);
                fprintf("nnz=%d, m= %d, noise: %d\n",nnz(C),minNNZ ,k)
                imshow(B)
                xlabel("noise")
            end
            continue
        end
    
    
        % Rectangular (excludes dices)
        if sx ~= sy
            rotated = false;
            % rotate to horizontal
            if sx>sy
                B = rot90(B);
                regions{k} = rot90(regions{k});
                rotated = true;
                sy = size(B,2);
                sx = size(B,1);
            end
    
            % Check Central Vertical Line
            perc = 4/100;
            t1 = 0.5-perc/2;
            t2 = 0.5+perc/2;
            area = round(perc*sy*sx);
    
            [gx,~] = imgradientxy(B(:,round(sy*t1):round(sy*t2)));
            vertlines = gx>0;
    
    
            if nnz(vertlines) > 0.3 * area % Dominos
                B(:,round(sy*t1):round(sy*t2)) = 0; % remove line
    
                % clean borders
                if false
                    perc = 2/100;
                    B(1:round(sy*perc),:)= 0;
                    B(end-round(sy*perc):end,:)= 0;
                    B(:,1:round(sx*perc))= 0;
                    %                     B(:,1:round(sx*perc*2))= 0;
                    B(:,end-round(sx*perc*2):end)= 0;
                end
    
    
                % Detect Pintas
                B = edge(B,'roberts');
                B = bwmorph(B,'close');
                B = bwareaopen(B,round(0.5*size(B,1)));
                [~,Nb] = bwlabel(B);
    
                % Pintas de cada lado
                B1 = B(:,1:round(size(B,2)/2));
                B2 = B(:,round(size(B,2)/2):end);
                [~,Nb1] = bwlabel(B1);
                [~,Nb2] = bwlabel(B2);
    
                if (Nb1>6 || Nb2>6 || Nb==0) % invalid number of pintas
                    fprintf("Remove Domino: %d,%d->%d\n",Nb1,Nb2,Nb);
                    noiseKs = [noiseKs k];
                    B = ones(size(B));
                else
                    if (rotated)
                        numDomsRoted = numDomsRoted + 1;
                    end
                    domKs = [domKs k];
                    if Nb1+Nb2 ~= Nb
                        fprintf("Erro Domino: %d + %d != %d\n",Nb1,Nb2,Nb);
                    end
                    PntDom = PntDom + Nb1 + Nb2;
                    if Nb1==Nb2
                        tDuplas = tDuplas + 1;
                    end
                end
    
                if showplot
                    SS = ceil(sqrt(N));
                    subplot(SS,SS,k);
                    imshow(B)
                    str = sprintf('Dom.%d,N1=%d,N2=%d',k,Nb1,Nb2);
                    xlabel(str);
                end
    
            else % cards
    
                % remove borders (with naipe and number)
                B = regions{k};
                cut = round(pxCutCenter*size(B,1));
                B = B(cut+1:end-cut,:);
                B = autobin(imadjust(B),false);
    
                B = edge(B,'roberts');
                B = bwmorph(B,'close');
                B = bwareaopen(B,round(0.5*size(B,1)));
    
                [~,Nb] = bwlabel(B);
                if (Nb>9 || Nb==0)
                    fprintf("Remove Carta:Nb=%d\n",Nb);
                    noiseKs = [noiseKs k];
                    B = ones(size(B));
                else
    
                    D = autobin(imadjust(regions{k}),false); % previous normal
    
                    [res,D] = sepCartas(D,percWhiteCorner,percBlackCorner,pxNN,pyNN);
                    tipo = res;
                    if res == 0
                        fprintf("Carta NA, k=%d\n",k)
                        tipo = 1; %%%%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    end
                    restipo = res;
    
                    if tipo ~=0
                        PntCartas = [PntCartas Nb];
                        cardKs = [cardKs k];
    
                        [resO,meansOuros(k),resC,meansCopa(k),resE,meansEspadas(k)] = classAllNaipe(D,ouro,copa,espada,tolOuros,tolCopas,tolEspadas,pxNN,pyNN,accept,tipo,scNaipe);
    
                        meansx = [meansOuros(k), meansEspadas(k), meansCopa(k)];
                        resx = [resO,resE,resC];
    
    
                        [~,sortedI] = sort(meansx);
                        str = sprintf("T%d,O:%.2f,C:%.2f,E:%.2f\n%s tp%d,Nb=%d",tipo,meansOuros(k),meansCopa(k),meansEspadas(k),"Desc.",restipo,Nb);
                        for idx=sortedI
                            if resx(idx)
                                str = sprintf("T%d,O:%.2f,C:%.2f,E:%.2f\n%s tp%d,Nb=%d",tipo,meansOuros(k),meansCopa(k),meansEspadas(k),strRes(idx),restipo,Nb);
                                if idx==1
                                    ourosk = [ourosk k];
                                elseif idx==3
                                    copask = [copask k];
                                else
                                end
                                break
                            end
                        end
    
                    end
    
    
    
                end
    
            end
        else
            B = bwareaopen(B,round(0.5*size(B,1)));
            B = bwmorph(B,'remove');
    
    
            [~,Nb] = bwlabel(B);
            if (Nb>6 || Nb==0) % NOISE
                noiseKs = [noiseKs k];
                if ismember(k,rodados)
                    rodados(rodados==k) = [];
                    fprintf("Removeu rodado %d, Nb:%d\n",k,Nb)
                end
                B = ones(size(B));
            else
                diceKs = [diceKs k];
                PntDad = PntDad + Nb;
            end
            if showplot
                str = sprintf('D.%d,N=%d',k,Nb);
            end
    
    
        end
    
        if showplot
            SS = ceil(sqrt(N));
            subplot(SS,SS,k);
            %                   imshow(regions{k})
            imshow(B)
            xlabel(str)
        end
    
    end
    
    
    %             regions{k} = double(B);
    
    
    
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
    CopOuros = Ouros + Copas; %+copas
    
    EspPaus = tCard - CopOuros;
    
    if showplot
        noiseKs
        domKs
        diceKs
        rodados
        cardKs
        ourosk
        copask
        fprintf("Total=%d, Dominos=%d, Dados=%d, Cartas=%d\n",N,tDom,tDice,tCard)
    end
    
    
    %% Write Table Entry
    T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, ...
        RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
    %         if idxImg==1
    %             writetable(T,'tp1_92993.txt', 'WriteVariableNames',false)
    %         else
    writetable(T,'tp1_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')
    %         end
    
    end

%         save


end


% function [res,B] = rotateDice(dado1,unaltered,percRotate,posDia,negDia,edgeGrad, reductRoted)
%     res = false;
%
%     % diamond exterior
%     A = strel('diamond',floor(size(dado1,1)/2)+posDia); %+2
%     dia = A.Neighborhood;
%
%     % diamond interior
%     C = strel('diamond',floor(size(dado1,1)/2)+negDia); %-1
%     diamin = C.Neighborhood;
%     deltas = round((size(dia,1)-size(diamin,1))/2);
%     d2 = zeros(size(dia));
%     d2(deltas+1:end-deltas,deltas+1:end-deltas) = diamin;
%
%     % zona esperada para a edge
%     zona = dia & not(d2);
%     area = nnz(zona);
%
%     % edges
%     [Gmag,~] = imgradient(dado1);
%     edges = Gmag>edgeGrad;
%     B = dado1;
%
%     if nnz(edges(zona(1:size(edges,1),1:size(edges,1)))) > percRotate * area %.2
%
%         res = true;
%
%         % rodar
%         A = imrotate(unaltered,45);
%
%         % reduzir imagem ao dado
%         x = size(dado1,1);
%         xmeio = round(size(A,1)/2);
%
%         l = floor(x/sqrt(2));
%         deltal = round(l/2)-reductRoted; % 6
%
%
%         B = autobin(imadjust(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal))));
%
%     end
%
% end

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

function [resO,meanO,resC,meanC,resE,meanE] = class1Naipe(B,ouro,copa,espada,tolO,tolC,tolE,px,py,acept,tipo,scNaipe)
% classify naipe next to the number
[resO,meanO] = classNaipe(B,tipo,ouro,px,py,tolO,acept,scNaipe);
[resC,meanC] = classNaipe(B,tipo,copa,px,py,tolC,acept,scNaipe);
[resE,meanE] = classNaipe(B,tipo,espada,px,py,tolE,acept,scNaipe);

end

function [resO,meanO,resC,meanC,resE,meanE] = classAllNaipe(carta,ouro,copa,espada,tolO,tolC,tolE,px,py,acept,tipo,scNaipe)
% classify all symbols except the one next to the number (tipo not
% relevant )

meanC = 0;
meanO = 0;
meanE = 0;

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
    meanE = meanE + mean(imresize(clean0s,scNaipe)~=imresize(espada,scNaipe*size(clean0s)),'all');


end

if count == 0 % did not find any valid middle card symbol
    [resO,meanO,resC,meanC,resE,meanE] = class1Naipe(carta,ouro,copa,espada,tolO,tolC,tolE,px,py,acept,tipo,scNaipe);
    return
end

meanC = meanC/count;
resC = meanC < tolC;

meanE = meanE/count;
resE = meanE < tolE;

meanO = meanO/count;
resO = meanO < tolO;

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

