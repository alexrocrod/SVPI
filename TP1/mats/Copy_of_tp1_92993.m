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

%     addpath('../sequencias/Seq160')
%     listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

    addpath('../sequencias/Seq530')
    listaF=dir('../sequencias/Seq530/svpi2022_TP1_img_*.png');

    MaxImg = size(listaF,1);
    showplot = false;
    for idxImg = 1:MaxImg
%     idxImg = 2; showplot = true;
        
        tDuplas = 0;
        PntDom = 0;
        PntDad = 0;

        imName = listaF(idxImg).name;
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));

        A = im2double(imread(imName));
%         imshow(A)
    
        %% SubImages (provisorio)
    
        regionsOrig=vs_getsubimages(A); %extract all regions
        regions=vs_getsubimages(A);
        N=numel(regions);
        SS=ceil(sqrt(N));
        
        if showplot
            figure(1)
            for k=1:N 
                subplot( SS, SS, k);
                imshow(regions{k})
                xlabel(k)
            end
        end

        %% Autobin
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

        % definem percentagem de nnz para separar tipos de cartas
        percWhiteCorner = 0.10; % 0.10    0.15 
        percBlackCorner = 0.05; % 0.05    0.10

        accept=5; % acept as center points for marker in getNaipe0

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
        percRotate = 0.2; % pecentage of area (border zone) -> rodado
        posDia = 2; % larger outside diamond 
        negDia = -1; % inner diamond
        edgeGrad = 1; % gradient that defines an edge
        reductRoted = 6; % reduction in the image to get the final diamond



        if showplot
            figure(7)
        end

        for k=1:N
            if showplot
                subplot(SS,SS,k);
            end
            cut = 2; 
%             regions{k} = medfilt2(filter2(fspecial('average',3),regionsOrig{k}(cut:end-cut,cut:end-cut)));
%             regions{k} = wiener2(regionsOrig{k}(cut:end-cut,cut:end-cut),[5 5]);
%             regions{k} = medfilt2(regionsOrig{k}(cut:end-cut,cut:end-cut));
            regions{k} = regionsOrig{k}(cut:end-cut,cut:end-cut);
            
            
            
            B = autobin(imadjust(regions{k}));
            
            sx = size(B,1);
            sy = size(B,2);

            % Test Noise
            C = bwmorph(B,'erode',2);
            minNNZ =  0.01*nnz(B);
            if nnz(C) < minNNZ
                noiseKs = [noiseKs k];
                if showplot
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
                    perc = 2/100;
                    B(1:round(sy*perc),:)= 0;
                    B(end-round(sy*perc):end,:)= 0;
                    B(:,1:round(sx*perc))= 0;
%                     B(:,1:round(sx*perc*2))= 0;
                    B(:,end-round(sx*perc*2):end)= 0;


                    % Detect Pintas
                    B = edge(B,'roberts');

%                     B = bwareaopen(B,round(0.5*size(B,1)));
%                     B = bwmorph(B,'close');

%                     B = edging(B);
                
                    [~,Nb] = bwlabel(B);

                    % Pintas de cada lado
                    B1 = B(:,1:round(size(B,2)/2));
                    B2 = B(:,round(size(B,2)/2):end);
                    [~,Nb1] = bwlabel(B1);
                    [~,Nb2] = bwlabel(B2);
                    
                    if (Nb1>6 || Nb2>6 || Nb==0) % invalid number of pintas
                        noiseKs = [noiseKs k];
                        B = ones(size(B));
                    else
                        if (rotated)
                            numDomsRoted = numDomsRoted + 1; 
                        end
                        domKs = [domKs k];
                        if Nb1+Nb2 ~= Nb
                            fprintf("Erro Domino: %d + %d != %d\n",Nb1,Nb2,N);
                        end
                        PntDom = PntDom + Nb1 + Nb2;
                        if Nb1==Nb2 
                            tDuplas = tDuplas + 1;
                        end
                    end
                    
                    if showplot
                        imshow(B)
                        str = sprintf('Dom.%d,N1=%d,N2=%d',k,Nb1,Nb2);
                        xlabel(str);
                    end

                else % cards

                    % remove borders (with naipe and number)
                    B = regions{k};
                    cut = round(pxNN*size(B,1));
                    B = B(cut:end-cut,:);
                    B = autobin(imadjust(B));

%                     B = bwareaopen(B,round(0.4*size(B,1))); % 0.5
%                     B = bwmorph(B,'close');

                    B = edging(B);

                    [~,Nb] = bwlabel(B);
                    if (Nb>9 || Nb==0) 
                        noiseKs = [noiseKs k];
                        B = ones(size(B));
                    else
                       
                        D = autobin(imadjust(regions{k})); % previous normal
                        
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
                            str = sprintf("T%d,O:%.2f,C:%.2f,E:%.2f\n%s tp%d",tipo,meansOuros(k),meansCopa(k),meansEspadas(k),"Desc.",restipo);
                            for idx=sortedI
                                if resx(idx)
                                    str = sprintf("T%d,O:%.2f,C:%.2f,E:%.2f\n%s tp%d",tipo,meansOuros(k),meansCopa(k),meansEspadas(k),strRes(idx),restipo);
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
                    if showplot
%                         imshow(regions{k})
                        imshow(B)
                        xlabel(str)
                    end
                    
                end

            else % Quadrados -> Dados e NOISE <<<<<

                % Perceber se estao a 45º
                c2=2;
                dado1 = autobin(imadjust(regionsOrig{k}(c2+1:end-c2,c2+1:end-c2))); 
                
                [res,B2] = rotateDice(dado1,regions{k},percRotate,posDia,negDia,edgeGrad, reductRoted);
                if res
                    B = B2;
                    rodados = [rodados k];
                end

%                 cut = 3;
%                 B = autobin(imadjust(double(B(cut+1:end-cut,cut+1:end-cut))));
%                 B = edge(B,'log');
%                 B = edge(B,'roberts');
% %                 B = imdilate(B,ones(3,1));
% %                 B = imdilate(B,ones(1,3));
% 
                B = bwareaopen(B,round(0.5*size(B,1)));
% %                 B = bwmorph(B,'close');
% 
                B = bwmorph(B,'remove');
%                 B = edge(B,'roberts');
                
%                 B = edging(B);


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
                    imshow(B)
                    str = sprintf('D.%d,N=%d',k,Nb);
                    xlabel(str);
                end
                
            end

            regions{k} = double(B);
            
            
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


function [res,B] = rotateDice(dado1,unaltered,percRotate,posDia,negDia,edgeGrad, reductRoted)
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
    [Gmag,~] = imgradient(dado1);
    edges = Gmag>edgeGrad;
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


        B = autobin(imadjust(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal))));
        
    end

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
    res = autobin(imadjust(CantoSup(dx2:end,:)));

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

%     figure(10)
%     subplot(1,2,1);
%     imshow(B)
%     hold on
%     axis on
%     axis ij
%     myAxis = axis;
%     subplot(1,2,2), hold on, axis ij, axis equal, axis(myAxis), grid on;

    [L,Nb] = bwlabel(B);

    count = 0;
    for x =1:Nb % select each boundary
        C = (L==x);
        BB = bwboundaries(C,'noholes');
        boundary = BB{1};
    
%         subplot(1,2,1);
%         plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);

        M = poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)); % from the boundary to a mask matrix (region)

        if median(boundary(:,1))>0.4*size(B,1) % invert lower card symbols
             M = rot90(rot90(poly2mask(boundary(:,2),boundary(:,1),size(B,1),size(B,2)))); 
        end
        
        
%         M = bwmorph(M,"bridge");
%         M = bwmorph(M,"fill");

        % remove all zeros rows and cols
        clean0s = M(:,any(M,1));
        clean0s = clean0s(any(clean0s,2),:);

        if nnz(clean0s)<3*size(clean0s,1)
%             subplot(1,2,2);
%             imshow(clean0s)
%             xlabel("discard")
%             pause(1)
            continue
        end
        count = count + 1;


%         subplot(1,2,2);
%         imshow(clean0s)
        
        meanO = meanO + mean(imresize(clean0s,scNaipe)~=imresize(ouro,scNaipe*size(clean0s)),'all');
        meanC = meanC + mean(imresize(clean0s,scNaipe)~=imresize(copa,scNaipe*size(clean0s)),'all');
        meanE = meanE + mean(imresize(clean0s,scNaipe)~=imresize(espada,scNaipe*size(clean0s)),'all');

%         pause(1)

    end

%     pause(1)

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

