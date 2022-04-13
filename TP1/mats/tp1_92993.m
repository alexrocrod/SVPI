% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Trabalho Pratico 1

function NumMec = tp1_92993()
    %% 

    close all
    clear al
    clc

    %% Init Vars
    NumMec = 92993;
%     NumSeq = 0;
%     NumImg = 0;
    tDom = 0;
    tDice = 0;
    tCard = 0;
    RDO = 0;
    RFO = 0;
%     tDuplas = 0;
%     PntDom = 0;
%     PntDad = 0;
%     CopOuros = 0;
%     EspPaus = 0;
%     Ouros = 0;
    StringPT = "";


    %% Open Image
%     addpath('../')
%     listaF=dir('../svpi2022_TP1_img_*.png'); %%%%%%%%%<<<<

    addpath('../sequencias/Seq160')
    listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

%     addpath('../sequencias/Seq350')
%     listaF=dir('../sequencias/Seq350/svpi2022_TP1_img_*.png');

    MaxImg = size(listaF,1);
    showplot = false;
    for idxImg = 1:MaxImg
%     idxImg = 11; showplot = true;
        
        tDuplas = 0;
        PntDom = 0;
        PntDad = 0;
        CopOuros = 0;
        EspPaus = 0;
        Ouros = 0;

        imName = listaF(idxImg).name;
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));

        A = im2double(imread(imName));
%         imshow(A)
    
        %% SubImages (provisorio)
    
        regions=vs_getsubimages(A); %extract all regions
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
        numDomsRoted = 0;

        if showplot
            figure(7)
        end

        for k=1:N
            if showplot
                subplot( SS, SS, k);
            end
            
            
            cut = 5;
            B = autobin(imadjust(regions{k}(cut+1:end-cut,cut+1:end-cut)));
%             B = autobin(imadjust(regions{k}));

            
            sx = size(B,1);
            sy = size(B,2);
            
                       
            if sx ~= sy
                rotated = false;
                % rotate to horizontal
                if sx>sy
                    B = rot90(B);
                    rotated = true;
                    sy = size(B,2);
                    sx = size(B,1);
                end

                % Clean Central Vertical Line
                perc = 4/100;
                t1 = 0.5-perc/2;
                t2 = 0.5+perc/2;
                area = round(perc*sy*sx);

                centerB = B(:,round(sy*t1):round(sy*t2));
                [gx,gy] = imgradientxy(centerB);
                vertlines = gx>0;

                
                if nnz(vertlines) > 0.3 * area % Dominos
                    B(:,round(sy*t1):round(sy*t2)) = 0; % clean
                    domKs = [domKs k];

                    if (rotated)
                        numDomsRoted=numDomsRoted+1; 
                    end

                    % clean borders               
%                     cut = 2;
%                     B = autobin(imadjust(double(B(cut+1:end-cut,cut+1:end-cut))));

                    perc = 2/100;
                    B(1:round(sy*perc),:)= 0;
                    B(end-round(sy*perc):end,:)= 0;
%                     B(:,1:round(sx*perc))= 0;
                    B(:,1:round(sx*perc*2))= 0;
                    B(:,end-round(sx*perc*2):end)= 0;

                    if showplot
                        imshow(B)
                        str = sprintf('Domino %d',k);
                        xlabel(str);
                    end

                else % cards e NOISE <<<<<

%                     B = B(10:end-10,:); % clean number and corner info

                    cardKs = [cardKs k];
                    
                    if showplot
                        imshow(B)
                        str = sprintf('Carta %d',k);
                        xlabel(str);
                    end
                    
                end

            else % Quadrados -> Dados e NOISE <<<<<
                diceKs = [diceKs k];
                

                % Perceber se estao a 45º
                c2=2;
                dado1 = autobin(imadjust(regions{k}(c2+1:end-c2,c2+1:end-c2))); 
                
                % diamond exterior
                A = strel('diamond',floor(size(dado1,1)/2)+2); %+2
                dia = A.Neighborhood;
            
                % diamond interior
                C = strel('diamond',floor(size(dado1,1)/2)-1); %-1
                diamin = C.Neighborhood;
                deltas = round((size(dia,1)-size(diamin,1))/2);
                d2 = zeros(size(dia));
                d2(deltas+1:end-deltas,deltas+1:end-deltas) = diamin;
                
                % zona esperada para a edge
                zona = dia & not(d2);
                area = nnz(zona);
                
                % edges
                [Gmag,Gdir] = imgradient(dado1);
                edges = Gmag>1;
            
                if nnz(edges(zona(1:size(edges,1),1:size(edges,1)))) > 0.2 * area %.2
            
                    rodados = [rodados k];
                    
                    % rodar
%                     A = imrotate(dado1,45);
                    A = imrotate(regions{k},45);
            
                    % reduzir imagem ao dado
                    x = size(dado1,1);
                    xmeio = round(size(A,1)/2);

                    l = floor(x/sqrt(2));
                    deltal = round(l/2)-6; % 8
            
%                     B = A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal);

                    B = autobin(imadjust(double(A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal))));
                    
                end
%                 B = autobin(imadjust(double(B(cut+1:end-cut,cut+1:end-cut))));
                    
              
                if showplot
                    imshow(B)
                    str = sprintf('Dado %d',k);
                    xlabel(str);
                end
                
            end

            regions{k} = double(B);
            
            
        end
%         tDom = length(domKs);
%         RDO = tDom - numDomsRoted; 
% 
%         tDice = length(diceKs);
%         RFO = tDice - length(rodados);
% 
%         tCard = length(cardKs);
% 
%         if showplot
%             domKs
%             diceKs
%             rodados
%             cardKs
%             fprintf("Total=%d, Dominos=%d, Dados=%d, Cartas=%d\n",N,tDom,tDice,tCard)
%         end
        

        %% Get Edges
        
        noiseKs = [];
        

        if showplot
            figure(8)
        end
        for k=1:N
            
%             B = edge(regions{k},'log');
            B = edge(regions{k},'roberts');
            B = bwareaopen(B,round(0.5*size(B,1)));
            B = bwmorph(B,'close');

            [L,Nb] = bwlabel(B);

            if showplot
                subplot( SS, SS, k);
                imshow(B)
                myAxis = axis;
                hold on, axis ij, axis equal, axis(myAxis), grid on;
                   

            for x = 1:Nb
                C = (L==x);
%                 if ( nnz(C) > 2*sx)
%                     BB = bwboundaries(C,'noholes');
%                     boundary = BB{1};
%                 
%                     plot(boundary(:,2),boundary(:,1),'r');
%                     continue
%                 end
            
                BB = bwboundaries(C,'noholes');
                boundary = BB{1};
            
                plot(boundary(:,2),boundary(:,1),'b');
           end

            str= sprintf("N=%d\n",Nb);  
            xlabel(str)

            end
            
            if ismember(k,domKs) % Dominos
                
                % Pintas de cada lado
                B1 = B(:,1:round(size(B,2)/2));
                B2 = B(:,round(size(B,2)/2):end);
                [L,Nb1] = bwlabel(B1);
                [L,Nb2] = bwlabel(B2);
                
                if (Nb1>6 || Nb2>6 || Nb==0) 
                    domKs(domKs==k) = []; % remove invalid
                    noiseKs = [noiseKs k];
                    continue
                end
                PntDom = PntDom + Nb1 + Nb2;
                if Nb1==Nb2 
                    tDuplas = tDuplas + 1;
%                     disp(k)
                end

            elseif ismember(k,diceKs) % Dados
                if (Nb>6 || Nb==0) 
                    diceKs(diceKs==k) = []; % remove invalid
                    noiseKs = [noiseKs k];
                    continue
                end
                PntDad = PntDad + Nb;
            elseif ismember(k,cardKs) % Cartas
                if (Nb>9 || Nb==0) 
                    cardKs(cardKs==k) = []; % remove invalid
                    noiseKs = [noiseKs k];
                    continue
                end
                PntCartas = [PntCartas Nb];
            end
        end
        
        PntCartas = sort(PntCartas);

        StringPT = strjoin(string(PntCartas),'');

            
        if showplot
            noiseKs
        end

        tDom = length(domKs);
        RDO = tDom - numDomsRoted; 

        tDice = length(diceKs);
        RFO = tDice - length(rodados);

        tCard = length(cardKs);

        if showplot
            domKs
            diceKs
            rodados
            cardKs
            fprintf("Total=%d, Dominos=%d, Dados=%d, Cartas=%d\n",N,tDom,tDice,tCard)
        end
        

        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, ...
            RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
        if idxImg==1
            writetable(T,'tp1_92993.txt', 'WriteVariableNames',false)
        else
            writetable(T,'tp1_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')
        end

%         pause(2)
    end

%         save


end

