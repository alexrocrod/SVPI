% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Trabalho Pratico 1

function NumMec = tp1_92993()

    close all
    clear al
    clc

    %% Init Vars
    NumMec = 92993;
    NumSeq = 0;
    NumImg = 0;
    tDom = 0;
    tDice = 0;
    tCard = 0;
    RDO = 0;
    RFO = 0;
    tDuplas = 0;
    PntDom = 0;
    PntDad = 0;
    CopOuros = 0;
    EspPaus = 0;
    Ouros = 0;
    StringPT = "";


    %% Open Image
%     addpath('../')
%     listaF=dir('../svpi2022_TP1_img_*.png'); %%%%%%%%%<<<<

    addpath('../sequencias/Seq160')
    listaF=dir('../sequencias/Seq160/svpi2022_TP1_img_*.png');

%     addpath('../sequencias/Seq350')
%     listaF=dir('../sequencias/Seq350/svpi2022_TP1_img_*.png');
    fig1 = figure(1);
    fig2 = figure(2);
    fig3 = figure(3);

    numFiles = size(listaF,1);
    for idxImg = 1:numFiles
        imName = listaF(idxImg).name;
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));

        clf(fig1)
   
        A = im2double(imread(imName));
        figure(fig1)
        imshow(A)
    
        %% SubImages (provisorio)
    
        regions=vs_getsubimages(A); %extract all regions
        N=numel(regions);
        SS=ceil(sqrt(N));

        F = zeros(3,3,4);
        F(:,:,1) = [ 1  1  1;  1  -8  1;  1  1  1];
        F(:,:,2) = [ 1  2  1;  2 -12  2;  1  2  1];
        F(:,:,3) = [-1  1 -1;  1   4  1; -1  1 -1];
        F(:,:,4) = [ 1  2  3;  4 -100 5;  6  7  8];
        
        
        whiteIsol = zeros(3,3); whiteIsol(2,2)=1;
        w1 = sum(sum(whiteIsol.*F(:,:,:)));
        W = reshape(w1,1,4);
        
        blackIsol = not(whiteIsol);
        MW1 = sum(sum(blackIsol.*F(:,:,:)));
        MW = reshape(MW1,1,4);

        n=3;
        Fiso = F(:,:,n);
        
        clf(fig2)
        clf(fig3)
        figure(fig2)
        for k=1:N 
            figure(fig2)
            
%             subplot( SS, SS, k);
            imshow( regions{k} );
            B = regions{k};

            temp = filter2(Fiso,B);
            C = (temp==MW(n));
            niso = nnz(C);

%             hold on;
%             [r,c] = find(C);
%             plot(c,r,'rx') 

            D = (temp==W(n));
            niso = niso + nnz(D);

%             hold on;
%             [r,c] = find(D);
%             plot(c,r,'bx')  
            
            str = sprintf('Isolados: %d',niso);
            xlabel(str);
            
            drawnow()

            pause(0.001)

            figure(fig3)
            hold off
%             subplot(SS, SS, k)
            E = zeros(size(B));
            E(D)=1;
            E(C)=1;
            imshow(E)
            drawnow()
            pause(1)

            if niso>0
                disp(niso)
                pause(2)
            end
        end
    
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, ...
            RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
        if idxImg==1
            writetable(T,'tp1_92993.txt', 'WriteVariableNames',false)
        else
            writetable(T,'tp1_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')
        end

        %% 


        %% 
    
        pause(1)
    end




end
