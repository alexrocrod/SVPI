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

    numFiles = size(listaF,1);
    for idxImg = 1:numFiles
        imName = listaF(idxImg).name;
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));
   
        A = im2double(imread(imName));
        imshow(A)
    
        %% SubImages (provisorio)
    
        regions=vs_getsubimages(A); %extract all regions
        N=numel(regions);
        SS=ceil(sqrt(N));
        for k=1:N
            subplot( SS, SS, k)
            imshow( regions{k} )
        end
    
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, tDom, tDice, tCard, RDO, ...
            RFO, tDuplas, PntDom, PntDad, CopOuros, EspPaus, Ouros, StringPT);
        writetable(T,'tp1_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')
    
        %% 


        %% 

        
    end




end

