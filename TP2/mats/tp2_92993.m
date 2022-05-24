% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Trabalho Pratico 2


%% Ideias

% Oreos pretas, bolacha vermelha
% encontrar zona oreo mal binarizada e tentar tratar so dessa zona
% estender a circulos as bolachas para perceber se sao partidas


%% Falhas

% falha imgidx = 2 nas bolachas iref=5
% falhas imgIdx=4 para oreos e vermelhas
% idxImg=6 varios fails e quase tudo classificado como uma ref
% idxImg=7 considera OK algumas partidas
% idxImg=8 fail total getSubImages
% idxImg=9 fail oriB
% idxImg=10 fail oriB


% De volta a B>0
% idxImg=1 perfeito
% idxImg=2 1 imagem classificada mal (a outra versao da vermelha foi partida)
% idxImg=3 parte pelo menos uma bolacha
% idxImg=4 quase bem 
% idxImg=5 falhas oreos
% idxImg=6 weird shapes do fundo
% idxImg=7 quase bem
% idxImg=8 fail regions
% idxImg=9 falha binarizacao pq tem noises > tentar binarizar depois cada regiao
% idxImg=10 algum ruido da poucos erros
% idxImg=11 quase bem
% idxImg=12 quase bem
% idxImg=13 quase bem
% idxImg=14 fundo preto, 3 partidas consideradas OK, tudo no mesmo Kref
% idxImg=15 menos 2 partidas, mais 1 OK, outra perde-se?
% idxImg=16 menos 2 partidas, mais 1 OK, outra perde-se? (same da 15)
% idxImg=17 menos 3 OK
% idxImg=18 1 partida foi OK
% idxImg=19 4 partidas a mais, 15 OK a menos
% idxImg=20 6 OKs foram partidas
% idxImg=21 fail regions
% idxImg=22 5 migalhas border, 40 partidas a mais, 13 ok a menos
% idxImg=23 2 partidas foram OK
% idxImg=24 2 OK foram partidas
% idxImg=25 quase bem, nao reconhece algumas pequenas
% idxImg=26 quase bem
% idxImg=27 quase bem
% idxImg=28 2 partidas a menos, 1 ok a mais
% idxImg=29 2 partidas a menos
% idxImg=30 1 border a menos, 3 partidas a menos



%% 


function NumMec = tp2_92993()

    close all
    clear all
    clc

    %% DATA

    FundoLims = zeros(9,3,2);

    FundoLims(:,:,1)=[  0.112	0.076	0.911
                        0.514	0.268	0.188
                        0.516	0	    0
                        0.614	0.132	0.019
                        0.588	0	    0
                        0.206	0.146	0.519
                        0.194	0	    0
                        0.995	0	    0
                        0.040	0	    0];
    
    
    FundoLims(:,:,2)=[  0.185	0.163	1
                        0.602	1	    1
                        0.569	1	    1
                        0.704	1	    0.493
                        0.929	1	    1
                        0.274	1	    1
                        1	    1	    0.181
                        0.008	0.014	0.190
                        0.185	1   	0.241];
    
    minSizesFundos = [100 10 100 10 100 10 1000 10 20]; 

    minAcceptFundo = 0.2;
    maxAcceptFundo = 0.4;

    AllFeatsRef=[   0.851757421935144	0.776536487873463	0.510835929768854	0.591310150603512	0.762683556835130	0.409440149116653	0.693367234603474	0.825033468306951	0.391962846012778	0.454997664459511	0.579605935638551	0.540806950963776	0.751610142929411	0.604613562596737	0.529719236915157	0.602079718756263	0.119561056745928	0.204589756481160	0.607132849584929	0.504846328775400	0.614112891171732	0.549581471933048	0.610448753462676	0.557517629858993	0.724518735995067	0.697333538727858	0.569078239703594	0.392564125082701
                    0.667890007872159	0.482861652772511	0.356984372402582	0.421177853857587	0.478931857103038	0.249924944070909	0.646081397231055	0.490997850692844	0.229720585522532	0.369524441163979	0.371631139502426	0.322822109036032	0.495912292542777	0.398282141139289	0.421917826657947	0.429879283026321	0.0977377819814778	0.180394632014547	0.483705903264233	0.267890692862511	0.377960132077778	0.307726861236300	0.545508228776282	0.478937478500177	0.605678857005266	0.585887744547872	0.536073990517136	0.371127465617478
                    0.364205506845117	0.0208778215908343	0.164601972821978	0.202811062696693	0.202958055260017	0.152009197461371	0.484374751797865	0.225464430066816	0.0997661237525818	0.246697117836050	0.134652099181641	0.107475171166373	0.268396318025101	0.197434933569389	0.245258986207018	0.269946425034674	0.0962885154061608	0.166975493418939	0.297087890257416	0.114772616363621	0.176805130922776	0.0624904059984650	0.416149041333989	0.330947389920888	0.407564301220585	0.383442632571171	0.514617536561157	0.358585588908595
                    0.851757421935144	0.776536487873463	0.510835929768854	0.591310150603512	0.762683556835130	0.409440149116653	0.693367536216197	0.825033468306951	0.391962846012778	0.454997664459511	0.579605935638551	0.540806950963776	0.751610142929411	0.604613562596737	0.529719236915157	0.602079718756263	0.119566412423553	0.204649366386924	0.607132849584929	0.504846328775400	0.614112891171732	0.549581471933048	0.610476888816541	0.557893661850737	0.724518735995067	0.697333538727858	0.569078239703594	0.392804524616656
                    0.0763256443439214	0.0634108204082636	0.0579644574833433	0.0647377341080383	0.0593242141782367	0.0328714018073162	0.0838656939805354	0.0630001066491051	0.0482651764354248	0.0663446521593569	0.0609389422941789	0.0544625035482754	0.0532092961546371	0.0412927230992731	0.0637984156555873	0.0680360571328213	-0.00629007846385020	0.0199744398426183	0.0613859144821097	0.0475189710684544	0.0573935434459818	0.0511254681015161	0.0763982087119962	0.0679412840482279	0.0537980047906273	0.0549540669010531	0.0747308840265686	0.0557370190027638
                    0.380640968561193	0.292221721566560	0.397809736937550	0.464076125748464	0.379061635220365	0.363563540445323	0.658483253154052	0.402754556551298	0.227977861763039	0.250262185759679	0.435664382572787	0.425783530115299	0.160470110684082	0.132751146035338	0.408379867761637	0.635484976474678	0.397282026639246	0.516719856455032	0.222439737965310	0.240934451647536	0.247381963370084	0.196840353714737	0.551673629006275	0.730636633487545	0.160228312053522	0.173831136030860	0.554908592577331	0.288800319533892
                    0.676184205749678	0.564454133544641	0.509791640325040	0.526112411548035	0.510271310163892	0.475886684439938	0.705592498340301	0.566049530775769	0.242397047728854	0.293313465870398	0.643257912003403	0.417047933624112	0.501016790405754	0.483902768785867	0.569554433061810	0.748267425843033	0.309928230586509	0.475428410662386	0.321239150123198	0.265908461927490	0.475353735794006	0.566946702340381	0.625315921725315	0.679054139388796	0.529854324513497	0.629637336463907	0.549331463082860	0.500171154006333
                    0.774485423331711	0.575924025182647	0.553018415947143	0.499776935637063	0.520986010591176	0.447159804273622	0.679846878868709	0.639863394665673	0.456117418613584	0.584768773080030	0.511561147772029	0.529975056119710	0.536703457422136	0.510879222093352	0.592723517226953	0.635817120014945	0.251164136691115	0.518116881413128	0.502708525971318	0.390916226172285	0.548849164454902	0.575155053728528	0.631073008961371	0.522096938535269	0.524582499676670	0.577882494137158	0.641989092258999	0.522250476545898
                    1.50786890918283	1.20170403219717	1.09353277177948	1.01280887374803	1.21614651043040	0.993356237071503	1.38219136878898	1.33905463595920	0.844749681369611	1.05453676890898	1.09085595570262	1.01354731795718	1.05557620505136	1.01981936241211	1.32143217860475	1.43163524677329	0.562632214584681	1.06021404382861	0.933481170303420	0.784424303686421	1.06148461204505	1.15122594904025	1.29579216570648	1.13318407736793	1.05799632417507	1.24057969695992	1.24684116972785	1.15668259202905
                    0.967105084223550	0.790715637799598	0.783016569878732	0.732340172839049	0.793275446763407	0.659666433182826	1.03962676607915	0.841474727914524	0.583616804471436	0.723464009314085	0.916991710700416	0.771118465821523	0.617011296312062	0.598500334062433	0.804881050916472	0.956086362799625	0.472613848338895	0.820660837722109	0.626232056423290	0.593131898689241	0.686473180672640	0.718102562211055	0.966639076923369	0.887742716480942	0.606971540891406	0.758086493368480	0.984743764439938	0.679725556986337
                    1.53207091815730	1.14897635420823	1.11407938374136	1.15420740770674	1.03663369566605	0.909542980272982	1.40108807840374	1.24335792379475	0.811173183601329	1.03246256207343	1.15452740597292	1.03110427377705	1.24534624028788	1.03312949009112	1.17393088131174	1.32825622897394	0.540282782055593	1.01934279820066	0.930622587255342	0.721250318346421	1.15737313832833	1.18881001478656	1.26586684141500	1.14940430245986	1.08971134733068	1.18412845479226	1.26711871442992	1.03364281937510
                    0.438462352712190	0.536236806523133	0.323747675892400	0.251193618047179	0.358487622809473	0.197933590658997	0.121668576081582	0.392015244506290	0.670468176413584	0.678318660079544	0.282720851789937	0.270975023484798	0.857733529096273	0.867797007796599	0.270851183121448	0.158553103229933	0.132635353313724	0.0985018381472082	0.712712888821645	0.607891399172778	0.668796199543946	0.748450560285473	0.143985778960761	0.127810988782563	0.857099128710967	0.839592766062695	0.163798400668826	0.485303591979776
                    0.947560044629749	0.960025141420490	0.992083177601848	0.989624900239425	0.941751922933313	0.964118512573569	0.972992320708057	0.996681337820981	0.991406348011542	0.987092261715521	0.990543420351625	0.985483754772711	0.960171198388721	0.964467005076142	0.993995976361122	0.992442916450441	0.991107012285498	0.992374038429518	0.923397449387601	0.950836172189532	0.915243751546647	0.926762265809571	0.976014171833481	0.978447683211880	1	                0.995176999438097	0.971305691821786	0.979185692541857];

    sizesRefs = [176	197; 165	198; 191	197; 189	192; 211	196; 193	196; 198	197; 216	198; 139	195 ;
                 141	195; 193	196; 190	186; 103	198; 99	196; 206	198; 200	198; 198	196; 195	195; 145	194;
                 164	196; 148	195; 122	196; 190	190; 192	190; 102	198; 108	198; 194	193; 170	197];

    oriRefs = [ 3.38919024622040	1.46852348518252	-21.2005264739338 4.36594225341164	86.0846686159345	7.35814106752544	50.9861987115747 -88.9717264840199	-0.541413482684292	1.83968022026784	-29.9285983834382	-37.3146530651279	-0.219720085201630	-0.158011705759180	76.8122729297615	65.1933090785739	-47.7495263654879	-31.8482857910492	-0.945324350175936	-10.7289926331406	-5.06439439977258	5.62219176743755	51.5618865742229	-80.5126715079722	0	-0.0159581128439894	58.1666067704519	-7.45185780985075];

    bigRefArea = 1.55e4;

    %% Init Vars
    NumMec = 92993;
    
    %% Open Image
    
%     addpath('../')

    addpath('../Seq29x')

%     listaF=dir('../Seq29x/svpi2022_TP2_img_*.png');
%     fileExact = fopen("svpi2022_tp2_seq_ALL.txt","r"); nLineExact = 0;

%     imgRef1 = im2double(imread("../svpi2022_TP2_img_001_01.png"));
    listaF=dir('../Seq29x/svpi2022_TP2_img_*1_*.png');
    fileExact = fopen("svpi2022_tp2_seq_291.txt","r"); nLineExact = 0;
    classe = 1;

%     imgRef2 = im2double(imread("../svpi2022_TP2_img_002_01.png"));
%     listaF=dir('../Seq29x/svpi2022_TP2_img_*2_*.png');
%     fileExact = fopen("svpi2022_tp2_seq_292.txt","r"); nLineExact = 0;
%     classe = 2;

    MaxImg = size(listaF,1);

    global showplot;
    showplot = false;

%     idxImg = 7; showplot = true;

%%
    [regionsRef,regionsRGBRef,~] = getRefImages(classe);
   
    for idxImg = 1:MaxImg
        tic
%     showplot = true;
%     for idxImg = 1
        fprintf("idxImg:%d\n",idxImg);

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));
        
        A0 = im2double(imread(imName));

        A = im2double(rgb2gray(imread(imName)));

%         A = medfilt2(filter2(fspecial("average",3),A));


        if showplot
            figure;
            title("A0 RGB")
            imshow(A0)
        end

        

        %% Vars
        ObjBord = 0; % numero de objs a tocar o bordo (nao para classificar)
        ObjPart = 0; % numero de objs partidos (nao para classificar)
        ObjOK = 0; % numero de objs para classificar (migalhas nao contam (5% do obj inicial))
        
        % Bolachas;
        beurre = 0;
        choco = 0;
        confit = 0;
        craker = 0;
        fan = 0;
        ginger = 0;
        lotus = 0;
        maria = 0;
        oreo = 0;
        palmier = 0;
        parijse = 0;
        sugar = 0;
        wafer = 0;
        zebra = 0;

        %% SubImages
        
%         A = F;
        minSize = 0.1; % 0.2  min nnz for aceptable boundary (percentage)
        minWidth = 0.01; % 0.04 min width of subimage (percentage)

        
%         minAreaMigalha = 0.05 * bigRefArea;
        minAreaMigalha = 0.05 * bigRefArea;

        fmaskRot = zeros(size(A));
        cutx = -3; 
        cuty = -3; 
        extend = false; %true;
        relSizes = 5; %3
        minSpare = 0.4; %0.2 da melhor na img4, melhor binarizacao das bolachas vermelhas??

        % Find other subimages
%         try 
        [regions,regionsRGB,~,ObjBord] = getSubImages(A,minSize,cutx,cuty,relSizes,minWidth,extend,fmaskRot,A0,minAreaMigalha,minSpare,FundoLims,minSizesFundos,minAcceptFundo,maxAcceptFundo);
%         catch
%             fprintf(">>>>>>>>>>>>>>>>fail binarization %d\n",idxImg)
%             T = table(NumMec, NumSeq, NumImg, ObjBord, ObjPart, ObjOK, beurre, ...
%                 choco, confit, craker, fan, ginger, lotus, maria, oreo , ...
%                 palmier, parijse, sugar, wafer, zebra);
% 
%             writetable(T,'tp2_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')
%             continue
%         end

        N = numel(regions);
        
   

        if showplot
            SS = ceil(sqrt(N));
            figure;
            title("Regions Bin")
            for k=1:N
                subplot(SS, SS, k);
                imshow(regions{k})
                xlabel(k)
            end
            figure;
            title("Regions RGB")
            for k=1:N
                subplot(SS, SS, k);
                imshow(regionsRGB{k})
                xlabel(k)
            end
        end

        %% Compare

        
        matchs = zeros(N,1);
        resx = zeros(N,1);
        partidas = [];

        fanKs = [];
        if classe==1
            fanKs = [9 10];
        elseif classe==2
            fanKs = 5;
        end

        nFeats = 13;
        for k=1:N
            
            [kRef,res,part,str] = getBestMatchv2(regionsRGB{k}, AllFeatsRef, oriRefs, sizesRefs, nFeats, fanKs);
            
            if  showplot %&& kRef == 19
                figure;
                subplot(1,2,1)
                imshow(regionsRGB{k})
                if part
                    xlabel(sprintf("part%d",k))
                else
                    xlabel(k)
                end
    
                subplot(1,2,2)
                imshow(regionsRGBRef{kRef})
                xlabel(sprintf("Kref:%d \n %s",kRef,str))

                pause(0.001)
            end

            matchs(k) = kRef;
            resx(k) = res;

            if part
%                 fprintf("Partida:%d\n",k)
                ObjPart = ObjPart + 1;
                partidas = [partidas k];
            else
                ObjOK = ObjOK + 1;
            end
        end

        %% Descartar partidas

        if showplot
            resx
            partidas
        end


        %% Contar
        
        for k=1:N
            if ismember(k,partidas), continue, end
            if classe == 1
                if matchs(k) < 3
                    beurre = beurre + 1;
                elseif matchs(k) < 5
                    choco = choco + 1;
                elseif matchs(k) < 7
                    confit = confit + 1;
                elseif matchs(k) < 9
                    craker = craker + 1;
                elseif matchs(k) < 11
                    fan = fan + 1;
                elseif matchs(k) < 13
                    ginger = ginger + 1;
                elseif matchs(k) < 15
                    lotus = lotus + 1;
                elseif matchs(k) < 17
                    maria = maria + 1;
                elseif matchs(k) < 19
                    oreo = oreo + 1;
                elseif matchs(k) < 21
                    palmier = palmier + 1;
                elseif matchs(k) < 23
                    parijse = parijse + 1;
                elseif matchs(k) < 25
                    sugar = sugar + 1;
                elseif matchs(k) < 27
                    wafer = wafer + 1;
                else
                    zebra = zebra + 1;
                end
            else
                if matchs(k) == 1
                    beurre = beurre + 1;
                elseif matchs(k) == 2
                    choco = choco + 1;
                elseif matchs(k) == 3
                    confit = confit + 1;
                elseif matchs(k) == 4
                    craker = craker + 1;
                elseif matchs(k) == 5
                    fan = fan + 1;
                elseif matchs(k) == 6
                    ginger = ginger + 1;
                elseif matchs(k) == 7
                    lotus = lotus + 1;
                elseif matchs(k) == 8
                    maria = maria + 1;
                elseif matchs(k) == 9
                    oreo = oreo + 1;
                elseif matchs(k) == 10
                    palmier = palmier + 1;
                elseif matchs(k) == 11
                    parijse = parijse + 1;
                elseif matchs(k) == 12
                    sugar = sugar + 1;
                elseif matchs(k) == 13
                    wafer = wafer + 1;
                else
                    zebra = zebra + 1;
                end
            end
        end
        
       

        %% Show vars

%         if showplot
            fprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d," + ...
                "%d,%d,%d,%d\n", NumMec, NumSeq, NumImg, ObjBord, ObjPart, ...
                ObjOK, beurre, choco, confit, craker, fan, ginger, lotus, ...
                maria, oreo , palmier, parijse, sugar, wafer, zebra);
            fprintf("Esperado:\n");
            while nLineExact < idxImg
                strExact = fgets(fileExact);
                nLineExact = nLineExact + 1;
            end
            fprintf("%s\n",strExact(2:end));            
%         end
       
        
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, ObjBord, ObjPart, ObjOK, beurre, ...
                choco, confit, craker, fan, ginger, lotus, maria, oreo , ...
                palmier, parijse, sugar, wafer, zebra);

        writetable(T,'tp2_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')

        toc
    end

        if showplot
            save
        end
    
    
end

function [regions,regionsRGB,bigRefArea] = getRefImages(classe)

    if classe == 1
        imgRef = im2double(imread("../svpi2022_TP2_img_001_01.png"));
    else
        imgRef = im2double(imread("../svpi2022_TP2_img_002_01.png"));
    end
    
    A = rgb2gray(imgRef);
    minSize = 0.1;
    relSizes = 2.5;
    minWidth = 0.05;
    fmaskPrev = zeros(size(A));
    
    A = A <1;

    B = maskNormal(A);
    
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
    
       
        selected = A.*mask;
        selectedRGB = imgRef.*repmat(mask,[1 1 3]);

        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
    
        % guardar regiao
        selectedRGB = selectedRGB(:,any(selected,1),:);
        selectedRGB = selectedRGB(any(selected,2),:,:);

        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);

        
        regions{count} = selected;

        % zona branca da palmier passa a preto

        for i = 1:size(selectedRGB,1)
            for j = 1:size(selectedRGB,2)
                if (sum(selectedRGB(i,j,:)) > 2.98)
                     % White pixel - do what you want to original image
                     selectedRGB(i,j,:) = [0 0 0]; % make it black, for example
                end
            end
        end

        regionsRGB{count} = selectedRGB;

        % compute better order for cookies
        [y, x] = ndgrid(1:size(mask, 1), 1:size(mask, 2));
        centroid = round(mean([x(logical(mask)), y(logical(mask))]));
        
        divx = int16(round(sx/10));
        divy = int16(round(sy/10));
        locals(count) = (int16(centroid(1))/divy) + (int16(centroid(2))/divx)*10 + 2;
        
        
        count = count + 1;
    
    end

    % compute better order for cookies
    [~,sortedIdx] = sort(locals);
    regionsOld = regions;
    regionsRGB_old = regionsRGB;

    for i = 1:count-1
        regions{i} = regionsOld{sortedIdx(i)};
        regionsRGB{i} = regionsRGB_old{sortedIdx(i)};
    end

    bigRefArea = inf;
    for k=1:length(regions)
        area = bwarea(regions{k});
        if area < bigRefArea
            bigRefArea = area;
        end
    end

end

function B = maskNormal(A)
    % mask for all other subimages
    
    B = edge(A,'roberts') | edge(A,'sobel');
    B = bwmorph(B,'close',inf);
end

function B = maskComplex(A0,minAreaMigalha)
    global showplot;
%     A = rgb2gray(A0);

    minAreaMigalha = round(minAreaMigalha);

%     tol = 0.2;
    rgbImg = A0;
    [idx,map] = rgb2ind(rgbImg, 0.03, 'nodither'); %// consider changing tolerance here
    m = mode( idx );
    frequentRGB = mode(map(m, : ));
    [~,freqChanel] = max(frequentRGB);
    
    Abin = A0;
    for i = 1:3
        Abin(:,:,i) = autobin(Abin(:,:,i));
        Abin(:,:,i) = bwmorph(Abin(:,:,i),"close",inf);
        Abin(:,:,i) = imfill(Abin(:,:,i),"holes");
        Abin(:,:,i) = bwareaopen(Abin(:,:,i),100);
    end
    
    Abin(:,:,freqChanel) = 0;
    
    B = sum(Abin,3)>0;
    B = bwmorph(B,"close",inf);
    B = bwmorph(B,"bridge",inf);
    B = imfill(B,"holes");
    B = bwareaopen(B,minAreaMigalha); % 1000
   
    saveB=B;

    if showplot
        figure;
        imshow(B)
        title("maskComplex 1")
    end
    
    %% clean most common
%     RGB = B.*A0;
%     
%     tol = 0.1;
    
%     A2R = RGB(:,:,1);
%     A2R(abs(A2R-frequentRGB(1))<tol) = 0;
%     A2G = RGB(:,:,2);
%     A2G(abs(A2G-frequentRGB(2))<tol) = 0;
%     A2B = RGB(:,:,3);
%     A2B(abs(A2B-frequentRGB(3))<tol) = 0;
%     
%     A2 = cat(3,A2R,A2G,A2B);
    A2 = A0;
    %%
    
    Abin = ~saveB.*A2;
    A2R = Abin(:,:,1);
    A2R(B) = frequentRGB(1);
    A2G = Abin(:,:,2);
    A2G(B) = frequentRGB(2);
    A2B = Abin(:,:,3);
    A2B(B) = frequentRGB(3);
    Abin = cat(3,A2R,A2G,A2B);
    
    for i = 1:3
        Abin(:,:,i) = autobin(Abin(:,:,i));
        Abin(:,:,i) = bwmorph(Abin(:,:,i),"close",inf);
%         Abin(:,:,i) = imfill(Abin(:,:,i),"holes");
        Abin(:,:,i) = bwareaopen(Abin(:,:,i),minAreaMigalha);%2000
    end
    
    Abin(:,:,freqChanel) = 0;
    
    B = sum(Abin,3)>0;
    B = autobinBW(double(B));
    B = bwareaopen(B,minAreaMigalha);%1000
    B = bwmorph(B,"open",inf);
    B = bwareaopen(B,minAreaMigalha);%1000
    
    
    saveB2=B|saveB;

    if showplot
        figure;
        imshow(saveB2)
        title("maskComplex 2")
    end

    %% HSV
    Abin = ~saveB2.*A2;
    A2R = Abin(:,:,1);
    A2R(saveB2) = frequentRGB(1);
    A2G = Abin(:,:,2);
    A2G(saveB2) = frequentRGB(2);
    A2B = Abin(:,:,3);
    A2B(saveB2) = frequentRGB(3);
    Abin = cat(3,A2R,A2G,A2B);
        
    Abin = rgb2hsv(Abin);
           
    B = autobin(Abin(:,:,1));
    B = bwareaopen(B,minAreaMigalha);%1000
    B = bwmorph(B,"open",inf);
    B = bwareaopen(B,minAreaMigalha);%1000
    B = bwmorph(B,"bridge",inf);
    B = imfill(B,"holes");
   
    B = (B|saveB2);

    if showplot
        figure;
        imshow(B)
        title("maskComplex 3")
    end
end

function [regions,regionsRGB,fullMask,countBord] = getSubImages(A,minSize,cutx,cuty,relSizes, ...
    minWidth,extend,fmaskPrev,imgRef,minAreaMigalha,minSparse,FundosLims,minSizesFundos,minAcceptFundo,maxAcceptFundo)
    % get all subimages(regions)

%     B = A;
%     T = adaptthresh(B);
% 
%     E = imbinarize(A,T);
% 
%     if mean(E,'all') > 0.3
%         E = not(E);
%     end

    global showplot;

    maxAccept = maxAcceptFundo;
    fundoUsed = 0;
    imgRefOld = imgRef;
    maskEnd = ones(size(A));
    for i=1:length(FundosLims)
        [AnoF,mask] = removeFundoDado(imgRefOld,FundosLims(i,:,:),minSizesFundos(i));
        nnzMask = mean(mask,"all");
        fprintf("fundo n%d, mean%.2f \n",i, nnzMask)
        if nnzMask < maxAccept && nnzMask > minAcceptFundo
            maxAccept = nnzMask;
            A = rgb2gray(AnoF);
            imgRef = AnoF;
            maskEnd = mask;
            fprintf("Usado fundo n%d, mean%.2f \n",i, nnzMask)
            fundoUsed = i;
        end
    end

%     if ~fundoUsed
    if ~fundoUsed || fundoUsed==7
        fprintf("Not using a fundo\n")
        E = maskComplex(imgRef,minAreaMigalha);
    %     E = E(2:end-1,2:end-1);
    
    %     E = bwareaopen(E,10);
        E = bwmorph(E,"bridge",inf);
        E = bwmorph(E,"close",inf);
        E = imfill(E,"holes");
        E = bwareaopen(E,100);
    
    else
        fprintf("Usou fundo n%d, mean%.2f \n",fundoUsed, maxAccept)
%         E = bwareaopen(mask,minAreaMigalha);
        E = maskEnd;
    end

    if showplot
        figure;
        imshow(E)
        title("Resultado MaskComplex")
    end

%     F = imclearborder(E);
    F = imclearborder(E(2:end-1,2:end-1));
    F = padarray(F,[1 1],0,"both");


    %% Bolachas Normais e Partidas
    B = F;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

    if showplot
        figure;
        imshow(B)
        title("Bolachas sem border")
        hold on
    end
    
%     for k = Nb+1:length(Bx) % use only interior boundaries
    for k = 1:Nb % use only exterior boundaries
        boundary = Bx{k};
    
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);
        if (nnz(mask) < minSize*sx), continue, end

        % remove already found
        if nnz(mask.*fmaskPrev), continue, end
    
        % clean all zeros cols and rows
        mask0s = mask(:,any(mask,1));
        mask0s = mask0s(any(mask0s,2),:);
        if (mean(mask0s,'all') < minSparse), continue, end % very sparse image % 0.4 ou 0.2??
    
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

        mask = bwmorph(mask,"dilate");
        mask = bwmorph(mask,"close",inf);
        mask = imfill(mask,"holes");

        if showplot
            plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
            pause(0.001)
        end
        
        selected = A.*mask;
        selectedRGB = imgRef.*repmat(mask,[1 1 3]);

        fullMask = fullMask | mask;
        fmaskPrev = fmaskPrev | mask;
    
        % guardar regiao
        selectedRGB = selectedRGB(:,any(selected,1),:);
        selectedRGB = selectedRGB(any(selected,2),:,:);

        selected = selected(:,any(selected,1));
        selected = selected(any(selected,2),:);

        if (nnz(mask) < minAreaMigalha)
%             figure(300)
%             imshow(selectedRGB)
            fprintf("migalha\n")
%             pause(2)
            continue
        end
        
        regions{count} = selected;

        % zona branca da palmier passa a preto
        for i = 1:size(selectedRGB,1)
            for j = 1:size(selectedRGB,2)
                if (sum(selectedRGB(i,j,:)) > 2.98)
                     % White pixel - do what you want to original image
                     selectedRGB(i,j,:) = [0 0 0]; % make it black, for example
                end
            end
        end

        regionsRGB{count} = selectedRGB;

        
        count = count + 1;
    
    end


    %% Borders


    G = imadjust(E.*not(F));

    G = bwareaopen(G,100);
    G = bwmorph(G,"close",inf);
    G = imfill(G,"holes");

    B = G;

    fullMask = zeros(size(B));
    
    [Bx,~,Nb] = bwboundaries(B);
    
    sx = size(B,1);
    sy = size(B,2);

    if showplot
        figure;
        title("Bolachas no border")
        imshow(B)
        hold on
    end

    countBord = 0;
    
%     for k = Nb+1:length(Bx) % use only interior boundaries
    for k = 1:Nb % use only exterior boundaries
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
    
        if showplot
            plot(boundary(:,2),boundary(:,1),'r','LineWidth',4);
            pause(0.001)
        end
      
        fmaskPrev = fmaskPrev | mask;

        if (nnz(mask) < minAreaMigalha)
            fprintf("migalha border\n")
            continue
        end

        countBord = countBord + 1;
    
    end

end

function [kRef,minres,part,str] = getBestMatchv2(img1, AllFeatsRef, oriRefs, sizesRefs, nFeats, fanKs)

    global showplot;

    part = false;
    solRefs = AllFeatsRef(end,:);

    Nref = length(oriRefs);
    partidaMean = zeros(Nref,1);
    partidaDiffY = zeros(Nref,1);
    dist = zeros(Nref,1);
    
    tolPartidasMean = 0.7;
    tolPartidasMinVal = 3.5e-1;% 3e-1;
    tolPartidasDiffY = 0.095; %0.1 falha 1 ou 2x no img3

    B = rgb2gray(img1);
    Brgb = img1;
    Bbin = B>0; % Bbin = B
%     Bbin = imbinarize(B,0.1); % 0.1
    Bbin = bwareaopen(Bbin,100); % 10

%     figure;
%     subplot(1,3,1)
%     imshow(img1)
%     subplot(1,3,2)
%     imshow(B)
%     subplot(1,3,3)
%     imshow(Bbin)
    
    eulerN = 1;
    for iRef=1:Nref
        oriRef = oriRefs(iRef);
        sxRef = sizesRefs(iRef,1);
        syRef = sizesRefs(iRef,2);

        Bbin = bwareaopen(Bbin,sxRef);
        oriB = regionprops(Bbin,'Orientation').Orientation;

        Brgb2 = imrotate(Brgb,oriRef-oriB);
        Bbin2 = imrotate(Bbin,oriRef-oriB);
        B2 = imrotate(B,oriRef-oriB);

        Bbin2 = bwareaopen(Bbin2,sxRef);
        Bbin2 = Bbin2(:,any(Bbin2,1));
        Bbin2 = Bbin2(any(Bbin2,2),:);
        
        Brgb2 = Brgb2(:,any(B2,1),:);
        Brgb2 = Brgb2(any(B2,2),:,:);

        B2 = B2(:,any(B2,1));
        B2 = B2(any(B2,2),:);

        Brgb2 = imresize(Brgb2,[sxRef NaN]);
        Bbin2 = imresize(Bbin2,[sxRef NaN]);
        B2 = imresize(B2,[sxRef NaN]);

        Bbin2 = bwmorph(Bbin2,"close",3);
%         Bbin2 = bwmorph(Bbin2,"dilate",1);
%         Bbin2 = bwmorph(Bbin2,"fill",10);
        Bbin2 = bwareaopen(Bbin2, sxRef);
        

        [~,Nb] = bwlabel(Bbin2);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb2 = Brgb;
            Bbin2 = Bbin;
            B2 = B;
        end

        featsIm = getFeats(Brgb2,B2,Bbin2,nFeats);
        dists(1) = norm(featsIm - AllFeatsRef(:,iRef));

        Brgb3 = imrotate(Brgb2,180);
        Bbin3 = imrotate(Bbin2,180);
        B3 = imrotate(B2,180);

        [~,Nb] = bwlabel(Bbin3);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb3 = Brgb;
            Bbin3 = Bbin;
            B3 = B;
        end

        featsIm2 = getFeats(Brgb3,B3,Bbin3,nFeats);
        dists(2) = norm(featsIm2 - AllFeatsRef(:,iRef));
        dist(iRef) = min(dists);
        
        if dists(2)<dists(1)
            partidaMean(iRef) =  mean(Bbin3,'all')/solRefs(iRef);
%             partidaDiffY(iRef) = size(Bbin3,2)/syRef;
            minres = dists(2);
            eulerN = regionprops(Bbin3,'EulerNumber').EulerNumber;

            sz1 = sort(size(Bbin3));
            szRa = sz1(1)/sz1(2);

            sz1 = sort(sizesRefs(iRef,:));
            szRef = sz1(1)/sz1(2);
            partidaDiffY(iRef) = szRa/szRef;
            
%             if iRef == 8 && size(Bbin3,2)/sxRef < partidaDiffY(iRef)
%             if size(Bbin3,2)/sxRef < partidaDiffY(iRef)
%                 partidaDiffY(iRef) = size(Bbin3,2)/sxRef;
%             end
        else
            partidaMean(iRef) =  mean(Bbin2,'all')/solRefs(iRef);
%             partidaDiffY(iRef) = size(Bbin2,2)/syRef;
            minres = dists(1);
            eulerN = regionprops(Bbin2,'EulerNumber').EulerNumber;

            sz1 = sort(size(Bbin2));
            szRa = sz1(1)/sz1(2);

            sz1 = sort(sizesRefs(iRef,:));
            szRef = sz1(1)/sz1(2);
            partidaDiffY(iRef) = szRa/szRef;

%             if iRef == 8 && size(Bbin2,2)/sxRef < partidaDiffY(iRef)
%             if size(Bbin2,2)/sxRef < partidaDiffY(iRef)
%                 partidaDiffY(iRef) = size(Bbin2,2)/sxRef;
%             end
        end
    end

    [minVal,kRef] = min(dist);

    if ismember(kRef,fanKs)
        tolPartidasMean = 0.5;
    end

    partidaDiffY = abs(1-partidaDiffY);
    
    if partidaMean(kRef)<tolPartidasMean || minVal > tolPartidasMinVal || partidaDiffY(kRef) > tolPartidasDiffY 
        part = true;
    end
    if kRef == 19 && eulerN~=0
%         strOla = sprintf("EulerN:%d",eulerN);
% 
%         figure;
%         sgtitle(strOla)
%         subplot(1,3,1)
%         imshow(Brgb)
%         subplot(1,3,2)
%         imshow(Brgb2)
%         subplot(1,3,3)
%         imshow(Brgb3)
% 
%         figure;
%         sgtitle(strOla)
%         subplot(1,3,1)
%         imshow(B)
%         subplot(1,3,2)
%         imshow(B2)
%         subplot(1,3,3)
%         imshow(B3)
% 
%         figure;
%         sgtitle(strOla)
%         subplot(1,3,1)
%         imshow(Bbin)
%         subplot(1,3,2)
%         imshow(Bbin2)
%         subplot(1,3,3)
%         imshow(Bbin3)

        part = true;
    end
    str = sprintf("meanRel=%.2f\n minVal=%d\n DiffY:%d",partidaMean(kRef),minVal,partidaDiffY(kRef));

    if false %showplot && kRef==8
        iRef = kRef;

        oriRef = oriRefs(iRef);
        sxRef = sizesRefs(iRef,1);

        Brgb2 = imrotate(Brgb,oriRef-oriB);
        Bbin2 = imrotate(Bbin,oriRef-oriB);
        B2 = imrotate(B,oriRef-oriB);

        Bbin2 = bwareaopen(Bbin2,sxRef);
        Bbin2 = Bbin2(:,any(Bbin2,1));
        Bbin2 = Bbin2(any(Bbin2,2),:);
        
        Brgb2 = Brgb2(:,any(B2,1),:);
        Brgb2 = Brgb2(any(B2,2),:,:);

        B2 = B2(:,any(B2,1));
        B2 = B2(any(B2,2),:);

        Brgb2 = imresize(Brgb2,[sxRef NaN]);
        Bbin2 = imresize(Bbin2,[sxRef NaN]);
        B2 = imresize(B2,[sxRef NaN]);

        Bbin2 = bwmorph(Bbin2,"close",3);
%         Bbin2 = bwmorph(Bbin2,"dilate",1);
%         Bbin2 = bwmorph(Bbin2,"fill",10);
        Bbin2 = bwareaopen(Bbin2, sxRef);
        

        [~,Nb] = bwlabel(Bbin2);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb2 = Brgb;
            Bbin2 = Bbin;
            B2 = B;
        end

        featsIm = getFeats(Brgb2,B2,Bbin2,nFeats);
        dists(1) = norm(featsIm - AllFeatsRef(:,iRef));

        Brgb3 = imrotate(Brgb2,180);
        Bbin3 = imrotate(Bbin2,180);
        B3 = imrotate(B2,180);

        [~,Nb] = bwlabel(Bbin3);
        if Nb==0 || Nb==2
            disp("fail")
            Brgb3 = Brgb;
            Bbin3 = Bbin;
            B3 = B;
        end
        
        figure;
        sgtitle("Brgb")
        subplot(1,3,1)
        imshow(Brgb)
        subplot(1,3,2)
        imshow(Brgb2)
        subplot(1,3,3)
        imshow(Brgb3)

        figure;
        sgtitle("B")
        subplot(1,3,1)
        imshow(B)
        subplot(1,3,2)
        imshow(B2)
        subplot(1,3,3)
        imshow(B3)

        figure;
        sgtitle("Bbin")
        subplot(1,3,1)
        imshow(Bbin)
        subplot(1,3,2)
        imshow(Bbin2)
        subplot(1,3,3)
        imshow(Bbin3)
    end
    
    
end

function feats = getFeats(ARGB,Agray,Abin,nFeats)
    s = regionprops(Abin,'Eccentricity','Solidity');
    meanR = mean(ARGB(:,:,1),'all');
    meanG = mean(ARGB(:,:,2),'all');
    meanB = mean(ARGB(:,:,3),'all');
    Ahsv = rgb2hsv(ARGB);
    meanV = mean(Ahsv(:,:,3),'all');
    ola = -real(log(invmoments(Agray)))/20;
    feats = [meanR meanG meanB meanV ola s.Eccentricity s.Solidity]';
end

function Ibin = autobin(I)
%     Ibin = double(imbinarize(I));

%     T = adaptthresh(I,0.2,'ForegroundPolarity','dark');
    [counts,x] = imhist(I,16);
    T = otsuthresh(counts);
    Ibin = double(imbinarize(I,T));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end

function Ibin = autobinBW(I)
    Ibin = double(imbinarize(I));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end

function [B,mask] = removeFundoDado(A,FundoLims,minS)
    HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);

    if FundoLims(:,1,1) > FundoLims(:,1,2) 
        mask = (H >= FundoLims(:,1,1) | H <= FundoLims(:,1,2)) & (S >= FundoLims(:,2,1) & S <= FundoLims(:,2,2)) & (V >= FundoLims(:,3,1) & V <= FundoLims(:,3,2)); %add a condition for value
    else
        mask = (H >= FundoLims(:,1,1) & H <= FundoLims(:,1,2)) & (S >= FundoLims(:,2,1) & S <= FundoLims(:,2,2)) & (V >= FundoLims(:,3,1) & V <= FundoLims(:,3,2)); %add a condition for value
    end

    mask=bwareaopen(mask,minS);

    mask=~mask; %mask for objects (negation of background)

    mask=bwareaopen(mask,minS); %in case we need some cleaning of "small" areas.

    %%%% Sempre??
    mask = bwmorph(mask,"close",inf);
    mask = imfill(mask,"holes");
    %%%%


    B = mask.*A;
end
