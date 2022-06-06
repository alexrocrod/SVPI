% SVPI
% Alexandre Rodrigues 92993
% Maio 2022
% Trabalho Pratico 2


function NumMec = Fasttp2_92993()
    %% DADOS

    FundoLims = zeros(9,3,2);

    FundoLims(:,:,1)=[  0.112	0.076	0.911
                        0.514	0.268	0.188
                        0.516	0	    0
                        0.614	0.132	0.019
                        0.588	0	    0
                        0.206	0.146	0.519
                        0.995	0	    0
                        0.040	0	    0
                        0.950	0	    0.089];
    
    
    FundoLims(:,:,2)=[  0.185	0.163	1
                        0.602	1	    1
                        0.569	1	    1
                        0.704	1	    0.493
                        0.929	1	    1
                        0.274	1	    1
                        0.008	0.014	0.190
                        0.185	1   	0.241
                        0.179	1   	1];
    
    minSizesFundos = [100 10 100 10 100 10 10 20 100]; 

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

    minSize = 0.1; % min nnz for aceptable boundary (percentage)

    fanKs = [9 10];


    %% Init Vars
    NumMec = 92993;
    
    %% Open Image
    addpath('../Seq29x')
    listaF=dir('../Seq29x/svpi2022_TP2_img_*1_*.png');

    MaxImg = size(listaF,1);
    
    for idxImg = 3
%     for idxImg = 1:MaxImg
        tic
        fprintf("idxImg:%d\n",idxImg);

        imName = listaF(idxImg).name;
        
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));
        
        A0 = im2double(imread(imName));

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

        % Find other subimages
%         try 
        [regionsRGB,ObjBord] = getSubImages(A0,FundoLims,minSizesFundos,minAcceptFundo,maxAcceptFundo);
%         catch
% %             fprintf(">>>>>>>>>>>>>>>>fail binarization %d\n",idxImg)
%             T = table(NumMec, NumSeq, NumImg, ObjBord, ObjPart, ObjOK, beurre, ...
%                 choco, confit, craker, fan, ginger, lotus, maria, oreo , ...
%                 palmier, parijse, sugar, wafer, zebra);
% 
%             writetable(T,'tp2_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')
%             continue
%         end

        N = numel(regionsRGB);

        %% Compare

        matchs = zeros(N,1);
        resx = zeros(N,1);
        partidas = [];

        for k=1:N
            [matchs(k),resx(k),part] = getBestMatchv2(regionsRGB{k}, AllFeatsRef, oriRefs, sizesRefs, fanKs);
            
            if part
                partidas = [partidas k];
            else
                ObjOK = ObjOK + 1;
            end
        end

        ObjPart = length(partidas);

        %% Contar
        
        for k=1:N
            if ismember(k,partidas), continue, end
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
        end
       
        
        %% Write Table Entry
        T = table(NumMec, NumSeq, NumImg, ObjBord, ObjPart, ObjOK, beurre, ...
                choco, confit, craker, fan, ginger, lotus, maria, oreo , ...
                palmier, parijse, sugar, wafer, zebra);

        writetable(T,'tp2_92993.txt', 'WriteVariableNames',false, 'WriteMode','append')

        toc
    end

end

function B = maskComplex(A0)

    B = autobin(rgb2gray(A0));
    B = bwmorph(B,"dilate",3);
    B = bwmorph(B,"close",inf);
    B = imfill(B,"holes");
    B = bwareaopen(B,300);

end

function [regionsRGB,countBord] = getSubImages(imgRef,FundosLims,minSizesFundos,minAcceptFundo,maxAcceptFundo)
    % get all subimages(regions)

    maxAccept = maxAcceptFundo;
    fundoUsed = 0;
    imgRefOld = imgRef;

    Ahsv = rgb2hsv(imgRef);
    H = Ahsv(:,:,1);
    modeH = mode(H,"all");

    if modeH < 1e-3 % Pretos
        indexes = 7;
    elseif modeH < 1.25e-1 % Preto Img 6 e 19
        indexes = 8;
    elseif modeH < 1.75e-1 % Branco
        indexes = 1;
    elseif modeH < 2.6e-1 % Verde
        indexes = 6;
    elseif modeH < 5.45e-1 % Azul Tabua
        indexes = 3;
    elseif modeH < 5.7e-1 % Azul
        indexes = 2;
    elseif modeH < 6.43e-1 % Azul Escuro
        indexes = 5;
    elseif modeH < 6.5e-1 % Azul Escuro 2
        indexes = 4;
    else % Preto Img 9 e 22
        indexes = 9;
    end

%     for i=1:length(FundosLims)
    for i=indexes
        [AnoF,mask] = removeFundoDado(imgRefOld,FundosLims(i,:,:),minSizesFundos(i),i==9);
        nnzMask = mean(mask,"all");
%         fprintf("fundo n%d, mean%.2f \n",i, nnzMask)
        if nnzMask < maxAccept && nnzMask > minAcceptFundo
            maxAccept = nnzMask;
            imgRef = AnoF;
            maskEnd = mask;
%             fprintf("Usado fundo n%d, mean%.2f \n",i, nnzMask)
            fundoUsed = i;
        end
    end

    if ~fundoUsed
%         fprintf("Not using a fundo\n")
        E = maskComplex(imgRef);
    
    else
%         fprintf("Usou fundo n%d, mean%.2f \n",fundoUsed, maxAccept)
        E = bwareaopen(maskEnd,300);
    end

    F = imclearborder(E(2:end-1,2:end-1));
    F = padarray(F,[1 1],0,"both");


    %% Bolachas Normais e Partidas
    B = F;
    
    [Bx,~,Nb] = bwboundaries(B,'noholes');
    
    sx = size(B,1);
    sy = size(B,2);
    
    count = 1;

    for k = 1:Nb % use only exterior boundaries
        boundary = Bx{k};
    
        mask = poly2mask(boundary(:,2), boundary(:,1),sx,sy);

        selectedRGB = imgRef.*repmat(mask,[1 1 3]);

        % guardar regiao
        selectedRGB = selectedRGB(:,any(mask,1),:);
        selectedRGB = selectedRGB(any(mask,2),:,:);

        regionsRGB{count} = selectedRGB;
        
        count = count + 1;
    
    end


    %% Borders

    G = imadjust(E.*not(F));

    G = bwareaopen(G,100);
    G = bwmorph(G,"close",inf);
    G = imfill(G,"holes");
    G = bwareaopen(G,300);


    [~,countBord] = bwlabel(G);

end

function [kRef,minres,part] = getBestMatchv2(img1, AllFeatsRef, oriRefs, sizesRefs, fanKs)


    part = false;
    solRefs = AllFeatsRef(end,:);

    Nref = length(oriRefs);
    partidaMean = 0;
    partidaDiffY = 0;
    minres = 1;
    kRef = 1;
    
    B = rgb2gray(img1);
    Brgb = img1;
    Bbin = B>0;
    Bbin = bwareafilt(Bbin,1);
    
    eulerN = 0;
    oriB = regionprops(Bbin,'Orientation').Orientation;

    listIrefs = 1:Nref;
    listIrefs(listIrefs==19) = [];
    listIrefs = [listIrefs 19];

    for iRef=listIrefs
        oriRef = oriRefs(iRef);
        sxRef = sizesRefs(iRef,1);

        Brgb2 = imrotate(Brgb,oriRef-oriB);
        B2 = rgb2gray(Brgb2);
        
        Brgb2 = Brgb2(:,any(B2,1),:);
        Brgb2 = Brgb2(any(B2,2),:,:);

        Brgb2 = imresize(Brgb2,[sxRef NaN]);
        B2 = rgb2gray(Brgb2);
        Bbin2 = B2>0;
        Bbin2 = bwareafilt(Bbin2, 1);
        
        Brgb2 = Brgb2.*repmat(Bbin2,[1 1 3]);
        B2 = B2.*Bbin2;

        featsIm = getFeats(Brgb2,B2,Bbin2);
        dist1 = norm(featsIm - AllFeatsRef(:,iRef));

        if dist1 < minres
            kRef = iRef;
            minres = dist1;
    
            partidaMean =  mean(Bbin2,'all')/solRefs(iRef);
            
            if iRef==19 % Smooth Edges
                windowSize = 21;
                kernel = ones(windowSize) / windowSize ^ 2;
                blurryImage = conv2(single(Bbin2), kernel, 'same');
                Bbin3 = blurryImage > 0.5; % re threshold
                eulerN = regionprops(Bbin3,'EulerNumber').EulerNumber;
            end
    
            sz1 = sort(size(Bbin2));
            szRa = sz1(1)/sz1(2);

            sz1 = sort(sizesRefs(iRef,:));
            szRef = sz1(1)/sz1(2);
            partidaDiffY = szRa/szRef;

        end
    end

    if ismember(kRef,fanKs)
        tolPartidasMean = 0.5;
        tolPartidasDiffY = 5e-2;
        tolPartidasMinVal = 2.2e-1;
    else
        tolPartidasMean = 0.7;
        tolPartidasMinVal = 3.5e-1; 
        tolPartidasDiffY = 0.095;
    end

    partidaDiffY = abs(1 - partidaDiffY);
    
    if partidaMean< tolPartidasMean || minres > tolPartidasMinVal || partidaDiffY > tolPartidasDiffY || (kRef == 19 && eulerN ~= 0)
        part = true;
    end
end

function feats = getFeats(ARGB,Agray,Abin)
    s = regionprops(Abin,'Eccentricity','Solidity');

    meanR = mean(ARGB,[1 2]);
    meanRGB = meanR(:)';

    Ahsv = rgb2hsv(ARGB);
    meanV = mean(Ahsv(:,:,3),'all');
    ola = -real(log(invmoments(Agray)))/20;
    
    feats = [meanRGB meanV ola s.Eccentricity s.Solidity]';
end

function Ibin = autobin(I)
    [counts,~] = imhist(I,16);
    T = otsuthresh(counts);
    Ibin = double(imbinarize(I,T));

    if mean(Ibin,'all') > 0.5 % always more black
        Ibin = not(Ibin);
    end
end

function [B,mask] = removeFundoDado(A,FundoLims,minS,is22)
    HSV=rgb2hsv(A); H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);

    if FundoLims(:,1,1) > FundoLims(:,1,2) 
        mask = (H >= FundoLims(:,1,1) | H <= FundoLims(:,1,2)) & (S >= FundoLims(:,2,1) & S <= FundoLims(:,2,2)) & (V >= FundoLims(:,3,1) & V <= FundoLims(:,3,2)); %add a condition for value
    else
        mask = (H >= FundoLims(:,1,1) & H <= FundoLims(:,1,2)) & (S >= FundoLims(:,2,1) & S <= FundoLims(:,2,2)) & (V >= FundoLims(:,3,1) & V <= FundoLims(:,3,2)); %add a condition for value
    end

    if ~is22
        mask=bwareaopen(mask,minS);
    
        mask=~mask; %mask for objects (negation of background)
    
        mask=bwareaopen(mask,minS); %in case we need some cleaning of "small" areas.
    
        mask = bwmorph(mask,"close",inf);
        mask = imfill(mask,"holes");
    else
        mask = bwmorph(mask,"close",inf);
        mask = bwmorph(mask,"bridge",inf);
        mask = imfill(mask,"holes");
        
        windowSize = 7;
        kernel = ones(windowSize) / windowSize ^ 2;
        blurryImage = conv2(single(mask), kernel, 'same');
        mask = blurryImage > 0.5; % Rethreshold
        
        mask = bwareaopen(mask,minS);
        mask = bwmorph(mask,"bridge",inf);
        mask = imfill(mask,"holes");
        mask = bwareaopen(mask,minS);
    end

    B = mask.*A;
end

function phi = invmoments(F)
    %INVMOMENTS Compute invariant moments of image.
    %   PHI = INVMOMENTS(F) computes the moment invariants of the image
    %   F. PHI is a seven-element row vector containing the moment
    %   invariants as defined in equations (11.3-17) through (11.3-23) of
    %   Gonzalez and Woods, Digital Image Processing, 2nd Ed.
    %
    %   F must be a 2-D, real, nonsparse, numeric or logical matrix.
    
    %   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
    %   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
    %   $Revision: 1.5 $  $Date: 2003/11/21 14:39:19 $
    
    if ~ismatrix(F) || issparse(F) || ~isreal(F) || ~(isnumeric(F) || islogical(F))
       error(['F must be a 2-D, real, nonsparse, numeric or logical ' ...
              'matrix.']);
    end
    
    F = double(F);
    phi = compute_phi(compute_eta(compute_m(F)));
end
 

function m = compute_m(F)
    [M, N] = size(F);
    [x, y] = meshgrid(1:N, 1:M);
     
    % Turn x, y, and F into column vectors to make the summations a bit
    % easier to compute in the following.
    x = x(:);
    y = y(:);
    F = F(:);
     
    % DIP equation (11.3-12)
    m.m00 = sum(F);
    % Protect against divide-by-zero warnings.
    if (m.m00 == 0)
       m.m00 = eps;
    end
    % The other central moments: 
    m.m10 = sum(x .* F);
    m.m01 = sum(y .* F);
    m.m11 = sum(x .* y .* F);
    m.m20 = sum(x.^2 .* F);
    m.m02 = sum(y.^2 .* F);
    m.m30 = sum(x.^3 .* F);
    m.m03 = sum(y.^3 .* F);
    m.m12 = sum(x .* y.^2 .* F);
    m.m21 = sum(x.^2 .* y .* F);

end

function e = compute_eta(m)
    % DIP equations (11.3-14) through (11.3-16).
    
    xbar = m.m10 / m.m00;
    ybar = m.m01 / m.m00;
    
    e.eta11 = (m.m11 - ybar*m.m10) / m.m00^2;
    e.eta20 = (m.m20 - xbar*m.m10) / m.m00^2;
    e.eta02 = (m.m02 - ybar*m.m01) / m.m00^2;
    e.eta30 = (m.m30 - 3 * xbar * m.m20 + 2 * xbar^2 * m.m10) / m.m00^2.5;
    e.eta03 = (m.m03 - 3 * ybar * m.m02 + 2 * ybar^2 * m.m01) / m.m00^2.5;
    e.eta21 = (m.m21 - 2 * xbar * m.m11 - ybar * m.m20 + ...
               2 * xbar^2 * m.m01) / m.m00^2.5;
    e.eta12 = (m.m12 - 2 * ybar * m.m11 - xbar * m.m02 + ...
               2 * ybar^2 * m.m10) / m.m00^2.5;

end

function phi = compute_phi(e)
    % DIP equations (11.3-17) through (11.3-23).
    
    phi(1) = e.eta20 + e.eta02;
    phi(2) = (e.eta20 - e.eta02)^2 + 4*e.eta11^2;
    phi(3) = (e.eta30 - 3*e.eta12)^2 + (3*e.eta21 - e.eta03)^2;
    phi(4) = (e.eta30 + e.eta12)^2 + (e.eta21 + e.eta03)^2;
    phi(5) = (e.eta30 - 3*e.eta12) * (e.eta30 + e.eta12) * ...
             ( (e.eta30 + e.eta12)^2 - 3*(e.eta21 + e.eta03)^2 ) + ...
             (3*e.eta21 - e.eta03) * (e.eta21 + e.eta03) * ...
             ( 3*(e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2 );
    phi(6) = (e.eta20 - e.eta02) * ( (e.eta30 + e.eta12)^2 - ...
                                     (e.eta21 + e.eta03)^2 ) + ...
             4 * e.eta11 * (e.eta30 + e.eta12) * (e.eta21 + e.eta03);
    phi(7) = (3*e.eta21 - e.eta03) * (e.eta30 + e.eta12) * ...
             ( (e.eta30 + e.eta12)^2 - 3*(e.eta21 + e.eta03)^2 ) + ...
             (3*e.eta12 - e.eta30) * (e.eta21 + e.eta03) * ...
             ( 3*(e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2 );
end