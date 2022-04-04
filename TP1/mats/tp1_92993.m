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
%     fig1 = figure(1);
%     fig2 = figure(2);
%     fig3 = figure(3);

    NumImg = size(listaF,1);
%     for idxImg = 1:NumImg
    idxImg = 1;
        imName = listaF(idxImg).name;
        NumSeq = str2double(imName(18:20));
        NumImg = str2double(imName(22:23));

%         clf(fig1)
   
        A = im2double(imread(imName));
%         figure(fig1)
%         imshow(A)
    
        %% SubImages (provisorio)
    
        regions=vs_getsubimages(A); %extract all regions
        N=numel(regions);
        SS=ceil(sqrt(N));
        
%         figure(2)
%         for k=1:N 
%             subplot( SS, SS, k);
%             imshow( regions{k} );
%             drawnow
%         end



        %% Filters??
%         F = zeros(3,3,4);
%         F(:,:,1) = [ 1  1  1;  1  -8  1;  1  1  1];
%         F(:,:,2) = [ 1  2  1;  2 -12  2;  1  2  1];
%         F(:,:,3) = [-1  1 -1;  1   4  1; -1  1 -1];
%         F(:,:,4) = [ 1  2  3;  4 -100 5;  6  7  8];
%         
%         
%         whiteIsol = zeros(3,3); whiteIsol(2,2)=1;
%         w1 = sum(sum(whiteIsol.*F(:,:,:)));
%         W = reshape(w1,1,4);
%         
%         blackIsol = not(whiteIsol);
%         MW1 = sum(sum(blackIsol.*F(:,:,:)));
%         MW = reshape(MW1,1,4);
% 
%         n=3;
%         Fiso = F(:,:,n);
%         
%         clf(fig2)
%         clf(fig3)
%         figure(fig2)
%         for k=1:N 
%             figure(fig2)
%             
% %             subplot( SS, SS, k);
%             imshow( regions{k} );
%             B = regions{k};
% 
%             temp = filter2(Fiso,B);
%             C = (temp==MW(n));
%             niso = nnz(C);
% 
% %             hold on;
% %             [r,c] = find(C);
% %             plot(c,r,'rx') 
% 
%             D = (temp==W(n));
%             niso = niso + nnz(D);
% 
% %             hold on;
% %             [r,c] = find(D);
% %             plot(c,r,'bx')  
%             
%             str = sprintf('Isolados: %d',niso);
%             xlabel(str);
%             
%             drawnow()
% 
%             pause(0.001)
% 
%             figure(fig3)
%             hold off
% %             subplot(SS, SS, k)
%             E = zeros(size(B));
%             E(D)=1;
%             E(C)=1;
%             imshow(E)
%             drawnow()
%             pause(1)
% 
%             if niso>0
%                 disp(niso)
%                 pause(2)
%             end
%         end

        %% Autobin
%         figure(4)
%         for k=1:N 
%             subplot( SS, SS, k);
%             B = autobin(regions{k});
%             imshow(B);
%             
%         
%             %% Get Domino Dots
%             is1 = false;
%             countDoms = 0;
%             for i=1:size(B,1)
%                 for j=1:size(B,2)
%                     if is1 == true && B(i,j)==0 
%                         is1 = false;
%                         countDoms = countDoms + 1;
%                     end
%                     if is1 == false && B(i,j)==1
%                         is1 = true;
%                     end
%                 end
%             end
% 
%             str = sprintf('Dots: %d',countDoms);
%             xlabel(str);
%             
%             drawnow
% 
%         end
        figure(7)
        domKs = [];
        for k=1:N
            subplot( SS, SS, k);
            B = autobin(regions{k});

            sx = size(B,1);
            sy = size(B,2);
            
            % rotate to horizontal
            if sx>sy
                B = rot90(B);
            end

            sy = size(B,2);
            sx = size(B,1);
            
            if sx ~= sy
                perc = 2/100;
                t1 = 0.5-perc/2;
                t2 = 0.5+perc/2;
                area = round(perc*sy*sx);
                if nnz(B(:,round(sy*t1):round(sy*t2))) > 0.7 * area
                    B(:,round(sy*t1):round(sy*t2)) = 0;
                    domKs = [domKs k];

                    B(1:round(sy*perc),:)= 0;
                    B(end-round(sy*perc):end,:)= 0;
%                     B(:,1:round(sx*perc))= 0;
                    B(:,1:round(sx*perc*2))= 0;
                    B(:,end-round(sx*perc):end)= 0;

                    imshow(B)
                    str = sprintf('Domino %d',k);
                    xlabel(str);
                    
                end
                
            end

            regions{k} = double(B);
            
            
        end
        domKs

        %% Detect squares
        figure(5)
        for k=1:N 
            if ismember(k,domKs)
                subplot( SS, SS, k);
                B = autobin(regions{k});
            
                % Get Domino Dots
    %             F = [ 1  1  1;  1  1  1;  1  1  1];
                l = 9;
                Fsq = ones(l,l);

                Floz = zeros(l,l);
                se = strel('diamond',floor(l/2));
                idx = se.Neighborhood;
                Floz(idx) = 1;

                Fcir = zeros(l,l);
                se = strel('disk',ceil(l/2));
                idx = se.Neighborhood;
                Fcir(idx) = 1;

%                 while nnz(B)>50
                    C = filter2(Fsq,B);
                    Bsq = (C==nnz(Fsq));
                    if nnz(Bsq)>0 && nnz(Bsq) < 12
                        B = Bsq;
                    else
                        C = filter2(Floz,B);
                        Bloz = (C==nnz(Floz));

                        if nnz(Bloz)>0 && nnz(Bloz) < 12
                            B = Bloz;
                        else
                            C = filter2(Fcir,B);
                            Bcir = (C==nnz(Fcir));
                            if nnz(Bloz)>0 && nnz(Bcir) < 12
                                B = Bcir;
                            end
                        end
                    end
%                 end
    
                imshow(B);
    
                str = sprintf('Dots: %d',nnz(B));
                xlabel(str);
                
                drawnow
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

        pause(2)
%     end




end

