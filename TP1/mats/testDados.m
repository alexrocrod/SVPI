clear
close all

load("matlab.mat","regions","diceKs")

for i=1:length(diceKs)
    ss = 7;
    figure(i)
    dado1 = regions{diceKs(i)};
    
    subplot(1,ss,1)
    imshow(dado1)

%     B = dado1;
% 
%     myAxis = axis;
%     hold on, axis ij, axis equal, axis(myAxis), grid on;
%     [L,Nb] = bwlabel(B);
% 
%     for x = 1:Nb
%         C = (L==x);
%     
%         BB = bwboundaries(C,'noholes');
%         boundary = BB{1};
%     
%         plot(boundary(:,2),boundary(:,1),'b');
%     end
%     str= sprintf("N=%d\n",Nb);  
%     xlabel(str)
%     
%     subplot(1,ss,2)
    A = strel('diamond',round(size(dado1,1)/2)-1);
    dia = A.Neighborhood;
%     imshow(dia)
%     
%     subplot(1,ss,3)
%     dado2 = zeros(size(dado1));
%     dado2(dia) = dado1(dia);
%     imshow(dado2)
% 
%     B = dado2;
% 
%     myAxis = axis;
%     hold on, axis ij, axis equal, axis(myAxis), grid on;
%     [L,Nb] = bwlabel(B);
% 
%     for x = 1:Nb
%         C = (L==x);
%     
%         BB = bwboundaries(C,'noholes');
%         boundary = BB{1};
%     
%         plot(boundary(:,2),boundary(:,1),'b');
%    end
% 
%     str= sprintf("N=%d\n",Nb);  
%     xlabel(str)

%     subplot(1,ss,4)
%     iguais = dado2==dado1;
%     imshow(iguais)
%     
% 
%     Niguais = nnz(iguais);
%     N = size(dado1,1)*size(dado1,2);
%     if Niguais < 0.5*N
%         fprintf("esta rodado")
%     end

    subplot(1,ss,5)
    A = strel('diamond',floor(size(dado1,1)/2));
    dia = A.Neighborhood;
    imshow(edge(dia))

%     subplot(1,ss,6)
    [Gmag,Gdir] = imgradient(dado1);
%     imshow(Gmag>3)
%     ola = Gmag>3;

%     dia2 = dia;
%     sx = size(dado1,1);
%     min = round(sx/4);
%     max = round(3*sx/4);
%     dia2(min:max,min:max)=0;

%     subplot(1,ss,7)

    B = strel('diamond',floor(size(dado1,1)/2)-2);
    diamin = B.Neighborhood;
    deltas = size(dia,1)-size(diamin,1);
    deltas = round(deltas/2);
    d2 = zeros(size(dia));
    d2(deltas+1:end-deltas,deltas+1:end-deltas) = diamin;
%     imshow(dia & not(d2))

    zona = dia & not(d2);
    area = nnz(zona);

%     ola = edge(dado1);
    ola = Gmag>1;

    if nnz(ola(zona(1:size(ola,1),1:size(ola,1)))) > 0.2 * area
        xlabel("rodado")
        fprintf("esta rodado %d\n",i)
    end

    

%     pause(2)
end