clear
close all

load("matlab.mat","regions","cardKs")

ss = 2;
exacts = [0     7     6     0     7     7     0     0     6     8     4     5     5];

for i=1:length(cardKs)
    figure(i)
    card1 = regions{cardKs(i)};

    subplot(1,ss,1)
    imshow(card1);

%     subplot(1,ss,2)
%     B = edge(card1,'roberts'); % roberts
%     C = bwareaopen(B,round(0.5*size(B,1))); % 0.5
%     D = bwmorph(C,'close');
%     imshow(D)
% 
%     [L,Nb] = bwlabel(D);
%     str = sprintf("N=%d\n",Nb);  
%     xlabel(str)
% 
%     if Nb ~= exacts(i)
%         fprintf("Errado v3 no %d\n",i);
%     end



    
end