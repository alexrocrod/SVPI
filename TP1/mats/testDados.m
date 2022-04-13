clear
close all

load("matlab.mat","regions","noiseKs")

ss = 2;
exacts = [0     7     6     0     7     7     0     0     6     8     4     5     5];

for i=1:length(noiseKs)
    figure(i)
%     dado1 = regions{noiseKs(i)}(cut+1:end-cut,cut+1:end-cut);
    cut = round(0.1*size(regions{noiseKs(i)},1)); % 0.1
    dado1 = regions{noiseKs(i)}(cut+1:end-cut,:);

    subplot(1,ss,1)
    imshow(dado1);

    subplot(1,ss,2)
    B = edge(dado1,'roberts'); % roberts
    C = bwareaopen(B,round(0.5*size(B,1))); % 0.5
    D = bwmorph(C,'close');
    imshow(D)

    [L,Nb] = bwlabel(D);
    str = sprintf("N=%d\n",Nb);  
    xlabel(str)

    if Nb ~= exacts(i)
        fprintf("Errado v3 no %d\n",i);
    end



    
end