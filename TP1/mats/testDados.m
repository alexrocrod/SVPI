clear
close all

load("matlab.mat","regions","diceKs")

ss = 3;
for i=1:length(diceKs)
    figure(i)
    dado1 = regions{diceKs(i)};

    subplot(1,ss,1)
    imshow(dado1);

    subplot(1,ss,2)
    B = edge(dado1,'log');
%             B = bwareaopen(B,10);
    imshow(B)
    myAxis = axis;
    hold on, axis ij, axis equal, axis(myAxis), grid on;
    [L,Nb] = bwlabel(B);

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