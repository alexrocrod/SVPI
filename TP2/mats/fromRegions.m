close all
clear all

load("matlab.mat","A","regionsRGBRef","regions")

% regions = regionsRGB;
N = numel(regionsRGBRef);
SS = ceil(sqrt(N));

figure(2)
for k=1:N
    subplot(SS,SS,k)
    imshow(regionsRGBRef{k})
    xlabel(k)
%     invM{k} = invmoments(regions{k});
end





