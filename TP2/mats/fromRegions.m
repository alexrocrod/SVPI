close all
clear all

load("untRegions001.mat","A","regionsRGB","regions")

% regions = regionsRGB;
N = numel(regions);
SS = ceil(sqrt(N));

figure(2)
for k=1:N
    subplot(SS,SS,k)
    imshow(regions{k})
    xlabel(k)
    invM{k} = invmoments(regions{k});
end





