clear
close all

load("matlab.mat","regions","diceKs")

rodados = [];
for i=1:length(diceKs)
    dado1 = regions{diceKs(i)};

    A = strel('diamond',floor(size(dado1,1)/2)+2); %+0
    dia = A.Neighborhood;

    [Gmag,Gdir] = imgradient(dado1);

    B = strel('diamond',floor(size(dado1,1)/2)-1); %-2
    diamin = B.Neighborhood;
    deltas = size(dia,1)-size(diamin,1);
    deltas = round(deltas/2);
    d2 = zeros(size(dia));
    d2(deltas+1:end-deltas,deltas+1:end-deltas) = diamin;

    zona = dia & not(d2);
    area = nnz(zona);

    edges = Gmag>1;

    if nnz(edges(zona(1:size(edges,1),1:size(edges,1)))) > 0.2 * area %.2

        rodadosK = [rodados diceKs(i)];

        
        A = imrotate(regions{diceKs(i)},45);

        x = size(regions{diceKs(i)},1);
        l = ceil(x/sqrt(2))+1;
        deltal = round(l/2)+1;
        xmeio = round(size(A,1)/2);


        regions{diceKs(i)} = A(xmeio-deltal:xmeio+deltal,xmeio-deltal:xmeio+deltal);
        
    end

end