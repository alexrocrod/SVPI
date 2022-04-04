% SVPI
% Alexandre Rodrigues 92993
% Abril 2022
% Aula 07

exlist = {'ex1','ex2','ex3','ex4','ex5','ex6','ex7'};

if ismember('ex1',exlist)
%% Ex1
figure(1)


end
if ismember('ex2',exlist)
%% Ex2
clearvars -except exlist
figure(2)


end
if ismember('ex3',exlist)
%% Ex3
clearvars -except exlist
figure(3)


end
if ismember('ex4',exlist)
%% Ex4
clearvars -except exlist
figure(4)

end
if ismember('ex5',exlist)
%% Ex5
clearvars -except exlist
figure(5)


end
if ismember('ex6',exlist)
%% Ex6
clearvars -except exlist
figure(6)


end

if ismember('ex7',exlist)
%% Ex7
clearvars -except exlist
figure(7)

end

function B = autobinwithmask(A,M)
    B = A;
    B(M) = autobin(A(M));
end


function M = circularROI(y0,x0,ri,re,A)
    M = zeros(size(A),'logical');
    for i=1:size(A,1)
        for j=1:size(A,2)
            temp = (i-x0)^2 + (j-y0)^2;
            if ((temp >= ri^2) && (temp <= re^2))
                M(i,j) = 1;
            end
        end
    end
end