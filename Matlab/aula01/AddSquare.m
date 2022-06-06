function X = AddSquare(I,line,col)
% function to add a white square with dimensions (10,10) to image I at
% pixel (line,col)
I(line:line+10-1, col:col+10-1) = 1;
X = I;
end