function B = MultiRegionBin(A,N,M)

    xsplit = round(size(A,1)/M);
    ysplit = round(size(A,2)/N);
    B = zeros(size(A));
    for i=1:M
        for j=1:N
            xi = (i-1)*xsplit+1;
            xf = i*xsplit;
            if i==M
                xf=size(A,1);
            end

            yi = (j-1)*ysplit+1;
            yf = j*ysplit;
            if j==N
                yf=size(A,2);
            end

            B(xi:xf,yi:yf) = imbinarize(A(xi:xf,yi:yf));
        end
    end

%     B = medfilt2(B);
end