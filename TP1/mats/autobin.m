function Ibin= autobin(I) 
    Ibin = double(imbinarize(I));
    
    if nnz(Ibin)>0.5*(size(Ibin,1)*size(Ibin,2))
        Ibin = not(Ibin);
    end
end

