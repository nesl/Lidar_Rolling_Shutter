function speckleint = model_speckles(asicshape,specklesize)

    randphasors = zeros(asicshape);
    nphasorsx = round(asicshape(1)/specklesize);
    nphasorsy = round(asicshape(2)/specklesize);

    
    for i=1:nphasorsx
        for j=1:nphasorsy
            randphasors(i,j) = exp(1i*2*pi*randn);
        end
    end

    specklefield = fft2(randphasors);
    speckleint = abs(specklefield.*conj(specklefield));
    speckleint= speckleint./sum(speckleint);
    
end  