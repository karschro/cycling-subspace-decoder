function binnedData = binData(data,binSize,method)

% Calculate number of bins (truncating last bin if too few samples).
nBins = floor(size(data,2)/binSize);

% Bin data.
binnedData = zeros(size(data,1),nBins);
for n = 1:nBins
    switch method
        case 'mean'
            binnedData(:,n) = mean(double(data(:,(1:binSize)+binSize*(n-1))),2);
        case 'sum'
            binnedData(:,n) = sum(double(data(:,(1:binSize)+binSize*(n-1))),2);
        otherwise
            error('Unrecognized binning method.')
    end
end
end