function [fDens,rDens] = initTrajectoryDensity(fSpikes,moveOnsets,M_dir,fwd_pdf_init,rev_pdf_init,initDecodeTime)

% Preallocate outputs.
fDens = zeros(size(moveOnsets));
rDens = zeros(size(moveOnsets));

% For each decoded movement...
for i = 1:length(moveOnsets)
    if moveOnsets(i)+initDecodeTime(end) <= size(fSpikes,2)        
        % Calculate 3D feature describing initial trajectory and calculate densities.
        features = M_dir'*fSpikes(:,moveOnsets(i)+initDecodeTime);
        fDens(i) = gaussianDensity(features, fwd_pdf_init(:,1), fwd_pdf_init(:,2:end));
        rDens(i) = gaussianDensity(features, rev_pdf_init(:,1), rev_pdf_init(:,2:end));
    else        
        % Not enough data to calculate features.
        fDens(i) = 0;
        rDens(i) = 0;        
    end
end