function [decVelUnscaled perfNums] = combineDecodes(decMove, moveOnsets, moveOffsets, fFwd_init, fRev_init, fFwd_av, fRev_av, fStop_av, fwd_pdf_init, rev_pdf_init, stop_pdf_av, Settings)

% Preallocate output.
decVelUnscaled = zeros(size(decMove));

% Define threshold densities for detecting outliers based on Mahalanobis distance thresholds.
initThreshFwd = exp(-Settings.initDecodeMahalThresh^2/2)/sqrt((2*pi)^size(fwd_pdf_init,1)*det(fwd_pdf_init(:,2:end)));
initThreshRev = exp(-Settings.initDecodeMahalThresh^2/2)/sqrt((2*pi)^size(rev_pdf_init,1)*det(rev_pdf_init(:,2:end)));
avThreshStop = exp(-Settings.avDecodeMahalThresh^2/2)/sqrt((2*pi)^size(stop_pdf_av,1)*det(stop_pdf_av(:,2:end)));

% For each decoded movement...
for i = 1:length(moveOnsets)
    
    % Create mask for samples associated with this movement.
    if any(moveOffsets > moveOnsets(i))
        mask = moveOnsets(i):moveOffsets(find(moveOffsets > moveOnsets(i),1,'first')); 
    else
        mask = moveOnsets(i):length(decMove);
    end
    
    % Ensure movement lasts longer than data accumulation epoch.
    if ismember(moveOnsets(i)+Settings.initDecodeTime,mask)
        initDecodeAccumEpoch = mask(1:Settings.initDecodeTime);
        
        % Check whether movement reaches end of initial decode range.
        if ismember(moveOnsets(i)+Settings.initDecodeTime+Settings.initDecodeDuration,mask)        
            initDecodeMask = mask(Settings.initDecodeTime+1:Settings.initDecodeTime+Settings.initDecodeDuration);
            
            % Check whether movement exceeds initial decode range.
            if length(mask) > (Settings.initDecodeTime+Settings.initDecodeDuration)
                avDecodeMask = mask(Settings.initDecodeTime+Settings.initDecodeDuration+1:end);
            else
                avDecodeMask = [];
            end
            
        else  
            initDecodeMask = mask(1:end);
            avDecodeMask = [];
        end
    else
        continue
    end
    
    % Decode initial portion of movement.
    if fFwd_init(i) > initThreshFwd || fRev_init(i) > initThreshRev
        pFwd_init = fFwd_init(i) / (fFwd_init(i) + fRev_init(i));
        decVelUnscaled(initDecodeMask) = 2*round(pFwd_init) - 1;
    else
        validIdx = fStop_av(initDecodeMask) < avThreshStop;
        pFwd_init = fFwd_av(initDecodeMask) ./ (fFwd_av(initDecodeMask) + fRev_av(initDecodeMask));
        pFwd_init(~validIdx) = 0.5+eps;
        decVelUnscaled(initDecodeMask) = 2*pFwd_init - 1;
    end
    decVelUnscaled(initDecodeAccumEpoch) = 0; % zero out initial epoch that we need to wait for to make decode
    
    % Decode remainder of movement.
    if ~isempty(avDecodeMask)
        validIdx = fStop_av(avDecodeMask) < avThreshStop;
        pFwd_steady = fFwd_av(avDecodeMask) ./ (fFwd_av(avDecodeMask) + fRev_av(avDecodeMask));
        pFwd_steady(~validIdx) = 0.5+eps;
        decVelUnscaled(avDecodeMask) = 2*pFwd_steady - 1;
    end
end

