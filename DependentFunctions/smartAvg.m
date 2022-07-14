function smoothDecVel = smartAvg(decVel,filtWidth)

% Preallocate output.
smoothDecVel = zeros(size(decVel));

% For each sample...
for t = 1:length(decVel)
    
    % Do not change zero decodes.
    if decVel(t) == 0
        smoothDecVel(t) = 0;
        continue
    else
        % Compute trailing average over non-zero samples in recent history.
        decVelHistory = decVel(max(1,t-filtWidth):t);
        mask = decVelHistory ~= 0;
        smoothDecVel(t) = dot(mask/sum(mask),decVelHistory);
    end
end