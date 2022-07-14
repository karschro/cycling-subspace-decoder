function neuralState = runKalman(fSpikes,A,C,K_inf,x0)

% Preallocate output.
nTrials = length(fSpikes);
neuralState = cell(nTrials,1);

% Calculate neural state for each trial.
for tr = 1:nTrials
    
    % Define observation matrix and preallocate neural state.
    Y = fSpikes{tr};
    X = zeros(length(x0),size(Y,2));

    % Perform first measurement update.
    X(:,1) = x0 + K_inf * (Y(:,1) - C * x0);

    % Perform updates.
    for t = 1:size(X,2)-1
    
        % Perform time update.
        X(:,t+1) = A*X(:,t);
       
        % Perform measurement update.
        X(:,t+1) = X(:,t+1) + K_inf*(Y(:,t+1) - C*X(:,t+1));

    end

    % Store neural state in cell.
    neuralState{tr} = X;

end

