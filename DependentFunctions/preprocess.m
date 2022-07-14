function [S, varargout] = preprocess(S, Settings, varargin)

% Calculate actual velocity.
S.vel = cellfun(@(p) [0, smooth(diff(p),10)'/Settings.Ts], S.pos,'UniformOutput',false);

% Convert spikes to doubles and apply spike filter.
S.spikes = cellfun(@double,S.spikes,'UniformOutput',false);
[S.filtSpikes,validSamples] = filterAcrossTrials(S.spikes,S.simTime,Settings);

% Restrict to valid samples (those without spike filter startup artifact).
for varName = S.Properties.VariableNames
    if iscell(S.(char(varName)))
        if all(cellfun(@(x) size(x,2), S.(char(varName))) == cellfun(@(x) size(x,2), S.filtSpikes))
            S.(char(varName)) = cellfun(@(var,validSamples) var(:,validSamples), S.(char(varName)), validSamples,'UniformOutput',false);
        end
    end
end
S.t = cellfun(@(t) 1:length(t), S.t,'UniformOutput',false);

% Remove trials for which there are no valid samples.
S = S(~cellfun('isempty',S.t),:);

if nargout == 3 && nargin == 2 % training
    
    % Remove unsuccessful trials or trials missing second target after valid sample trimming.
    Success = S.Properties.UserData.TaskStates{strcmp(S.Properties.UserData.TaskStates(:,2),'Success'),1};
    SecondTargetAppears = S.Properties.UserData.TaskStates{strcmp(S.Properties.UserData.TaskStates(:,2),'SecondTargetAppears'),1};
    S = S(cell2mat(cellfun(@(taskState) taskState(end) == Success & ismember(SecondTargetAppears,taskState), S.taskState,'UniformOutput',false)),:);

    % Restrict to most recent X trials for each 7 cycle condition.
    fwdIdx = find(S.direction == 1 & S.targetDistance == 7,Settings.MaxTrainingTrialsPerCond,'last');
    revIdx = find(S.direction == -1 & S.targetDistance == 7,Settings.MaxTrainingTrialsPerCond,'last');
    S = S(union(fwdIdx,revIdx),:);
    
end

% Determine mean and normalization factor from data or inputs.
if nargout == 3 && nargin == 2 % training
    mu = mean(cell2mat(S.filtSpikes'),2);
    range = max(cell2mat(S.filtSpikes'),[],2);
    normFactor = 1./(range+Settings.lambdaNorm);
    varargout{1} = mu;
    varargout{2} = normFactor;
else % testing
    mu = varargin{1};
    normFactor = varargin{2};
end

% Mean subtract and normalize.
S.filtSpikes = cellfun(@(filtSpikes) (filtSpikes-mu).*normFactor, S.filtSpikes,'UniformOutput',false);

% Apply high pass filter.
order = 2;
[b,a] = butter(order,Settings.HPCutoff*2*Settings.Ts,'high');
S.filtfiltSpikes = cellfun(@(filtSpikes) filter(b,a,filtSpikes,[],2),S.filtSpikes,'UniformOutput',false);

end

function [filtSpikes,validSamples] = filterAcrossTrials(spikes,simTime,Settings)

% Define filter length.
filtLen = Settings.spikeFiltWidth/(Settings.Ts*1000);

% Generate filtSpikes - use spikes from previous trials to avoid reinitializing spike filter on every trial.
filtSpikes = cell(length(spikes),1);
validSamples = cell(length(spikes),1);
for tr = 1:length(spikes)
    if tr == 1                                                 % first trial, no previous spikes to use
        filtSpikes{tr} = betaFilter(spikes{tr},filtLen,Settings.Ts);
        if length(simTime{tr}) > filtLen     
            validSamples{tr} = filtLen+1:length(simTime{tr});
        else
            validSamples{tr} = [];
        end
    elseif simTime{tr}(1)-simTime{tr-1}(end) > 1.5*Settings.Ts % gap between trials
        filtSpikes{tr} = betaFilter(spikes{tr},filtLen,Settings.Ts);
        if length(simTime{tr}) > filtLen     
            validSamples{tr} = filtLen+1:length(simTime{tr});
        else
            validSamples{tr} = [];
        end
    elseif length(simTime{tr-1}) < filtLen                     % previous trial shorter than filtLen
        filtSpikes{tr} = betaFilter([spikes{tr-1} spikes{tr}],filtLen,Settings.Ts);
        filtSpikes{tr} = filtSpikes{tr}(:,length(simTime{tr-1})+1:end);
        if length(simTime{tr}) > filtLen-length(simTime{tr-1})
            validSamples{tr} = filtLen-length(simTime{tr-1})+1:length(simTime{tr});
        else
            validSamples{tr} = [];
        end
    else                                                       % previous trial sufficiently long
        filtSpikes{tr} = betaFilter([spikes{tr-1}(:,end-filtLen+1:end) spikes{tr}],filtLen,Settings.Ts);
        filtSpikes{tr} = filtSpikes{tr}(:,filtLen+1:end);
        validSamples{tr} = 1:length(simTime{tr});
    end
end

end
