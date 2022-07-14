function S = alignObservations(S)

% Calculate time points to interpolate over and number of samples to use per condition.
conds = [S.direction, S.startPos, S.targetDistance];
condList = unique(conds,'rows');
tAll = cell(size(S,1),1);
nSamples = zeros(size(S,1),1);
for c = 1:size(condList,1)
    
    % Restrict to trials for this condition.
    trialIdx = find(all(conds == condList(c,:),2));
    p = S.pos(trialIdx);
    t = cellfun(@(p) 1:length(p), p,'UniformOutput',false);
    dir = condList(c,1);
    tDist = condList(c,3);
    
    % Remove position offset.
    targPos = round(cellfun(@(p) p(end),p)*2)/2;
    initPos = targPos - dir.*tDist;
    p = cellfun(@(pos,initPos) pos-initPos, p, num2cell(initPos),'UniformOutput',false);
    S.pos(trialIdx) = p;
    
    % Fit line to middle cycles and use this line to extract full cycles.
    tMiddle = cellfun(@(pos) find(pos*dir > .25 & pos*dir < tDist-.25), p,'UniformOutput',false);
    beta = cellfun(@(t,p) [ones(length(t),1),t'] \ p(t)', tMiddle, p,'UniformOutput',false);
    tAll(trialIdx) = cellfun(@(beta,t) find((beta(1)+beta(2)*t)*dir > 0,1,'first'):find((beta(1)+beta(2)*t)*dir < tDist,1,'last'), beta, t,'UniformOutput',false);
    
    % Determine number of samples to interpolate using average length.
    nSamples(trialIdx) = round(mean(cell2mat(cellfun(@(t) t(end)-t(1), tAll(trialIdx),'UniformOutput',false))));
    
end

% Get list of fields that need interpolating.
varNames = S.Properties.VariableNames;
varNames = varNames(~strcmp(varNames,'t'));
interpVars = cell(0);
for i = 1:length(varNames)
    var = S.(varNames{i});
    if iscell(var)
        if ~any(cell2mat(cellfun(@(x) isa(x,'uint8'), var,'UniformOutput',false)))
            if all(cell2mat(cellfun(@(x,y) size(x,2) == size(y,2) | size(x,2) == size(y,1), S.t, var,'UniformOutput',false)))
                interpVars = [interpVars,varNames{i}]; %#ok
            end
        end
    end
end

% Interpolate variables.
for i = 1:length(interpVars)
    varAll = cellfun(@(t,var) var(:,t), tAll, S.(interpVars{i}),'UniformOutput',false);
    S.(interpVars{i}) = cellfun(@(tAll,varAll,nSamples) interp1(tAll,varAll',linspace(tAll(1),tAll(end),nSamples))', tAll, varAll, num2cell(nSamples),'UniformOutput',false);
    if size((S.(interpVars{i}){1}),1) > 1 && size((S.(interpVars{i}){1}),2) == 1
        S.(interpVars{i}) = cellfun(@(x) x', S.(interpVars{i}),'UniformOutput',false);
    end
end
S.t = cellfun(@(nSamples) 1:nSamples, num2cell(nSamples),'UniformOutput',false);

