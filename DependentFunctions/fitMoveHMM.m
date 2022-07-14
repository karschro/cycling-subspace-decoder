function [M, move_pdf_hmm, stop_pdf_hmm] = fitMoveHMM(spikes, state, goodChans)

% Restrict to observations that are vigorously moving or definitely stopped.
spikes = cellfun(@(spikes,state) spikes(:,ismember(state,[0 1])), spikes, state, 'UniformOutput',false);
state = cellfun(@(state) state(ismember(state,[0 1])), state, 'UniformOutput',false);

% Fit linear discriminant using only good channels.
M = zeros(size(spikes{1},1),1);
goodSpikes = cellfun(@(spikes) spikes(goodChans,:), spikes,'UniformOutput',false);
discrObj = fitcdiscr(cell2mat(goodSpikes')',cell2mat(state')','DiscrimType','linear');
M(goodChans) = discrObj.Coeffs(2,1).Linear;

% Project onto discriminant and separate moving & stopped observations.
obs_move = cell2mat(cellfun(@(spikes,state) M'*spikes(:,state == 1),  spikes, state,'UniformOutput',false)');
obs_stop = cell2mat(cellfun(@(spikes,state) M'*spikes(:,state == 0),  spikes, state,'UniformOutput',false)');

% Fit parameters of multivariate Gaussians.
move_pdf_hmm = [mean(obs_move,2) std(obs_move,[],2)];
stop_pdf_hmm = [mean(obs_stop,2) std(obs_stop,[],2)];