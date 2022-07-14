function [Params, Settings] = TrainRotationsDecoder(S, Settings)

% Default settings.
DefaultSettings.Ts                       = .001;
DefaultSettings.MaxTrainingTrialsPerCond = inf; 
DefaultSettings.moveStopThresh           = .05;
DefaultSettings.moveVigorouslyThresh     = 1;
DefaultSettings.binSize                  = 10;
DefaultSettings.moveTransMat             = [.998 .0001; .002 .9999];
DefaultSettings.pMoveThresh              = [.1 .9];
DefaultSettings.spikeFiltWidth           = 275;
DefaultSettings.HPCutoff                 = 1;
DefaultSettings.lambdaNorm               = 5;
DefaultSettings.smallWeightCutoff        = .005;
DefaultSettings.firingRateCutoff         = 0.1;
DefaultSettings.initDecodeFittingEpoch   = 0:200;
DefaultSettings.initDecodeTime           = 175;
DefaultSettings.initDecodeDuration       = 200;
DefaultSettings.initDecodeMahalThresh    = 10;
DefaultSettings.avDecodeMahalThresh      = 3;
DefaultSettings.decVelSmoothWidth        = 500;

% Fill in missing user-provided settings with defaults.
missingFields = setdiff(fieldnames(DefaultSettings),fieldnames(Settings));
for i = 1:length(missingFields)
    Settings.(missingFields{i}) = DefaultSettings.(missingFields{i});
end

% Preprocess data.
[S, mu, normFactor] = preprocess(S,Settings);

% Find rotational subspace.
conds = [S.direction, S.startPos, S.targetDistance];
condList = unique(conds,'rows');
S_align = alignObservations(S);
Y = S_align.filtfiltSpikes;
Y_bar = cell(size(condList,1),1);
Y_bar_full = cell(size(Y));
for c = 1:size(condList,1)
    condMask = all(conds == condList(c,:),2);
    Y_bar{c} = mean(cat(3,Y{condMask}),3);
    Y_bar_full(condMask) = Y_bar(c);
end
Yf_bar = cell2mat(Y_bar(condList(:,1) == 1  & condList(:,3) == 7)');
Yr_bar = cell2mat(Y_bar(condList(:,1) == -1 & condList(:,3) == 7)');
M = optimizeSubspace('varDiff', Yf_bar, Yr_bar);
goodChans = any(abs(M) >= Settings.smallWeightCutoff, 2) & mu > Settings.firingRateCutoff;
M(~goodChans,:) = 0;  % zero out bad channels
M(abs(M) < Settings.smallWeightCutoff) = 0; % zero out small weights on remaining good channels to avoid extremely large values in K_inf

% Fit Kalman filter and calculate neural state.
[A, C, K_inf, x0] = fitKalman(M, Y, Y_bar, Y_bar_full, goodChans);
X = runKalman(S.filtfiltSpikes, A, C, K_inf, x0);

% Fit multivariate Gaussians to arealVels during forwards & reverse conditions.
arealVels = cellfun(@(X) [0, X(1,1:end-1).*X(2,2:end)-X(2,1:end-1).*X(1,2:end); 0, X(3,1:end-1).*X(4,2:end)-X(4,1:end-1).*X(3,2:end)], X,'UniformOutput',false);
fwdAVs = cell2mat(cellfun(@(av,v) av(:, v >  Settings.moveVigorouslyThresh)', arealVels, S.vel, 'UniformOutput',false));
revAVs = cell2mat(cellfun(@(av,v) av(:, v < -Settings.moveVigorouslyThresh)', arealVels, S.vel, 'UniformOutput',false));
stopAVs = cell2mat(cellfun(@(av,v) av(:, abs(v) < Settings.moveStopThresh)', arealVels, S.vel, 'UniformOutput',false));
fwd_pdf_av = [mean(fwdAVs,1)' cov(fwdAVs)];
rev_pdf_av = [mean(revAVs,1)' cov(revAVs)];
stop_pdf_av = [mean(stopAVs,1)' cov(stopAVs)];
fFwd_av = cellfun(@(obs) gaussianDensity(obs, fwd_pdf_av(:,1), fwd_pdf_av(:,2:end)), arealVels,'UniformOutput',false);
fRev_av = cellfun(@(obs) gaussianDensity(obs, rev_pdf_av(:,1), rev_pdf_av(:,2:end)), arealVels,'UniformOutput',false);
fStop_av = cellfun(@(obs) gaussianDensity(obs, stop_pdf_av(:,1), stop_pdf_av(:,2:end)), arealVels,'UniformOutput',false);

% Decode move/stop with an HMM.
binnedSpikes = cellfun(@(spikes) sqrt(binData(spikes,Settings.binSize,'sum')/(Settings.binSize*Settings.Ts)), S.spikes,'UniformOutput',false);
state = cellfun(@(vel,binnedSpikes) getBinnedState(vel,size(binnedSpikes,2),Settings), S.vel, binnedSpikes,'UniformOutput',false);
[M_hmm,move_pdf_hmm,stop_pdf_hmm] = fitMoveHMM(binnedSpikes, state, goodChans);
f_move = cellfun(@(binnedSpikes) gaussianDensity(M_hmm'*binnedSpikes, move_pdf_hmm(:,1), diag(move_pdf_hmm(:,2).^2)), binnedSpikes,'UniformOutput',false);
f_stop = cellfun(@(binnedSpikes) gaussianDensity(M_hmm'*binnedSpikes, stop_pdf_hmm(:,1), diag(stop_pdf_hmm(:,2).^2)), binnedSpikes,'UniformOutput',false);
alpha_move = cellfun(@(f1,f2) forwardAlgorithm([f1;f2], Settings.moveTransMat, [0;1]), f_move, f_stop,'UniformOutput',false);
pMove = cellfun(@(alpha_move) [zeros(1,Settings.binSize-1),reshape(repmat(alpha_move(1,:),Settings.binSize,1),1,[])], alpha_move,'UniformOutput',false);
pMove = cellfun(@(pMove,vel) pMove(1:length(vel)), pMove, S.vel,'UniformOutput',false);
decMove = cellfun(@(pMove) classifyMoveStop(pMove,Settings.pMoveThresh), pMove,'UniformOutput',false);
moveOnsets = cellfun(@(decMove) find(diff(decMove)==1)+1, decMove,'UniformOutput',false);
moveOffsets = cellfun(@(decMove) find(diff(decMove)==-1), decMove,'UniformOutput',false);

% Calculate top PCs for initial separation of trajectories around movement onset.
Y = cellfun(@(fSpikes,moveOnsets,decMove) moveOnsetSnippet(fSpikes,moveOnsets,decMove,Settings.initDecodeFittingEpoch), S.filtSpikes, moveOnsets, decMove,'un',false);
Yf = Y(~cellfun('isempty',Y) & S.direction == 1);
Yr = Y(~cellfun('isempty',Y) & S.direction == -1);
Yf_bar = mean(cat(3,Yf{:}),3);
Yr_bar = mean(cat(3,Yr{:}),3);
M_dir = pca([Yf_bar'; Yr_bar']);
M_dir = M_dir(:,1:3);

% Fit multivariate Gaussians to initial trajectories after movement onset.
Y = cellfun(@(fSpikes,moveOnsets,decMove) moveOnsetSnippet(fSpikes,moveOnsets,decMove,Settings.initDecodeTime), S.filtSpikes, moveOnsets, decMove,'un',false);
Yf = Y(~cellfun('isempty',Y) & S.direction == 1);
Yr = Y(~cellfun('isempty',Y) & S.direction == -1);
Xf = cellfun(@(Y_f) M_dir'*Y_f, Yf,'un',false);
Xr = cellfun(@(Y_r) M_dir'*Y_r, Yr,'un',false);
features_f = cell2mat(Xf');
features_r = cell2mat(Xr');
fwd_pdf_init = [mean(features_f,2) cov(features_f')];
rev_pdf_init = [mean(features_r,2) cov(features_r')];
[fFwd_init,fRev_init] = cellfun(@(fSpikes,moveOnsets) ...
    initTrajectoryDensity(fSpikes,moveOnsets,M_dir,fwd_pdf_init,rev_pdf_init,Settings.initDecodeTime), ...
    S.filtSpikes, moveOnsets,'UniformOutput',false);

% Combine decoded variables into unscaled estimate of velocity.
decVelUnscaled = cellfun(@(decMove,fFwd_init,fRev_init,fFwd_av,fRev_av,fStop_av,moveOnsets,moveOffsets) ...
    combineDecodes(decMove,moveOnsets,moveOffsets,fFwd_init,fRev_init,fFwd_av,fRev_av,fStop_av,...
    fwd_pdf_init,rev_pdf_init,stop_pdf_av,Settings), ...
    decMove, fFwd_init, fRev_init, fFwd_av, fRev_av, fStop_av, moveOnsets, moveOffsets,'UniformOutput',false);
decVelUnscaled = cellfun(@(decVelUnscaled) smartAvg(decVelUnscaled,Settings.decVelSmoothWidth), decVelUnscaled,'UniformOutput',false);

% Fit a scaling factor for forwards and backwards.
decPosUnscaledF = cellfun(@(decVelUnscaledF) cumsum(decVelUnscaledF)*Settings.Ts, decVelUnscaled(S.direction == 1),'UniformOutput',false);
decPosUnscaledR = cellfun(@(decVelUnscaledR) cumsum(decVelUnscaledR)*Settings.Ts, decVelUnscaled(S.direction == -1),'UniformOutput',false);
actualPosF = cellfun(@(p) p-p(1), S.pos(S.direction == 1),'UniformOutput',false);
actualPosR = cellfun(@(p) p-p(1), S.pos(S.direction == -1),'UniformOutput',false);
scaleFactorF = cell2mat(decPosUnscaledF')' \ cell2mat(actualPosF')';
scaleFactorR = cell2mat(decPosUnscaledR')' \ cell2mat(actualPosR')';

% Store parameters.
Params.mu           = mu;
Params.normFactor   = normFactor;
Params.A            = A;
Params.C            = C;
Params.K_inf        = K_inf;
Params.x0           = x0;
Params.M_hmm        = M_hmm;
Params.M_dir        = M_dir;
Params.move_pdf_hmm = move_pdf_hmm;
Params.stop_pdf_hmm = stop_pdf_hmm;
Params.fwd_pdf_av   = fwd_pdf_av;
Params.rev_pdf_av   = rev_pdf_av;
Params.stop_pdf_av  = stop_pdf_av;
Params.fwd_pdf_init = fwd_pdf_init;
Params.rev_pdf_init = rev_pdf_init;
Params.velScaling   = [scaleFactorF, scaleFactorR];


