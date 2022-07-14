function S = TestRotationsDecoder(S, Params, Settings)

% Preprocessing.
Settings.MaxTrainingTrialsPerCond = inf; % preprocess all trials for testing
S = preprocess(S, Settings, Params.mu, Params.normFactor);

% Decode move/stop with HMM.
binnedSpikes = cellfun(@(spikes) sqrt(binData(spikes,Settings.binSize,'sum')/(Settings.binSize*Settings.Ts)), S.spikes,'UniformOutput',false);
f_move = cellfun(@(binnedSpikes) gaussianDensity(Params.M_hmm'*binnedSpikes, Params.move_pdf_hmm(:,1), diag(Params.move_pdf_hmm(:,2).^2)), binnedSpikes,'UniformOutput',false);
f_stop = cellfun(@(binnedSpikes) gaussianDensity(Params.M_hmm'*binnedSpikes, Params.stop_pdf_hmm(:,1), diag(Params.stop_pdf_hmm(:,2).^2)), binnedSpikes,'UniformOutput',false);
alpha_move = cellfun(@(f1,f2) forwardAlgorithm([f1;f2], Settings.moveTransMat, [0;1]), f_move, f_stop,'UniformOutput',false);
pMove = cellfun(@(alpha_move) [zeros(1,Settings.binSize-1),reshape(repmat(alpha_move(1,:),Settings.binSize,1),1,[])], alpha_move,'UniformOutput',false);
pMove = cellfun(@(pMove,vel) pMove(1:length(vel)), pMove, S.vel,'UniformOutput',false);
decMove = cellfun(@(pMove) classifyMoveStop(pMove,Settings.pMoveThresh), pMove,'UniformOutput',false);
moveOnsets = cellfun(@(decMove) find(diff(decMove)==1)+1, decMove,'UniformOutput',false);
moveOffsets = cellfun(@(decMove) find(diff(decMove)==-1), decMove,'UniformOutput',false);

% Decode direction at start of movement.
[fFwd_init,fRev_init] = cellfun(@(fSpikes,moveOnsets) ...
    initTrajectoryDensity(fSpikes,moveOnsets,Params.M_dir,Params.fwd_pdf_init,Params.rev_pdf_init,Settings.initDecodeTime), ...
    S.filtSpikes, moveOnsets,'UniformOutput',false);

% Decode steady-state direction based on areal velocities.
X = runKalman(S.filtfiltSpikes, Params.A, Params.C, Params.K_inf, Params.x0);
arealVels = cellfun(@(X) [0, X(1,1:end-1).*X(2,2:end)-X(2,1:end-1).*X(1,2:end); 0, X(3,1:end-1).*X(4,2:end)-X(4,1:end-1).*X(3,2:end)], X,'UniformOutput',false);
fFwd_av = cellfun(@(obs) gaussianDensity(obs, Params.fwd_pdf_av(:,1), Params.fwd_pdf_av(:,2:end)), arealVels,'UniformOutput',false);
fRev_av = cellfun(@(obs) gaussianDensity(obs, Params.rev_pdf_av(:,1), Params.rev_pdf_av(:,2:end)), arealVels,'UniformOutput',false);
fStop_av = cellfun(@(obs) gaussianDensity(obs, Params.stop_pdf_av(:,1), Params.stop_pdf_av(:,2:end)), arealVels,'UniformOutput',false);

% Combine into unscaled decode of velocity.
decVelUnscaled = cellfun(@(decMove,fFwd_init,fRev_init,fFwd_av,fRev_av,fStop_av,moveOnsets,moveOffsets) ...
    combineDecodes(decMove,moveOnsets,moveOffsets,fFwd_init,fRev_init,fFwd_av,fRev_av,fStop_av,...
    Params.fwd_pdf_init,Params.rev_pdf_init,Params.stop_pdf_av,Settings), ...
    decMove, fFwd_init, fRev_init, fFwd_av, fRev_av, fStop_av, moveOnsets, moveOffsets,'UniformOutput',false);

% Scale and smooth velocity.
decVel = cellfun(@(decVelUnscaled) decVelUnscaled.*(Params.velScaling(1)*(sign(decVelUnscaled) == 1)+Params.velScaling(2)*(sign(decVelUnscaled) == -1)), decVelUnscaled,'UniformOutput',false);
decVel = cellfun(@(decVel) smartAvg(decVel,Settings.decVelSmoothWidth), decVel,'UniformOutput',false);

% Integrate to get position.
decPos = cellfun(@(decVel) cumsum(decVel)*Settings.Ts, decVel,'UniformOutput',false);

% Store a few outputs.
S.decPos = decPos;
S.decVel = decVel;
S.decMove = decMove;
S.pMove = pMove;
proj = cellfun(@(binnedSpikes) Params.M_hmm'*binnedSpikes, binnedSpikes,'UniformOutput',false);
proj = cellfun(@(proj) [zeros(1,Settings.binSize-1),reshape(repmat(proj,Settings.binSize,1),1,[])], proj,'UniformOutput',false);
proj = cellfun(@(proj,vel) proj(1:length(vel)), proj, S.vel,'UniformOutput',false);
S.proj = proj;
S.neuralState = X;
S.arealVels = arealVels;
S.moveOnsets = moveOnsets;
projInit = cellfun(@(fspikes) Params.M_dir'*fspikes, S.filtSpikes,'UniformOutput',false);
S.projInit = projInit;




