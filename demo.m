%% DEMO
% This code runs an offline version of the decode algorithm. A dataset from
% one day of training data and online control is used as an example (Monkey G, data from 10-16-2018).

%% Decoder training
clear
addpath('DependentFunctions');

% Load training dataset
load('TrainingData.mat')

% Train decoder parameters:
% The training code finds rotational, translational, and initial direction
% dimensions. It fits a Kalman filter to the rotations for estimating the
% neural state, and fits probability distributions for move/stop and
% direction. All details exaplined in the Methods section.
[Params, Settings] = TrainRotationsDecoder(S_train, Settings);

%% Decoder testing

% Load data. Each row in the table corresponds to one trial.
load('TestingData.mat')

% Run offline version of algorithm on the data (takes 1-2 minutes).
S = TestRotationsDecoder(S_test, Params, Settings);

% The resulting table contains decoded position (decPos) and decoded
% velocity (decVel) along with some other intermediate variables.


%% Make a few plots

% Choose some random trials
nTrials = size(S,1);
trialList = randperm(nTrials,16);

% Insert target locations
S = addTargetLocations(S);

%
%
% Plot decoded position with target
%
%
figure('Renderer', 'painters', 'Position', [10 10 1000 1000])
pink = [235 0 139]/255;
for i = 1:16
    tr = trialList(i);
    % Calculate size and location of target
    pos = S.direction(tr).*(cell2mat(S.worldPosition(tr))-S.worldPosition{tr,1}(1));
    tcenter = S.direction(tr).*(S.targetLoc{tr}-S.worldPosition{tr,1}(1));
    targetOn = find(S.taskState{tr,1}==4,1);
    LLx = targetOn;
    LLy = tcenter-.5;
    wi=(length(S.taskState{tr,1}(targetOn:end)));
    hi=1;
    
    subplot(4,4,i);
    rectangle('Position',[LLx,LLy,wi,hi])
    hold on
    plot(pos,'Color',pink,'LineWidth',1.5)
    ylim([-8 8])
    set(gca,'FontSize',15)
    if i == 1
        ylabel('Position (cycles)')
        xlabel('time (ms)')
    end
end
sgtitle('Decoded position for 16 random trials, as in paper Fig 3c')

%
%
% Plot rotational dynamics
%
%
green = [88 167 106]/255;
red = [194 79 84]/255;

figure('Renderer', 'painters', 'Position', [10 10 1000 1000])
for i = 1:16
    tr = trialList(i);
    subplot(4,4,i);
    plot(S.neuralState{tr}(1,:),S.neuralState{tr}(2,:),'Color',green,'LineWidth',1.5);
    hold on
    plot(S.neuralState{tr}(3,:),S.neuralState{tr}(4,:),'Color',red,'LineWidth',1.5)
    xlim([-2 2])
    ylim([-2 2])
    set(gca,'FontSize',15)
    if i == 1
        legend({'Forward plane','Backward plane'},'Location','Best')
        ylabel('Dim 2 (a.u.)')
        xlabel('Dim 1 (a.u.)')
    end
end
sgtitle('Rotational trajectories for same 16 trials, as in paper Fig. 5')

%
%
% Plot cross correlation between decoded velocity and pedal velocity.
%
%
figure('Renderer', 'painters', 'Position', [10 10 600 600]);hold on

[xCor,lags] = cellfun(@(decVel,vel) xcorr(decVel,vel,1000,'coeff'), S.decVel,S.vel,'UniformOutput',false);
xCor = cellfun(@(xCor,lags) xCor(abs(lags) <= 600), xCor,lags,'UniformOutput',false);
xCorAvg = mean(cell2mat(xCor),1);
sem = std(cell2mat(xCor),[],1)/sqrt(size(cell2mat(xCor),1));
[val,I] = max(xCorAvg);

plot(-600:600,xCorAvg,'LineWidth',1.5)
plot(-600:600,xCorAvg + sem,'k:','LineWidth',1.5)
plot(-600:600,xCorAvg - sem,'k:','LineWidth',1.5)
plot(I-600,val,'r^','markerfacecolor',[1 0 0])
text(I-600,val+.01,sprintf('[%0.0f,%0.2f]',I-600,val),'FontSize',15,'Color','r')
set(gca,'FontSize',15)
xlabel('Lag (ms)')
ylabel('Velocity correlation')
title('Cross correlation btwn decoded and pedal velocity; as in paper Fig. 3a')

