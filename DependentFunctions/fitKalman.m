function [A, C, K_inf, x0] = fitKalman(M, Y, Y_bar, Y_bar_full, goodChans)

% Compute measurement matrix based on desired subspace components.
C_red = pinv(M(goodChans,:)');
C = zeros(size(M));
C(goodChans,:) = C_red;

% Estimate neural state roughly with a least squares fit.
X = cellfun(@(Y) M'*Y, Y,'UniformOutput',false);
X_bar = cellfun(@(Y_bar) M'*Y_bar, Y_bar,'UniformOutput',false);
X_bar_full = cellfun(@(Y_bar_full) M'*Y_bar_full, Y_bar_full,'UniformOutput',false);
X_curr = cell2mat(cellfun(@(X_bar) X_bar(:,1:end-1), X_bar,'UniformOutput',false)');
X_next = cell2mat(cellfun(@(X_bar) X_bar(:,2:end), X_bar,'UniformOutput',false)');

% Compute state transition matrix.
A = X_next*X_curr' / (X_curr*X_curr');

% Compute state and measurement noise covariances.
sResidual = X_next - A*X_curr;
mResidual = cell2mat(cellfun(@(Y,X_bar_full) Y-C*X_bar_full, Y, X_bar_full,'UniformOutput',false)');
Q = sResidual*sResidual'/size(sResidual,2);
R = mResidual*mResidual'/size(mResidual,2);
R_red = R(goodChans,goodChans);

% Compute steady state error covariance by solving discrete algebraic
% Riccati equation (DARE). Only use good channels to avoid numerical
% issues.
P_inf = dare(A',C_red',Q,R_red);

% Compute steady state Kalman gain.
K_inf = zeros(size(C'));
K_inf(:,goodChans) = P_inf*C_red'*pinv(C_red*P_inf*C_red'+R_red);

% Estimate initial state.
x0 = mean(cell2mat(X'),2);

end

