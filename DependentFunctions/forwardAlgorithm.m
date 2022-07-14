function alpha = forwardAlgorithm(f, T, alpha_0)

% Compute joint probability of being in a particular state and having all
% previous measurements using forward algorithm.
alpha = zeros(size(f));
alpha_prev = alpha_0; 
for t = 1:size(alpha,2)
    alpha(:,t) = f(:,t) .* (T*alpha_prev);
    alpha(:,t) = alpha(:,t)/sum(alpha(:,t));
    alpha_prev = alpha(:,t);
end