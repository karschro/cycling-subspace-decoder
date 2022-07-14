function f = gaussianDensity(obs, mu, Sigma)

% Calculate density of observations with respect to multivariate Gaussian
% distribution with mean "mu" and covariance "Sigma".
f = exp(-.5*dot(obs-mu,Sigma\(obs-mu),1)) / sqrt((2*pi)^length(mu)*det(Sigma));