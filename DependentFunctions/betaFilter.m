function psth = betaFilter(spikes,filtLen,Ts)

% Define kernel function.
alpha = 3;
beta = 5;
x = linspace(0,1,filtLen+1);
y = betapdf(x,alpha,beta);

% Normalize to spikes/second.
y = y/(Ts*sum(y));

% Convolve spikes with kernel function.
psth = zeros(size(spikes));
for n = 1:size(spikes,1)
    c = conv(double(spikes(n,:)),y);
    psth(n,:) = c(1:length(spikes(n,:)));
end