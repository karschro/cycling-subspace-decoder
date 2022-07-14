function M = optimizeSubspace(method, varargin)

% Optimization settings.
warning('off', 'manopt:getHessian:approx');
warning('off', 'MATLAB:nargchk:deprecated');
options.verbosity = 0;

% Run appropriate method.
switch method
    
    case 'varDiff'
        
        % Parse inputs.
        Yf_bar = varargin{1};
        Yr_bar = varargin{2};
        
        % Define regularization parameters.
        p = 2; % matrix p-norm
        l = 0;
        
        % Initialize subspaces with PCA.
        dim = 2;
        [Vf,~] = svd(cov(Yf_bar'));
        [Vr,~] = svd(cov(Yr_bar'));
        Qf = Vf(:,1:dim);
        Qr = Vr(:,1:dim);
        
        % Optimize forward subspace.
        Z = cov(Yr_bar')-cov(Yf_bar');
        problem.M = stiefelfactory(size(Qf,1),size(Qf,2));
        problem.cost  = @(Q) trace(Q'*Z*Q) + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) 2*Z*Q         + l*(p*Q).*(abs(Q).^(p-2));
        Qf = trustregions(problem, Qf, options);
        
        % Optimize reverse subspace.
        Z = cov(Yf_bar')-cov(Yr_bar');
        problem.M = stiefelfactory(size(Qr,1),size(Qr,2));
        problem.cost  = @(Q) trace(Q'*Z*Q) + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) 2*Z*Q         + l*(p*Q).*(abs(Q).^(p-2));
        Qr = trustregions(problem, Qr, options);
        
        % Form M from top two PCs in each subspace.
        [Vf,~] = svd(cov(Yf_bar'*Qf));
        [Vr,~] = svd(cov(Yr_bar'*Qr));
        Qf = Qf*Vf;
        Qr = Qr*Vr;
        M = [Qf(:,1:2),Qr(:,1:2)];
        
    case 'varSum'
        
        % Parse inputs.
        Yf_bar = varargin{1};
        Yr_bar = varargin{2};
        
        % Define regularization parameters.
        p = 2; % matrix p-norm
        l = 0;
        
        % Initialize subspaces with PCA.
        dim = 2;
        [V,~] = svd(cov([Yf_bar'; Yr_bar']));
        Q = V(:,1:dim);
        
        % Optimize subspace.
        Z = cov(Yr_bar')+cov(Yf_bar');
        problem.M = stiefelfactory(size(Q,1),size(Q,2));
        problem.cost  = @(Q) -trace(Q'*Z*Q) + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) -2*Z*Q         + l*(p*Q).*(abs(Q).^(p-2));
        Q = trustregions(problem, Q, options);
        
        % Form M from top two PCs in each subspace.
        M = Q;
        
    case 'stdDiff'
        
        % Parse inputs.
        Yf_bar = varargin{1};
        Yr_bar = varargin{2};
        
        % Define regularization parameters.
        p = 2; % matrix p-norm
        l = 0;
        
        % Initialize subspaces with PCA.
        dim = 2;
        [Vf,~] = svd(cov(Yf_bar'));
        [Vr,~] = svd(cov(Yr_bar'));
        Qf = Vf(:,1:dim);
        Qr = Vr(:,1:dim);
        
        % Optimize forward subspace.
        C1 = cov(Yf_bar');
        C2 = cov(Yr_bar');
        problem.M = stiefelfactory(size(Qf,1),size(Qf,2));
        problem.cost  = @(Q) sqrt(trace(Q'*C2*Q)) - sqrt(trace(Q'*C1*Q))           + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) C2*Q/sqrt(trace(Q'*C2*Q)) - C1*Q/sqrt(trace(Q'*C1*Q)) + l*(p*Q).*(abs(Q).^(p-2));
        Qf = trustregions(problem, Qf, options);
        
        % Optimize reverse subspace.
        C1 = cov(Yr_bar');
        C2 = cov(Yf_bar');
        problem.M = stiefelfactory(size(Qr,1),size(Qr,2));
        problem.cost  = @(Q) sqrt(trace(Q'*C2*Q)) - sqrt(trace(Q'*C1*Q))           + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) C2*Q/sqrt(trace(Q'*C2*Q)) - C1*Q/sqrt(trace(Q'*C1*Q)) + l*(p*Q).*(abs(Q).^(p-2));
        Qr = trustregions(problem, Qr, options);
        
        % Form M from top two PCs in each subspace.
        [Vf,~] = svd(cov(Yf_bar'*Qf));
        [Vr,~] = svd(cov(Yr_bar'*Qr));
        Qf = Qf*Vf;
        Qr = Qr*Vr;
        M = [Qf(:,1:2),Qr(:,1:2)];
        
    case 'dist'
        
        % Parse inputs.
        Yf_bar = varargin{1};
        Yr_bar = varargin{2};
        
        % Define regularization parameters.
        p = 2; % matrix p-norm
        l = 0;
        
        % Initialize subspaces with PCA.
        dim = 2;
        Z = Yf_bar - Yr_bar;
        [V,~] = svd(cov(Z'));
        Q = V(:,1:dim);
        
        % Optimize subspace.
        problem.M = stiefelfactory(size(Q,1),size(Q,2));
        problem.cost  = @(Q) -norm(Q'*Z,'fro')^2 + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) -2*(Z*Z')*Q         + l*(p*Q).*(abs(Q).^(p-2));
        Q = trustregions(problem, Q, options);
        
        % Rotate Q to order dimensions by principal components.
        [V,~] = svd(cov(Z'*Q));
        M = Q*V;
        
    case 'distSingle'
        
        % Parse inputs.
        Yf = varargin{1};
        Yr = varargin{2};
        
        % Define regularization parameters.
        p = 2; % matrix p-norm
        l = 0;
        
        % Initialize subspaces with PCA.
        dim = 4;
        Yf_bar = mean(cat(3,Yf{:}),3);
        Yr_bar = mean(cat(3,Yr{:}),3);
        Z = Yf_bar - Yr_bar;
        [V,~] = svd(cov(Z'));
        Q = V(:,1:dim);
        
        % Optimize subspace.
        problem.M = stiefelfactory(size(Q,1),size(Q,2));
        problem.cost  = @(Q) -norm(Q'*Z,'fro')^2 + l*sum(sum(abs(Q).^p));
        problem.egrad = @(Q) -2*(Z*Z')*Q         + l*(p*Q).*(abs(Q).^(p-2));
        Q = trustregions(problem, Q, options);
        
    case 'arealVel'
        
        % Parse inputs.
        Yf_bar = varargin{1};
        Yr_bar = varargin{2};
        
        % Initialize subspaces with PCA.
        dim = 2;
        [Vf,~] = svd(cov(Yf_bar'));
        [Vr,~] = svd(cov(Yr_bar'));
        Qf = Vf(:,1:dim);
        Qr = Vr(:,1:dim);
        Q = [Qf Qr];
        
        % Optimize subspace.
        problem.M = stiefelfactory(size(Q,1),size(Q,2));
        problem.cost  = @(Q) avcost(Q,Yf_bar,Yr_bar);
        problem.egrad = @(Q) avcostgrad(Q,Yf_bar,Yr_bar);
        Q = trustregions(problem, Q, options);
        M = Q;
        
    case 'arealVelAbs'
        
        % Parse inputs.
        Yf_bar = varargin{1};
        Yr_bar = varargin{2};
        
        % Initialize subspaces with PCA.
        dim = 2;
        [Vf,~] = svd(cov(Yf_bar'));
        [Vr,~] = svd(cov(Yr_bar'));
        Qf = Vf(:,1:dim);
        Qr = Vr(:,1:dim);
        Q = [Qf Qr];
        
        % Optimize subspace.
        problem.M = stiefelfactory(size(Q,1),size(Q,2));
        problem.cost  = @(Q) avabscost(Q,Yf_bar,Yr_bar);
        problem.egrad = @(Q) avabscostgrad(Q,Yf_bar,Yr_bar);
        Q = trustregions(problem, Q, options);
        M = Q;
        
end

end

function cost = avcost(Q,Yf,Yr)
Xf = Q'*Yf;
Xr = Q'*Yr;
cost = mean((Xf(3,2:end).*Xf(4,1:end-1)-Xf(4,2:end).*Xf(3,1:end-1)) - (Xf(1,2:end).*Xf(2,1:end-1)-Xf(2,2:end).*Xf(1,1:end-1))) ...
    + mean((Xr(1,2:end).*Xr(2,1:end-1)-Xr(2,2:end).*Xr(1,1:end-1)) - (Xr(3,2:end).*Xr(4,1:end-1)-Xr(4,2:end).*Xr(3,1:end-1)));
end

function grad = avcostgrad(Q,Yf,Yr)

Xf = Q'*Yf;
Xr = Q'*Yr;
Tf = size(Xf,2);
Tr = size(Xr,2);
sum1 = zeros(size(Q));
sum2 = zeros(size(Q));
for t = 2:Tf
    sum1 = sum1 + [Yf(:,t) Yf(:,t-1)] * [-Xf(2,t-1) Xf(1,t-1) Xf(4,t-1) -Xf(3,t-1); Xf(2,t) -Xf(1,t) -Xf(4,t) Xf(3,t)];
end
for t = 2:Tr
    sum2 = sum2 + [Yr(:,t) Yr(:,t-1)] * [Xr(2,t-1) -Xr(1,t-1) -Xr(4,t-1) Xr(3,t-1); -Xr(2,t) Xr(1,t) Xr(4,t) -Xr(3,t)];
end
sum1 = sum1/(Tf-1);
sum2 = sum2/(Tr-1);
grad = sum1+sum2;
end

function cost = avabscost(Q,Yf,Yr)
Xf = Q'*Yf;
Xr = Q'*Yr;
cost = mean(abs(Xf(3,2:end).*Xf(4,1:end-1)-Xf(4,2:end).*Xf(3,1:end-1)) - abs(Xf(1,2:end).*Xf(2,1:end-1)-Xf(2,2:end).*Xf(1,1:end-1))) ...
    + mean(abs(Xr(1,2:end).*Xr(2,1:end-1)-Xr(2,2:end).*Xr(1,1:end-1)) - abs(Xr(3,2:end).*Xr(4,1:end-1)-Xr(4,2:end).*Xr(3,1:end-1)));
end

function grad = avabscostgrad(Q,Yf,Yr)

Xf = Q'*Yf;
Xr = Q'*Yr;
Tf = size(Xf,2);
Tr = size(Xr,2);
sum1 = zeros(size(Q));
sum2 = zeros(size(Q));
for t = 2:Tf
    sf12 = sign(Xf(1,t)*Xf(2,t-1)-Xf(2,t)*Xf(1,t-1));
    sf34 = sign(Xf(3,t)*Xf(4,t-1)-Xf(4,t)*Xf(3,t-1));
    sum1 = sum1 + [Yf(:,t-1) Yf(:,t)] * [Xf(2,t) -Xf(1,t) -Xf(4,t) Xf(3,t); -Xf(2,t-1) Xf(1,t-1) Xf(4,t-1) -Xf(3,t-1)] * (eye(4).*[sf12 sf12 sf34 sf34]);
end
for t = 2:Tr
    sr12 = sign(Xr(1,t)*Xr(2,t-1)-Xr(2,t)*Xr(1,t-1));
    sr34 = sign(Xr(3,t)*Xr(4,t-1)-Xr(4,t)*Xr(3,t-1));
    sum2 = sum2 + [Yr(:,t-1) Yr(:,t)] * [-Xr(2,t) Xr(1,t) Xr(4,t) -Xr(3,t); Xr(2,t-1) -Xr(1,t-1) -Xr(4,t-1) Xr(3,t-1)] * (eye(4).*[sr12 sr12 sr34 sr34]);
end
sum1 = sum1/(Tf-1);
sum2 = sum2/(Tr-1);
grad = sum1+sum2;
end