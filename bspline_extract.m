function [ Xf ] = bspline_extract( X, A, k, gammas, add_bias, clip_eps )
% Extract kth-rder B-spline kernel features centered on the rows of A.
% Evaluate kernels for each bandwidth in "gammas".
%
if ~exist('add_bias','var')
    add_bias = 0;
end
if ~exist('clip_eps','var')
    clip_eps = 1e-6;
end

anc_count = size(A,1);
gam_count = numel(gammas);

% Compute distances, and correct for numerical leakage
Xd = bsxfun(@plus, sum(X.^2,2), sum(transpose(A).^2,1)) - (2*X*A');
Xd = max(Xd,0);

scales = [];
for i=1:gam_count,
    scales = [scales gammas(i)*ones(1,anc_count)];
end

K_MAX = bspline_zero(0,k);

Xf = repmat(sqrt(Xd),1,numel(gammas));
Xf = bsxfun(@times, Xf, scales);
Xf = bspline_zero(Xf, k);
Xf = Xf ./ K_MAX;

% Add a bias term if desired
if (add_bias == 1)
    Xf = [Xf ones(size(Xf,1),1)];
end

% Do clipping as desired
Xf(abs(Xf) < clip_eps) = 0;

return
end

function [ bs_vals ] = bspline_zero( X, k )
% Compute B-spline weight, for a kth-order B-spline centered at 0.
%
bs_vals = zeros(size(X));
for r=0:(2*(k+1)),
  bs_vals = bs_vals + ((-1)^r * binomial((2 * (k + 1)), r) * ...
      max(0, (X + (k + 1) - r)).^((2 * k) + 1));
end
return
end

function b = binomial(n,k)
%BINOMIAL compute binomial coefficient
%
%  Usage: b = binomial(n,k)
%
%  Parameters:     ( n )
%              b = (   )
%                  ( k )
%
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)

b = prod(k+1:n)/prod(1:(n-k));

return
end