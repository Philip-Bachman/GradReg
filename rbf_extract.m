function [ Xf ] = rbf_extract( X, A, gammas, add_bias, clip_eps )
% Extract RBF features centered on the rows of A, with rbf bandwidths "gammas".
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

Xf = repmat(Xd,1,numel(gammas));
Xf = bsxfun(@times, Xf, scales);
Xf = exp(-Xf);

% Add a bias term if desired
if (add_bias == 1)
    Xf = [Xf ones(size(Xf,1),1)];
end

% Do clipping as desired
Xf(abs(Xf) < clip_eps) = 0;

return
end

