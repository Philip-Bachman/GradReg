function [ Xf ] = sac_extract( X, A, gamma, clip_knn, add_bias )
% Extract RBF features centered on the rows of A, with rbf bandwidths "gammas".
%
% For each observation, clip all but the clip_knn nearest to zero.
%
if ~exist('add_bias','var')
    add_bias = 0;
end
if ~exist('clip_knn','var')
    clip_knn = ceil(size(A,1) / 5);
end

anc_count = size(A,1);

Xd = bsxfun(@plus, sum(X.^2,2), sum(transpose(A).^2,1)) - (2*X*A');

[vals idx] = sort(Xd,2,'ascend');

% Xd = -Xd;
% for i=1:size(Xd,1),
%     Xd(i,:) = Xd(i,:) - Xd(i,idx(i,(clip_knn+1)));
% end
% Xf = max(Xd,0);
% Xf = bsxfun(@rdivide, Xf, sum(Xf,2));

for i=1:size(Xd,1),
    Xd(i,idx(i,(clip_knn+1):end)) = -1;
end

Xf = Xd;
Xf(Xf >= 0) = exp(-gamma * Xf(Xf >= 0));
Xf(Xf < 0) = 0;
Xf = bsxfun(@rdivide, Xf, (sum(Xf,2)+1e-8));

if (add_bias == 1)
    Xf = [Xf ones(size(Xf,1),1)];
end

return
end

