function [ FX ] = usps_features( X, Y, scale )
% Generate a function handle to a feature extractor for tests with USPS data.

class_clusts = [10]; %[16 32];
class_scales = scale * ones(1,numel(class_clusts));
feat_funcs = {};
[vals Yc] = max(Y,[],2);

for i=1:numel(class_clusts),
    fprintf('Computing %d clust features:',class_clusts(i));
    class_centers = [];
    class_dists = 0;
    for j=1:size(Y,2),
        fprintf('.');
        Xj = X(Yc==j,:);
        [C D] = clust_func(Xj, class_clusts(i));
        class_centers = [class_centers; C];
        class_dists = class_dists + sum(D);
    end
    %clust_dists = squareform(pdists(class_centers));
    %clust_dists = sort(clust_dists,2,'ascend');
    clust_scale = class_dists / size(X,1);
    feat_funcs{end+1} = ...
        @( x ) rbf_extract(x, class_centers, (class_scales(i)/clust_scale));
    fprintf('DONE\n');
end

FX = @( x ) extract_all(x, feat_funcs);

return
end

function [ centers dists ] = clust_func(X, clust_count)
% Make a feature extractor for RBFs centered on k-means cluster centers.
opts = statset('Display','off');
[IDX,C,SUMD] = kmeans(X,clust_count,'emptyaction','singleton',...
    'onlinephase','off','options',opts);
centers = C;
dists = SUMD;
return
end


function [ Xf ] = extract_all(X, feat_funcs)
% Extract the housing data features.
range = zeros(length(feat_funcs),2);
for i=1:length(feat_funcs),
    func = feat_funcs{i};
    xf = func(X(1,:));
    if (i == 1)
        range(i,1) = 1;
        range(i,2) = numel(xf);
    else
        range(i,1) = range((i-1),2) + 1;
        range(i,2) = range((i-1),2) + numel(xf);
    end
end

Xf = zeros(size(X,1),max(range(:,2)));
for i=1:length(feat_funcs),
    func = feat_funcs{i};
    Xf(:,range(i,1):range(i,2)) = func(X);
end

return
end

