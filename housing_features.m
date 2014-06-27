function [ FX ] = housing_features( Xtr )
% Generate a function handle to a feature extractor suited to the housing data.

feat_count = 13;

unifrm_counts = [5 0 5 0 5 5 0 0 0 0 0 5 0]; 
qntile_counts = [10 10 10 2 10 10 10 10 10 10 10 10 10];
rbf_centers = cell(1,feat_count);
rbf_stds = cell(1,feat_count);

% pause on;
for i=1:feat_count,
    Xi = sort(unique(Xtr(:,i)),'ascend');
    if (unifrm_counts(i) > 0)
        ctrs = linspace(min(Xi),max(Xi),(unifrm_counts(i)+2));
        ctrs = ctrs(2:(end-1));
        ctrs = reshape(ctrs,1,numel(ctrs));
    else
        ctrs = [];
    end
    %val_count = numel(Xi);
    %val_idx = unique(round(linspace(1,val_count,qntile_counts(i))));
    %centers = zeros(1, numel(val_idx));
    %for j=1:numel(val_idx),
    %    centers(j) = Xi(val_idx(j));
    %end
    centers = unique(quantile(Xi,linspace(0,1,qntile_counts(i))));
    centers = sort([centers ctrs],'ascend');
    stds = zeros(1, numel(centers));
    for j=1:numel(centers),
        if (j == 1)
            stds(j) = abs(centers(j) - centers(j+1));
        end
        if (j == numel(centers))
            stds(j) = abs(centers(j) - centers(j-1));
        end
        if ((j ~= 1) && (j ~= numel(centers)))
            d_l = abs(centers(j) - centers(j-1));
            d_r = abs(centers(j) - centers(j+1));
            stds(j) = max(d_l, d_r);
        end
    end
    stds = max(0.05,(stds ./ 1.0));
    rbf_centers{i} = centers;
    rbf_stds{i} = stds;
%     stem(Xi);
%     hold on;
%     for j=1:numel(centers),
%         plot(1:numel(Xi),centers(j)*ones(1,numel(Xi)),'r--');
%     end
%     pause();
%     cla; 
end

FX = @( x ) extract_features(x, rbf_centers, rbf_stds);

return
end


function [ Xf ] = extract_features(X, rbf_centers, rbf_stds)
% Extract the housing data features.

feat_count = 13;
total_feats = 0;
for i=1:feat_count,
    total_feats = total_feats + numel(rbf_centers{i});
end

Xf = zeros(size(X,1),total_feats);
start_idx = 1;
for i=1:feat_count,
    d_feat = bsxfun(@minus, X(:,i), rbf_centers{i});
    f_feat1 = exp(-bsxfun(@rdivide, d_feat.^2, rbf_stds{i}.^2));
    end_idx = start_idx + (size(f_feat1,2) - 1);
    Xf(:,start_idx:end_idx) = f_feat1;
    %f_feat2 = exp(-bsxfun(@rdivide, d_feat.^2, (2*rbf_stds{i}).^2));
    %end_idx = start_idx + (size(f_feat1,2) + size(f_feat2,2) - 1);
    %Xf(:,start_idx:end_idx) = [f_feat1 f_feat2];
    start_idx = end_idx + 1;
end
Xf(Xf < 0.05) = 0;

return
end

