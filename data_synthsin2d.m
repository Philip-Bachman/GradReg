function [ Xtr Ytr Xte Yte Xs ] = data_synthsin2d(tr_count, te_count, noise_std)
% Generate some random sinish data
%

s_count = (tr_count + te_count);

cycle_count = 2.0; % 3.0;
cycle_scale = 4.0; % 2.5;

%X = rand(s_count,2);
[X1 X2] = meshgrid(linspace(0,1,round(8*sqrt(s_count))));
X = [X1(:) X2(:)];
idx = randsample(size(X,1),s_count);
X = X(idx,:);
X(1,:) = [0 0];
X(2,:) = [0 1];
X(3,:) = [1 0];
X(4,:) = [1 1];
X(5,:) = [0 0.5];
X(6,:) = [1 0.5];

Y = sin(2*pi*cycle_count*X(:,1));
Y = Y .* (1 + ((X(:,1) ./ max(X(:,1)))));

cycle_loc = cycle_count * X(:,1);
X1 = cycle_scale.^cycle_loc;
X1 = X1 - 1;
X2 = (X(:,2) - 0.5) .* ((X1+1e-3) ./ (X(:,1)+1e-3)); 

X(:,1) = X1;
X(:,2) = X2;
X = X * (8 / max(X(:,1)));

X_scale = max(X) - min(X);
X_min = min(X);
Xs = bsxfun(@plus, bsxfun(@times, rand(10000,2), X_scale), X_min);

Xtr = X(1:tr_count,:);
Ytr = Y(1:tr_count,:);
noise_scales = 1 + (Xtr(:,1) ./ max(Xtr(:,1)));
Ytr = Ytr + (noise_std * (randn(size(Ytr)) .* noise_scales));

Xte = X((tr_count+1):end,:);
Yte = Y((tr_count+1):end,:);

% Xtr = 0.1 * randn(5,2);
% Ytr = 0.2*ones(5,1);
% circ = randn(15,2);
% circ = 3 * bsxfun(@rdivide, circ, max(1e-8,sqrt(sum(circ.^2,2))));
% Xtr = [Xtr; circ];
% Ytr = [Ytr; -0.2*ones(size(circ,1),1)];
% Xte = 2.5 * randn(5000,2);
% Yte = randn(size(Xte,1),1);
% 
% Xs = randn(10000,2);

return
end

function [ Xs ] = skewed_uniform(s_count, skew_fact)
% Sample via rejection sampling from a "skewed" uniform distribution, which has
% a PDF that decreases linearly from p(0) to p(1), s.t. p(0)/p(1) is skew_fact.
%
% The skewed uniform PDF is a trapezoid with base length 1 and parallel sides
% whose lengths L0 and L1 are set such that (L0/L1 == skew_fact) and the area,
% given by (L1 + ((L0 - L1) / 2)), is 1. (i.e. it's a valid PDF). 
%
% L0/L1 = skew_fact
% L0 = L1 * skew_fact
% L1 = L0 / skew_fact
% L1 + ((L0 - L1) / 2) = 1
% L1 + (((L1*skew_fact) - L1) / 2) = 1
% L1 = 2 / (skew_fact + 1)
% L0 = (2 * skew_fact) / (skew_fact + 1)
%

L0 = (2 * skew_fact) / (skew_fact + 1);
L1 = 2 / (skew_fact + 1);

Xs = zeros(s_count, 1);
for s=1:s_count,
    accept = 0;
    while (accept ~= 1)
        % Sample a candidate value
        x = rand();
        % Use linear interpolation to get its rejection threshold
        r_x = ((1 - x) * L0) + (x * L1);
        % Rejection sample with this probability
        if ((L0 * rand()) < r_x)
            Xs(s) = x;
            accept = 1;
        end
    end
end

return
end