function [ Xtr Ytr Xte Yte Xs ] = data_synthsin1d(tr_count, te_count, noise_std)
% Generate some random sinish data
%

s_count = (tr_count + te_count);

cycle_count = 4.0;
cycle_scale = 2.5;

vals = linspace(0,1,8*s_count);
vals = randsample(vals, s_count, false);

%vals = transpose(skewed_uniform(s_count, 2.0));

X = vals';
X(1) = 0;
Y = sin(2*pi*cycle_count*X);
Y = Y .* (1 + (X ./ max(X)));
X = cycle_scale.^(cycle_count * X);
X = X - 1;
X = X * (16 / max(X));

X_scale = max(X) - min(X);
X_min = min(X);
Xs = (rand(10000,1) * X_scale) + X_min;

Xtr = X(1:tr_count);
Ytr = Y(1:tr_count);
[vals idx] = sort(Xtr,'ascend');
Xtr = Xtr(idx);
Ytr = Ytr(idx);
noise_scales = 1 + (Xtr ./ max(Xtr));
Ytr = Ytr + (noise_std * (randn(size(Ytr)) .* noise_scales));

Xte = X((tr_count+1):end);
Yte = Y((tr_count+1):end);

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