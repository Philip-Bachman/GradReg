clear;

load('housing.mat');

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, (std(X)+1e-8));
Y = Y - mean(Y);

sample_counts = sort(unique(100 * round(logspace(0,4,20))),'ascend');

train_size = 300;
test_count = 50;
orders = [1 2 3 4];

l_accs = zeros(1, test_count);
samp_accs = zeros(numel(sample_counts), test_count);

mat_frob = zeros(numel(sample_counts),test_count);
mat_divr = zeros(numel(sample_counts),test_count);
fun_diff_fin = zeros(numel(sample_counts),test_count);

for i=1:test_count,

    fprintf('STARTING TEST %d!\n',i);
    
    tr_idx = randsample(size(X,1),train_size);
    te_idx = setdiff(1:size(X,1),tr_idx);

    Xtr = X(tr_idx,:);
    Ytr = Y(tr_idx,:);
    Xte = X(te_idx,:);
    Yte = Y(te_idx,:);
    
    % Setup the feature extractors to use.
    anchors = Xtr;
    fx1 = housing_features(X);
    fx2 = @( x ) rbf_extract(x, anchors, 0.05);
    FX = @( x ) [fx1( x ) fx2( x )];
    
    clear RME;
    RME = RegMatEst(FX, size(Xtr,2), size(FX(Xtr),2));
    RME.block_size = 100;
    fuzz_len = RegMatEst.compute_grad_len(Xtr, 200);
    smplr_1 = RegMatEst.obs_sampler_fuzzy(Xtr, fuzz_len);
    smplr_2 = RegMatEst.box_sampler(X, fuzz_len);
    obs_sampler = RegMatEst.compound_sampler(smplr_1, smplr_2, 0.75);
    K_multi = RME.estimate_arbo_multi(obs_sampler,orders,(fuzz_len/2),sample_counts);
    K_fin = K_multi{end};

    fprintf('Testing converged regularizer...\n');
    clear SVMg;
    SVMg = LinReg(Xtr, Ytr, FX);
    SVMg.lam_rdg = 1e-4;
    SVMg.lam_fun = 1.0;
    M0 = RegMatEst.mix_regmats(K_fin,0);
    M0 = M0 ./ max(eig(M0));
    SVMg.train(Xtr, Ytr, M0);
    SVMg.test(Xte,Yte);
    F_fin = SVMg.evaluate(Xte);
    
    fprintf('Testing partial regularizers...\n');
    for j=1:numel(sample_counts),
        K_smp = K_multi{j};
        M1 = RegMatEst.mix_regmats(K_smp,0);
        M1 = M1 ./ max(eig(M1));
        SVMg = LinReg(Xtr, Ytr, FX);
        SVMg.lam_rdg = 1e-4;
        SVMg.lam_fun = 1.0;
        SVMg.train(Xtr, Ytr, M1);
        samp_accs(j,i) = SVMg.test(Xte,Yte);
        F_smp = SVMg.evaluate(Xte);
        mat_frob(j,i) = sum((M1(:)-M0(:)).^2) / numel(M0);
        mat_divr(j,i) = RegMatEst.regmat_divergence(M1, M0, 1e-5);
        fun_diff_fin(j,i) = var(F_smp(:)-F_fin(:)) / var(F_fin(:));
    end

    fprintf('Testing basic L2 regression...\n');
    clear SVMl;
    SVMl = LinReg(Xtr, Ytr, FX);
    SVMl.lam_rdg = 0.002;
    SVMl.lam_fun = 0.0;
    SVMl.train(Xtr, Ytr, M0);
    l_accs(i) = SVMl.test(Xte, Yte);
    
end

save('result_approx_housing.mat');









%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%