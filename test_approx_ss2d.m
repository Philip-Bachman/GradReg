
sample_counts = 100 * round(logspace(0,4,20));
    
obs_count = 75;
test_count = 50;
gammas = [0.5 2.0 8.0];
orders = [1 2];

l2_accs = zeros(1, test_count);
opt_accs = zeros(1, test_count);
samp_accs = zeros(numel(sample_counts), test_count);

mat_frob = zeros(numel(sample_counts),test_count);
mat_divr = zeros(numel(sample_counts),test_count);
fun_diff_fin = zeros(numel(sample_counts),test_count);
fun_diff_opt = zeros(numel(sample_counts),test_count);

for i=1:test_count,

    fprintf('STARTING TEST %d!\n',i);
    
    [Xtr Ytr Xte Yte Xs] = data_synthsin2d(obs_count, 5000, 0.1);

    FX = @(x) rbf_extract(x, Xtr, gammas, 0);
    clear RME;
    RME = RegMatEst(FX, size(Xtr,2), size(FX(Xtr),2));
    RME.use_strict_len = 0;
    RME.block_size = 100;
    obs_sampler = RegMatEst.obs_sampler_inv(Xtr,1.0,2.0);
    %obs_sampler = RegMatEst.box_sampler([Xtr; Xte],2.0);
    K_multi = RME.estimate_arbo_multi(obs_sampler,orders,0.1,sample_counts);
    K_fin = K_multi{end};
    
    fprintf('Learning optimal function...\n');
    clear SVMl;
    SVMl = LinReg(Xtr, Ytr, FX);
    SVMl.lam_rdg = 1e-8;
    SVMl.lam_fun = 0;
    K = K_fin{2} ./ max(eig(K_fin{2}));
    SVMl.train(Xte, Yte, K);
    opt_accs(i) = SVMl.test(Xte, Yte);
    F_opt = SVMl.evaluate(Xte);

    fprintf('Testing converged regularizer...\n');
    clear SVMg;
    SVMg = LinReg(Xtr, Ytr, FX);
    SVMg.lam_rdg = 1e-5;
    SVMg.lam_fun = 0.1;
    M0 = K_fin{2} ./ max(eig(K_fin{2}));
    SVMg.train(Xtr, Ytr, M0);
    SVMg.test(Xte,Yte);
    F_fin = SVMg.evaluate(Xte);
    
    fprintf('Testing partial regularizers...\n');
    for j=1:numel(sample_counts),
        K_smp = K_multi{j};
        SVMg = LinReg(Xtr, Ytr, FX);
        SVMg.lam_rdg = 1e-5;
        SVMg.lam_fun = 0.1;
        M1 = K_smp{2} ./ max(eig(K_smp{2}));
        SVMg.train(Xtr, Ytr, M1);
        samp_accs(j,i) = SVMg.test(Xte,Yte);
        F_smp = SVMg.evaluate(Xte);
        mat_frob(j,i) = sum((M1(:)-M0(:)).^2) / numel(M0);
        mat_divr(j,i) = RegMatEst.regmat_divergence(M1, M0, 1e-5);
        fun_diff_fin(j,i) = var(F_smp(:)-F_fin(:)) / var(F_fin);
        fun_diff_opt(j,i) = var(F_smp(:)-F_opt(:)) / var(F_opt);
    end

    fprintf('Testing basic L2 regression...\n');
    clear SVMl;
    SVMl = LinReg(Xtr, Ytr, FX);
    SVMl.lam_rdg = 3e-4;
    SVMl.lam_fun = 0;
    K = K_fin{2} ./ max(eig(K_fin{2}));
    SVMl.train(Xtr, Ytr, K);
    l2_accs(i) = SVMl.test(Xte, Yte);
end

save('result_approx_ss2d.mat');









%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%