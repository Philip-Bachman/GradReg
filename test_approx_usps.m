clear;

load('usps.mat');
Y = class_inds(Y);

sample_counts = sort(unique(100 * round(logspace(0,4,20))),'ascend');

train_size = 500;
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
    FX = @( x ) rbf_extract(x, Xtr, 0.015);
    
    clear RME;
    RME = RegMatEst(FX, size(Xtr,2), size(FX(Xtr),2));
    RME.block_size = 100;
    fuzz_len = RegMatEst.compute_grad_len(Xtr, 250);
    smplr_1 = RegMatEst.obs_sampler_fuzzy(Xtr, 4*fuzz_len);
    smplr_2 = RegMatEst.box_sampler(Xtr, 4*fuzz_len);
    obs_sampler = RegMatEst.compound_sampler(smplr_1, smplr_2, 0.75);
    tic();
    K_multi = RME.estimate_arbo_multi(obs_sampler,orders,(fuzz_len/2),sample_counts);
    t=toc(); fprintf('time: %.4f\n',t);
    K_fin = K_multi{end};

    fprintf('Testing converged regularizer...\n');
    clear SVMg;
    SVMg = LinReg(Xtr, Ytr, FX);
    SVMg.loss_func = @LinReg.mcl2_loss_grad;
    SVMg.lam_rdg = 1e-6;
    SVMg.lam_fun = 0.25;
    M0 = RegMatEst.mix_regmats(K_fin,0);
    M0 = M0 ./ max(eig(M0));
    SVMg.train(Xtr, Ytr, M0);
    [Yh acc] = SVMg.classify(Xte,Yte);
    fprintf('accuracy: %.4f\n',acc);
    F_fin = SVMg.evaluate(Xte);
    
    fprintf('Testing partial regularizers...\n');
    for j=1:numel(sample_counts),
        K_smp = K_multi{j};
        M1 = RegMatEst.mix_regmats(K_smp,0);
        M1 = M1 ./ max(eig(M1));
        SVMg = LinReg(Xtr, Ytr, FX);
        SVMg.loss_func = @LinReg.mcl2_loss_grad;
        SVMg.lam_rdg = 1e-6;
        SVMg.lam_fun = 0.25;
        SVMg.train(Xtr, Ytr, M1);
        [Yh acc] = SVMg.classify(Xte,Yte);
        fprintf('accuracy: %.4f\n',acc);
        samp_accs(j,i) = acc;
        F_smp = SVMg.evaluate(Xte);
        mat_frob(j,i) = sum((M1(:)-M0(:)).^2) / numel(M0);
        mat_divr(j,i) = RegMatEst.regmat_divergence(M1, M0, 1e-5);
        fun_diff_fin(j,i) = var(F_smp(:)-F_fin(:)) / var(F_fin(:));
    end

    fprintf('Testing basic L2 regression...\n');
    clear SVMl;
    SVMl = LinReg(Xtr, Ytr, FX);
    SVMl.loss_func = @LinReg.mcl2_loss_grad;
    SVMl.lam_rdg = 1e-4;
    SVMl.lam_fun = 0.0;
    SVMl.train(Xtr, Ytr, M0);
    [Yh acc] = SVMl.classify(Xte,Yte);
    fprintf('accuracy: %.4f\n',acc);
    l_accs(i) = acc;
    
    save('result_approx_usps.mat');
end










%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%