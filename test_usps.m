% Run some simple tests on the 'USPS' digit classification task 
%

clear;

% Load and scale the data
load('usps.mat');

Y = class_inds(Y);

train_size = 500;
test_count = 100;
lams_l2 = [1e-7 1e-6 3e-6 1e-5 3e-5 1e-4];
lams_rkhs = [1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0];
lams_sarn = [0.01 0.03 0.1 0.3 1.0 3.0 10.0];

l_accs = zeros(numel(lams_l2),test_count);
k_accs = zeros(numel(lams_rkhs),test_count);
y1_accs = zeros(numel(lams_sarn),test_count);
n1_accs = zeros(numel(lams_sarn),test_count);


for i=1:test_count,
    % Split training set into training and test sets
    train_count = train_size;
    tr_idx = randsample(size(X,1),train_count);
    te_idx = setdiff(1:size(X,1),tr_idx);
    te_idx = te_idx(randperm(numel(te_idx)));

    Xtr = X(tr_idx,:);
    Ytr = Y(tr_idx,:);
    Xte = X(te_idx,:);
    Yte = Y(te_idx,:);

    % Setup the feature extractors to use.
    FX = @( x ) rbf_extract(x, Xtr, 0.015);

    RME = RegMatEst(FX, size(Xtr,2), size(FX(Xtr),2));
    RME.block_size = 500;
    fuzz_len = RegMatEst.compute_grad_len(Xtr, 250);
    smplr_1 = RegMatEst.obs_sampler_fuzzy(Xtr, 4*fuzz_len);
    smplr_2 = RegMatEst.box_sampler(Xtr, 4*fuzz_len);
    obs_sampler = RegMatEst.compound_sampler(smplr_1, smplr_2, 0.75);
    orders = [1 2 3 4];
    K_all = RME.estimate_arbo(obs_sampler, orders, (fuzz_len/2), 100000);
    K_k = FX(Xtr);
    K_k = K_k ./ max(eig(K_k));

    for j=1:numel(lams_l2),
        SVM = LinReg(Xtr, Ytr, FX);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = lams_l2(j);
        SVM.lam_fun = 0.0;
        SVM.train(Xtr,Ytr);
        [Yh acc] = SVM.classify(Xte,Yte);
        l_accs(j,i) = acc;
        fprintf('L2----ACC: %.4f\n',acc);
    end
    
    for j=1:numel(lams_rkhs),
        SVM = LinReg(Xtr, Ytr, FX);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = 1e-6;
        SVM.lam_fun = lams_rkhs(j);
        SVM.train(Xtr,Ytr,K_k);
        [Yh acc] = SVM.classify(Xte,Yte);
        k_accs(j,i) = acc;
        fprintf('RKHS--ACC: %.4f\n',acc);
    end

    for j=1:numel(lams_sarn),
        K = RegMatEst.mix_regmats(K_all,0);
        %K = RegMatEst.mix_rm_rbfish(K_all,orders,1.0);
        K = K ./ max(eig(K));
        SVM = LinReg(Xtr, Ytr, FX);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = 1e-6;
        SVM.lam_fun = lams_sarn(j);
        SVM.train(Xtr,Ytr,K);
        [Yh acc] = SVM.classify(Xte,Yte);
        y1_accs(j,i) = acc;
        fprintf('SAR14-ACC: %.4f\n',acc);
    end

    for j=1:numel(lams_sarn),
        K_some = cell(1,length(K_all)-1);
        for k=2:length(K_all),
            K_some{k-1} = K_all{k};
        end
        K = RegMatEst.mix_regmats(K_some,0);
        K = K ./ max(eig(K));
        SVM = LinReg(Xtr, Ytr, FX);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = 1e-5;
        SVM.lam_fun = lams_sarn(j);
        SVM.train(Xtr,Ytr,K);
        [Yh acc] = SVM.classify(Xte,Yte);
        n1_accs(j,i) = acc;
        fprintf('SAR24-ACC: %.4f\n',acc);
    end

    save('res_usps_acc.mat');

end






