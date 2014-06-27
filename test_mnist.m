% Run some simple tests on the 'MNIST' digit classification task.
%

clear;

% Load and scale the data
load('mnist_70k_trte.mat');

train_size = 1000;
test_size = 10000;
test_count = 30;
lams_l2 = [1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3];
lams_rkhs = [1e-4 3e-4 1e-3 3e-3 1e-2 3e-2];
lams_sarn = [0.003 0.01 0.03 0.1 0.3 1.0 3.0];

l_accs = zeros(numel(lams_l2),test_count);
k_accs = zeros(numel(lams_rkhs),test_count);
s_accs = zeros(numel(lams_sarn),test_count);

for i=1:test_count,
    % Split observations into training and test sets
    tr_idx = randsample(size(X_train,1),train_size);
    Xtr = double(X_train(tr_idx,:)) ./ 255;
    Xte = double(X_test(:,:)) ./ 255;
    Ytr = class_inds(Y_train(tr_idx,:));
    Yte = class_inds(Y_test(:,:));

    % Setup the feature extractors to use. One uses Gaussian RBFs centered
    % on the training points, and the other uses the encoding parameters
    % learned by a denoising autoencoder.
    FK = @( x ) rbf_extract(x, Xtr, 0.025);

    % Estimate an RKHSish higher-order gradient regularizer using SAR
    clear RME;
    RME = RegMatEst(FK, size(Xtr,2), size(FK(Xtr),2));
    RME.block_size = 500;
    fuzz_len = RegMatEst.compute_grad_len(Xtr, 500);
    smplr_1 = RegMatEst.obs_sampler_fuzzy(Xtr, 4*fuzz_len);
    smplr_2 = RegMatEst.box_sampler(Xtr, 4*fuzz_len);
    obs_sampler = RegMatEst.compound_sampler(smplr_1, smplr_2, 0.75);
    orders = [1 2 3 4];
    K_all = RME.estimate_arbo(obs_sampler, orders, (fuzz_len/2), 150000);
    K_sar = RegMatEst.mix_regmats(K_all,0);
    K_sar = K_sar ./ max(eig(K_sar));
    K_krn = FK(Xtr);
    K_krn = K_krn ./ max(eig(K_krn));
    
    % Train L2-regularized joint RBF/DAE features
    for j=1:numel(lams_l2),
        SVM = LinReg(Xtr, Ytr, FK);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = lams_l2(j);
        SVM.lam_fun = 0.0;
        SVM.train(Xtr,Ytr);
        [Yh acc] = SVM.classify(Xte,Yte);
        l_accs(j,i) = acc;
        fprintf('L2----ACC: %.4f\n',acc);
    end
    
    % Train RKHS-regularized RBFs
    for j=1:numel(lams_rkhs),
        SVM = LinReg(Xtr, Ytr, FK);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = 1e-5;
        SVM.lam_fun = lams_rkhs(j);
        SVM.train(Xtr,Ytr,K_krn);
        [Yh acc] = SVM.classify(Xte,Yte);
        k_accs(j,i) = acc;
        fprintf('RKHS--ACC: %.4f\n',acc);
    end

    % Train SAR-regularized joint RBF/DAE features
    for j=1:numel(lams_sarn),
        lam_sar = lams_sarn(j);
        SVM = LinReg(Xtr, Ytr, FK);
        SVM.loss_func = @LinReg.mcl2_loss_grad;
        SVM.lam_rdg = 1e-5;
        SVM.lam_fun = lams_sarn(j);
        SVM.train(Xtr,Ytr,K_sar);
        [Yh acc] = SVM.classify(Xte,Yte);
        s_accs(j,i) = acc;
        fprintf('SAR---ACC: %.4f\n',acc);
    end

    save('res_mnist_acc_3.mat');

end






