% Run some simple tests on the 'MNIST' digit classification task.
%
% Use "image-manifold-structured" regularization.
%
% Test different relative weights for balancing the "manifold-structured"
% regularizer versus the normal RKHSish regularizer, and for balancing the
% translation and rotation components of the "manifold-structured" regularizer.
%

clear;

% Load and scale the data
load('mnist_70k_trte.mat');

train_size = 500;
ss_size = 30000;
test_size = 10000;
test_count = 30;
lams_l2 = [1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3];
lams_rkhs = [1e-4 3e-4 1e-3 3e-3 1e-2 3e-2];
lams_sarn = [0.01 0.03 0.1 0.3 1.0];
lams_ss = [0.0 0.1 0.3 1.0];
alphas_ss = [0.25 0.5 0.75];

l_accs = zeros(numel(lams_l2),test_count);
k_accs = zeros(numel(lams_rkhs),test_count);
ss_accs = zeros(numel(lams_sarn),numel(lams_ss),numel(alphas_ss),test_count);

for i=1:test_count,
    % Split observations into training, semi-supervised, and test sets
    tr_idx = randsample(size(X_train,1),train_size);
    ss_idx = setdiff(1:size(X_train,1),tr_idx);
    ss_idx = ss_idx(randperm(numel(ss_idx)));
    te_idx = ss_idx((ss_size+1):(ss_size+test_size));
    ss_idx((ss_size+1):end) = [];

    Xtr = double(X_train(tr_idx,:)) ./ 255;
    Xss = double(X_train(ss_idx,:)) ./ 255;
    Xte = double(X_train(te_idx,:)) ./ 255;
    Ytr = class_inds(Y_train(tr_idx,:));
    Yte = class_inds(Y_train(te_idx,:));

    % Setup the feature extractors to use.
    FX = @( x ) rbf_extract(x, Xtr, 0.025);

    % Estimate an RKHSish higher-order gradient regularizer using SAR
    clear RME;
    RME = RegMatEst(FX, size(Xtr,2), size(FX(Xtr),2));
    RME.block_size = 500;
    fuzz_len = RegMatEst.compute_grad_len(Xtr, 250);
    smplr_1 = RegMatEst.obs_sampler_fuzzy(Xtr, 4*fuzz_len);
    smplr_2 = RegMatEst.box_sampler(Xtr, 4*fuzz_len);
    obs_sampler = RegMatEst.compound_sampler(smplr_1, smplr_2, 0.75);
    orders = [1 2 3 4];
    K_all = RME.estimate_arbo(obs_sampler, orders, (fuzz_len/2), 150000);
    K_sar = RegMatEst.mix_regmats(K_all,0);
    K_sar = K_sar ./ max(eig(K_sar));
    K_krn = FX(Xtr);
    K_krn = K_krn ./ max(eig(K_krn));
    
    % Estimate image-oriented regularizers based on translation and rotation
    clear IME;
    IME = ImgMatEst(FX, 256, 500);
    K_tra = IME.trans_mat([Xtr; Xss], 0.5);
    K_tra = K_tra ./ max(eig(K_tra));
    K_rot = IME.rot_mat([Xtr; Xss], 1.0);
    K_rot = K_rot ./ max(eig(K_rot));
    

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
        SVM.train(Xtr,Ytr,K_krn);
        [Yh acc] = SVM.classify(Xte,Yte);
        k_accs(j,i) = acc;
        fprintf('RKHS--ACC: %.4f\n',acc);
    end

    for j1=1:numel(lams_sarn),
        for j2=1:numel(lams_ss),
            for j3=1:numel(alphas_ss),
                % Get the parameters for constructing the joint regularizer
                lam_sar = lams_sarn(j1);
                lam_ss = lams_ss(j2);
                alpha_ss = alphas_ss(j3);
                % Construct the joint gradient regularizer for these parameters
                K_jnt = (lam_sar * K_sar) + ...
                    (lam_ss * ((alpha_ss * K_tra) + ((1 - alpha_ss) * K_rot)));
                SVM = LinReg(Xtr, Ytr, FX);
                SVM.loss_func = @LinReg.mcl2_loss_grad;
                SVM.lam_rdg = 1e-6;
                SVM.lam_fun = 1.0;
                SVM.train(Xtr,Ytr,K_jnt);
                [Yh acc] = SVM.classify(Xte,Yte);
                ss_accs(j1,j2,j3,i) = acc;
                fprintf('SAR lam_sar=%.4f, lam_ss=%.4f, alpha_ss=%.4f, acc=%.4f\n',...
                    lam_sar, lam_ss, alpha_ss, acc);
            end
        end
    end

    save('res_mnist_manifold_acc.mat');

end






