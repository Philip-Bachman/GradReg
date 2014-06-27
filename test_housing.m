% Run some simple tests on the 'housing' regression dataset
%

clear;

train_sizes = [150 200 250 300 350 400 450];

for ts_num=1:numel(train_sizes),

    train_size = train_sizes(ts_num);
    
    % Load and scale the data
    load('housing.mat');

    X = bsxfun(@minus, X, mean(X));
    X = bsxfun(@rdivide, X, (std(X)+1e-8));
    Y = Y - mean(Y);

    test_count = 100;
    lams_l2 = [1e-4 2e-4 5e-4 1e-3 2e-3 5e-3];
    lams_rkhs = [1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3];
    lams_sarn = [0.1 0.3 1.0 3.0 10.0 25.0 50.0];
    lams_aspln = [1e-4 1e-3 3e-3 1e-2 3e-2 1e-1];

    l_accs = zeros(numel(lams_l2),test_count);
    k_accs = zeros(numel(lams_rkhs),test_count);
    b_accs = zeros(numel(lams_rkhs),test_count);
    a_accs = zeros(numel(lams_aspln),test_count);
    y1_accs = zeros(numel(lams_sarn),test_count);
    t_accs = zeros(1,test_count);

    for i=1:test_count,
        % Split training set into a 'supervised' and 'unsupervised' portion
        train_count = train_size;
        tr_idx = randsample(size(X,1),train_count);
        te_idx = setdiff(1:size(X,1),tr_idx);

        Xtr = X(tr_idx,:);
        Ytr = Y(tr_idx,:);
        Xte = X(te_idx,:);
        Yte = Y(te_idx,:);

        % Setup the feature extractors to use. One is "additive" features,
        % comprising 1D RBFs and the other is "full" features, comprising 
        % Gaussian RBFs defined over all input dimensions simultaneously.
        % There's also an extractor for 4th-order B-Spline RBF kernels.
        FA = housing_features(X);
        FK = @( x ) rbf_extract(x, Xtr, 0.05);
        FX = @( x ) [FA( x ) FK( x )];
        FB = @( x ) bspline_extract(x, Xtr, 4, 0.2);
        
        % Estimate regularizer stuff, for RKHS, SAR, and B-Spline kernels.
        RME = RegMatEst(FX, size(Xtr,2), size(FX(Xtr),2));
        RME.block_size = 500;
        fuzz_len = RegMatEst.compute_grad_len(Xtr, 200);
        smplr_1 = RegMatEst.obs_sampler_fuzzy(Xtr, fuzz_len);
        smplr_2 = RegMatEst.box_sampler(X, fuzz_len);
        obs_sampler = RegMatEst.compound_sampler(smplr_1, smplr_2, 0.75);
        orders = [1 2 3 4];
        K_all = RME.estimate_arbo(obs_sampler, orders, (fuzz_len/2), 150000);
        K_k = FK(Xtr);
        K_k = K_k ./ max(eig(K_k));
        K_b = FB(Xtr);
        K_b = K_b ./ max(eig(K_b));

        % Train L2-regularized regressions with same features as for SAR
        for j=1:numel(lams_l2),
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FX);
            SVM.lam_rdg = lams_l2(j);
            SVM.lam_fun = 0.0;
            SVM.train(Xtr,Ytr,K_all{1});
            Wl = SVM.W;
            acc = SVM.test(Xte,Yte);
            l_accs(j,i) = acc;
            fprintf('L2----ACC: %.4f\n',acc);
        end

        % Train RKHS-regularized Gaussian RBF regressions
        for j=1:numel(lams_rkhs),
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FK);
            SVM.lam_rdg = 1e-6;
            SVM.lam_fun = lams_rkhs(j);
            SVM.train(Xtr,Ytr,K_k);
            acc = SVM.test(Xte,Yte);
            k_accs(j,i) = acc;
            fprintf('RKHS--ACC: %.4f\n',acc);
        end
        
        % Train RKHS-regularized 4th-order B-spline RBF regressions
        for j=1:numel(lams_rkhs),
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FB);
            SVM.lam_rdg = 1e-6;
            SVM.lam_fun = lams_rkhs(j);
            SVM.train(Xtr,Ytr,K_b);
            acc = SVM.test(Xte,Yte);
            b_accs(j,i) = acc;
            fprintf('BSPLN-ACC: %.4f\n',acc);
        end
        
        % Train additive penalized 2nd-order B-spline regressions
        opts_tr = struct();
        opts_tr.batch_count = 1;
        opts_tr.batch_iters = 1000;
        opts_sp = struct();
        opts_sp.cp_count = 30;
        for j=1:numel(lams_aspln),
            opts_sp.ord_lams = lams_aspln(j) * [0.1 1.0 1.0 1.0 1.0];
            clear PSF;
            PSF = PSplineFit(Xtr, Ytr, opts_sp);
            PSF.train(Xtr, Ytr, opts_tr);
            Yh = PSF.evaluate(Xte);
            acc = 1 - (sum((Yh(:)-Yte(:)).^2) / sum((Yte(:)-mean(Yte(:))).^2));
            a_accs(j,i) = acc;
            fprintf('PSPLN-ACC: %.4f\n',acc);
        end

        % Train boosted regression trees of depth 3
        clear TREE;
        TREE = TreeBooster(Xtr, Ytr, 3, 0.1);
        TREE.multi_extend(Xtr, Ytr, 250);
        Fte = TREE.evaluate(Xte);
        t_accs(i) = 1 - (var(Fte - Yte) / var(Yte));
        fprintf('TREE--ACC: %.4f\n',t_accs(i));

        % Train SAR-regularized regressions with combo features
        for j=1:numel(lams_sarn),
            K = RegMatEst.mix_regmats(K_all,0);
            %K = RegMatEst.mix_rm_rbfish(K_all,orders,0.5);
            K = K ./ max(eig(K));
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FX);
            SVM.lam_rdg = 1e-4;
            SVM.lam_fun = lams_sarn(j);
            SVM.train(Xtr,Ytr,K);
            acc = SVM.test(Xte,Yte);
            y1_accs(j,i) = acc;
            fprintf('SAR14-ACC: %.4f\n',acc);
        end

    end
    
    save(sprintf('res_housing_ts%d.mat',train_size));
    
end





