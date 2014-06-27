% Select kernel bandwidth parameters on housing data for Gaussian RBF and
% B-Spline RBF kernels.
%

clear;

train_sizes = 250; %[150 200 250 300 350 400 450];

for ts_num=1:numel(train_sizes),

    train_size = train_sizes(ts_num);
    
    % Load and scale the data
    load('housing.mat');

    X = bsxfun(@minus, X, mean(X));
    X = bsxfun(@rdivide, X, (std(X)+1e-8));
    Y = Y - mean(Y);

    test_count = 25;
    lams = [1e-6 3e-6 1e-5 3e-5 1e-4 3e-4];
    gams_gauss = [0.05 logspace(-1,0,4)];
    gams_bspln = [0.05 logspace(-1,0,4)];
    lams_aspln = [1e-4 1e-3 3e-3 1e-2 3e-2 1e-1];
    cp_counts = [10 15 20 25 30 40];

    g_accs = zeros(numel(gams_gauss),numel(lams),test_count);
    b_accs = zeros(numel(gams_bspln),numel(lams),test_count);
    s_accs = zeros(numel(cp_counts),numel(lams_aspln),test_count);

    for i=1:test_count,
        fprintf('TRAIN SIZE %d, TEST %d\n',train_size,i);
        % Split training set into a 'supervised' and 'unsupervised' portion
        train_count = train_size;
        tr_idx = randsample(size(X,1),train_count);
        te_idx = setdiff(1:size(X,1),tr_idx);

        Xtr = X(tr_idx,:);
        Ytr = Y(tr_idx,:);
        Xte = X(te_idx,:);
        Yte = Y(te_idx,:);
        
        % Test various hyperparameters for B-Spline RBF
        fprintf('BSpln:');
        for g_num=1:numel(gams_bspln),
            FX = @( x ) bspline_extract(x, Xtr, 4, gams_bspln(g_num));
            K = FX(Xtr);
            K = K ./ max(eig(K));
            if (min(K(:)) < 0)
                K = K - min(K(:));
            end
            for l_num=1:numel(lams),
                SVM = LinReg(Xtr, Ytr, FX);
                SVM.lam_rdg = 1e-6;
                SVM.lam_fun = lams(l_num);
                SVM.train(Xtr,Ytr,K);
                acc = SVM.test(Xte,Yte);
                b_accs(g_num,l_num,i) = acc;
                fprintf('.');
            end
        end
        fprintf('\n');
        
        % Test various hyperparameters for Gaussian RBF
        fprintf('Gauss:');
        for g_num=1:numel(gams_gauss),
            FX = @( x ) rbf_extract(x, Xtr, gams_gauss(g_num));
            K = FX(Xtr);
            K = K ./ max(eig(K));
            for l_num=1:numel(lams),
                SVM = LinReg(Xtr, Ytr, FX);
                SVM.lam_rdg = 1e-6;
                SVM.lam_fun = lams(l_num);
                SVM.train(Xtr,Ytr,K);
                acc = SVM.test(Xte,Yte);
                g_accs(g_num,l_num,i) = acc;
                fprintf('.');
            end
        end
        fprintf('\n');
        
        % Test various hyperparameters for additive penalized splines
        opts_tr = struct();
        opts_tr.batch_count = 1;
        opts_tr.batch_iters = 1500;
        test_var = sum((Yte(:)-mean(Yte(:))).^2);
        fprintf('ASpln:');
        for l_num=1:numel(lams_aspln),
            for cp_num=1:numel(cp_counts),
                opts_sp = struct();
                opts_sp.ord_lams = lams_aspln(l_num) * [0.1 1.0 1.0 1.0 1.0];
                opts_sp.cp_count = cp_counts(cp_num);
                clear PSF;
                PSF = PSplineFit(Xtr, Ytr, opts_sp);
                PSF.bs_order = 2;
                PSF.train(Xtr, Ytr, opts_tr);
                Yh = PSF.evaluate(Xte);
                acc = sum((Yh(:)-Yte(:)).^2) / test_var;
                s_accs(cp_num,l_num,i) = acc;
                fprintf('.');
            end
        end
        fprintf('\n');
        
    end
    
    %save(sprintf('hypers_housing_ts%d.mat',train_size));
    
end





