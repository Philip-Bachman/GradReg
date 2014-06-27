
% test_count = 20;
% gammas = [1.0 2.0 4.0 8.0 16.0];
% lambdas = logspace(-4,0,21);
% 
% Ag = zeros(numel(gammas),numel(lambdas),test_count);
% Gg = zeros(numel(gammas),numel(lambdas));
% Lg = zeros(numel(gammas),numel(lambdas));
% 
% for g=1:numel(gammas),
%     gam = gammas(g);
%     for l=1:numel(lambdas),
%         lam = lambdas(l);
%         accs = zeros(1,test_count);
%         for i=1:test_count,
%             [Xtr Ytr Xte Yte Xs] = data_synthsin2d(70, 2000, 0.2);
%             FFk = @(x) rbf_extract(x, Xtr, gam, 0);
%             Kk = FFk(Xtr);
%             Kk = Kk ./ max(eig(Kk));
%             clear SVMk;
%             SVMk = LinReg(Xtr, Ytr, FFk);
%             SVMk.lam_rdg = 1e-6;
%             SVMk.lam_fun = lam;
%             SVMk.train(Xtr, Ytr, Kk);
%             fprintf('Test: ');
%             Ag(g,l,i) = SVMk.test(Xte, Yte);
%         end
%         fprintf('GAM: %.4f, LAM: %.4f, ACC: %.4f\n',gam,lam,mean(Ag(g,l,:)));
%         Gg(g,l) = gam;
%         Lg(g,l) = lam;
%     end
% end
% Ag_mean = mean(Ag,3);

sample_counts = 50:5:100;

for s_num=1:numel(sample_counts),
    
    sample_count = sample_counts(s_num);
    test_count = 50;
    gammas = [0.5 2.0 8.0];
    k_lambdas = logspace(-5,0,6);
    lambdas = logspace(-3,1,9);
    l2_lambdas = logspace(-4,-1,7);
    kern_accs = zeros(numel(gammas), numel(k_lambdas), test_count);
    hess_accs = zeros(numel(lambdas), test_count);
    bias_accs = zeros(numel(lambdas), test_count);
    l2_accs = zeros(numel(l2_lambdas), test_count);

    for i=1:test_count,

        [Xtr Ytr Xte Yte Xs] = data_synthsin2d(sample_count, 2000, 0.1);

        fprintf('Testing kernel-regularized regression...\n');
        kSVMs = cell(1,numel(gammas));
        for g=1:numel(gammas),
            FFk = @(x) rbf_extract(x, Xtr, gammas(g), 0);
            Kk = FFk(Xtr);
            Kk = Kk ./ max(eig(Kk));
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FFk);
            SVM.lam_rdg = 1e-6;
            max_acc = 0;
            for l=1:numel(k_lambdas),
                SVM.lam_fun = k_lambdas(l);
                fprintf('gamma = %.4f, lam = %.4f\n',gammas(g),k_lambdas(l));
                SVM.train(Xtr, Ytr, Kk);
                kg_acc = SVM.test(Xte, Yte);
                kern_accs(g,l,i) = kg_acc;
                if (kg_acc > max_acc)
                    max_acc = kg_acc;
                    kSVMs{g} = SVM;
                end
            end
        end

        FFh = @(x) rbf_extract(x, Xtr, gammas, 0);
        clear RME;
        RME = RegMatEst(FFh, size(Xtr,2), size(FFh(Xtr),2));
        Kh = RME.estimate_hess(RegMatEst.obs_sampler_inv(Xtr,1.0,2.0),0.2,100000);
        %Kh = RME.estimate_hess(RegMatEst.obs_sampler_fuzzy(Xtr,1.0),0.2,75000);
        %Kg = RME.estimate_grad(RegMatEst.obs_sampler_fuzzy(Xtr,1.0),0.2,75000);
        %Kh = Kh + (0.1 * (Kh * Kh));
        %Khd = diag(diag(Kh).^(-1/2));
        %Kh = Khd * Kh * Khd;
        Kh = Kh ./ max(eig(Kh));

        fprintf('Testing approx-regularized regression...\n');
        hSVMs = cell(1,numel(lambdas));
        for l=1:numel(lambdas),
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FFh);
            SVM.lam_rdg = 1e-5;
            SVM.lam_fun = lambdas(l);
            SVM.train(Xtr, Ytr, Kh);
            hess_accs(l,i) = SVM.test(Xte, Yte);
            hSVMs{l} = SVM;
        end
        
        FFh = @(x) rbf_extract(x, Xtr, gammas, 0);
        clear RME;
        RME = RegMatEst(FFh, size(Xtr,2), size(FFh(Xtr),2));
        Mb = diag([1 8]);
        Kh = RME.estimate_hess(RegMatEst.obs_sampler_inv(Xtr,1.0,2.0),0.2,100000,Mb);
        Kh = Kh ./ max(eig(Kh));

        fprintf('Testing biased-regularized regression...\n');
        bSVMs = cell(1,numel(lambdas));
        for l=1:numel(lambdas),
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FFh);
            SVM.lam_rdg = 1e-5;
            SVM.lam_fun = lambdas(l);
            SVM.train(Xtr, Ytr, Kh);
            bias_accs(l,i) = SVM.test(Xte, Yte);
            bSVMs{l} = SVM;
        end
        
        fprintf('Testing L2-regularized regression...\n');
        lSVMs = cell(1,numel(l2_lambdas));
        for l=1:numel(l2_lambdas),
            clear SVM;
            SVM = LinReg(Xtr, Ytr, FFh);
            SVM.lam_rdg = l2_lambdas(l);
            SVM.lam_fun = 0;
            SVM.train(Xtr, Ytr, Kh);
            l2_accs(l,i) = SVM.test(Xte, Yte);
            lSVMs{l} = SVM;
        end

    end
    
    test_name = sprintf('synthsin2d_final_%d.mat',sample_count);
    save(test_name);
    
end