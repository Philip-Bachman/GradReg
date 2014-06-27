function [Gg Lg Ag] = ...
    rbf_svm_cv(X, Y, tr_count, te_count, gammas, lambdas, cv_rounds)
% Train RKHS-regularized RBF regression/classification.
%

[Gg Lg] = meshgrid(gammas,lambdas);
Ag = zeros(size(Gg));
obs_count = size(X,1);

for c=1:cv_rounds,
    fprintf('STARTING CV ROUND %d...\n',c);
    idx = 1;
    timer = 1;
    for i=1:size(Gg,1),    
        lam = lambdas(i);
        for j=1:size(Gg,2),
            gam = gammas(j);
            fprintf('GAM = %.4f, LAM = %.4f\n',gam,lam);
            % Sample a train/test split for this round of CV.
            tr_idx = randsample(obs_count,tr_count);
            te_idx = setdiff(1:obs_count,tr_idx);
            te_idx = te_idx(randsample(numel(te_idx),te_count));
            Xtr = X(tr_idx,:);
            Ytr = Y(tr_idx,:);
            Xte = X(te_idx,:);
            Yte = Y(te_idx,:);
            % Get and RBF extractor and compute its kernel matrix
            FX = @( x ) rbf_extract(x, Xtr, gam);
            K = FX(Xtr);
            % Train RKHS-regularized RBF regression with this gam/lam pair
            SVM = LinReg(Xtr, Ytr, FX);
            SVM.lam_rdg = 1e-5;
            SVM.lam_fun = lam;
            SVM.train(Xtr,Ytr,K);
            [Yh acc] = SVM.classify(Xte,Yte);
            Ag(i,j) = Ag(i,j) + acc;
            %svm_opts = sprintf('-t 2 -g %.4f -c %.4f -q',gam,lam);
            %SVM = svmtrain(Ytr,Xtr,svm_opts);
            %[labs accs] = svmpredict(Yte,Xte,SVM);
            %Ag(i,j) = Ag(i,j) + accs(1);
            % Advance timer for pleasant viewer experience
            idx = idx + 1;
            if (mod(idx,floor(numel(Gg)/10)) == 0)
                fprintf('TIMER = %d\n',timer);
                timer = timer+1;
            end
        end
    end
    save('result_svm_cv_partial.mat');
end

Ag = Ag ./ cv_rounds;

return
end

