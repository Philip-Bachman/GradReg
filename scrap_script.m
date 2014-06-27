% Generate samples from Synthsin2d, to plot "true" function

[Xtr Ytr Xte Yte Xs] = data_synthsin2d(5000, 100, 0.0);
FFk = @(x) rbf_extract(x, Xtr, 2.0, 0);
Kk = FFk(Xtr);
Kk = Kk ./ max(eig(Kk));
clear SVM;
SVM = LinReg(Xtr, Ytr, FFk);
SVM.lam_rdg = 1e-5;
SVM.lam_fun = 1e-1;
SVM.train(Xtr, Ytr, Kk);
g_acc = SVM.test(Xte, Yte);
