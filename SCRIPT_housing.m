clear;
warning off all;

OBS_COUNTS = zeros(1,7);

K_ACCS = zeros(3,7); % Gaussian RBF kernel
L_ACCS = zeros(3,7); % L2-regularized combo features
S_ACCS = zeros(3,7); % SAR-regularizer combo features
B_ACCS = zeros(3,7); % B-Spline RBF kernel
A_ACCS = zeros(3,7); % Additive P-Splines
T_ACCS = zeros(3,7); % Boosted trees

for II=1:7,
    load(sprintf('res_housing_ts%d.mat',(100+(II*50))));
    
    OBS_COUNTS(II) = train_size;
    
    % process all gaussian rbf regression results
    [val idx] = max(mean(k_accs,2));
    K_ACCS(2,II) = mean(k_accs(idx,:));
    [muhat,sigmahat,muci,sigmaci] = normfit(k_accs(idx,:));
    K_ACCS(1,II) = muci(1); %quantile(k_accs(idx,:),0.25);
    K_ACCS(3,II) = muci(2); %quantile(k_accs(idx,:),0.75);
    
    % process all l2 regression results
    [val idx] = max(mean(l_accs,2));
    L_ACCS(2,II) = mean(l_accs(idx,:));
    [muhat,sigmahat,muci,sigmaci] = normfit(l_accs(idx,:));
    L_ACCS(1,II) = muci(1); %quantile(l_accs(idx,:),0.25);
    L_ACCS(3,II) = muci(2); %quantile(l_accs(idx,:),0.75);
    
    % process all sar regression results
    [val idx] = max(mean(y1_accs,2));
    S_ACCS(2,II) = mean(y1_accs(idx,:));
    [muhat,sigmahat,muci,sigmaci] = normfit(y1_accs(idx,:));
    S_ACCS(1,II) = muci(1); %quantile(y1_accs(idx,:),0.25);
    S_ACCS(3,II) = muci(2); %quantile(y1_accs(idx,:),0.75);
    
    % process all additive p-spline regression results
    [val idx] = max(mean(a_accs,2));
    A_ACCS(2,II) = mean(a_accs(idx,:));
    [muhat,sigmahat,muci,sigmaci] = normfit(a_accs(idx,:));
    A_ACCS(1,II) = muci(1); %quantile(y1_accs(idx,:),0.25);
    A_ACCS(3,II) = muci(2); %quantile(y1_accs(idx,:),0.75);
    
    % process all b-spline rbf results
    [val idx] = max(mean(b_accs,2));
    B_ACCS(2,II) = mean(b_accs(idx,:));
    [muhat,sigmahat,muci,sigmaci] = normfit(b_accs(idx,:));
    B_ACCS(1,II) = muci(1); %quantile(y1_accs(idx,:),0.25);
    B_ACCS(3,II) = muci(2); %quantile(y1_accs(idx,:),0.75);
    
    % process all boosted tree regression results
    [val idx] = max(mean(t_accs,2));
    T_ACCS(2,II) = mean(t_accs(idx,:));
    [muhat,sigmahat,muci,sigmaci] = normfit(t_accs(idx,:));
    T_ACCS(1,II) = muci(1); %quantile(t_accs(idx,:),0.25);
    T_ACCS(3,II) = muci(2); %quantile(t_accs(idx,:),0.75);

end

figure();
hold on;

gray = [0.31 0.31 0.31];

% plot(OBS_COUNTS,K_ACCS(2,:),...
%     '-','LineWidth',2.0,'Color',[0.85 0.16 0]);
% 
% plot(OBS_COUNTS,S_ACCS(2,:),...
%     '-','LineWidth',2.0,'Color',[0.17 0.51 0.34]);
% 
% plot(OBS_COUNTS,L_ACCS(2,:),...
%     '-','LineWidth',2.0,'Color',[0 0.75 0.75]);
% 
% plot(OBS_COUNTS,T_ACCS(2,:),...
%     '-','LineWidth',2.0,'Color',[0.75 0.75 0.0]);

h = errorbar(OBS_COUNTS,S_ACCS(2,:),(S_ACCS(1,:)-S_ACCS(2,:)),...
    (S_ACCS(3,:)-S_ACCS(2,:)),'-','LineWidth',2.0,'Color',[0.51 0.77 0.48]);
set(h,'DisplayName','SAR4');

h = errorbar(OBS_COUNTS,L_ACCS(2,:),(L_ACCS(1,:)-L_ACCS(2,:)),...
    (L_ACCS(3,:)-L_ACCS(2,:)),'-','LineWidth',2.0,'Color',[0.16 0.82 0.82]);
set(h,'DisplayName','L2');

h = errorbar(OBS_COUNTS,K_ACCS(2,:),(K_ACCS(1,:)-K_ACCS(2,:)),...
    (K_ACCS(3,:)-K_ACCS(2,:)),'-','LineWidth',2.0,'Color',[0.68 0.14 0.14]);
set(h,'DisplayName','Gauss');

h = errorbar(OBS_COUNTS,T_ACCS(2,:),(T_ACCS(1,:)-T_ACCS(2,:)),...
    (T_ACCS(3,:)-T_ACCS(2,:)),'-','LineWidth',2.0,'Color',[1.00 0.57 0.20]);
set(h,'DisplayName','Boost-Trees');

h = errorbar(OBS_COUNTS,B_ACCS(2,:),(B_ACCS(1,:)-B_ACCS(2,:)),...
    (B_ACCS(3,:)-B_ACCS(2,:)),'-','LineWidth',2.0,'Color',[0.5 0.5 0.5]);
set(h,'DisplayName','B-Spline');

h = errorbar(OBS_COUNTS,A_ACCS(2,:),(A_ACCS(1,:)-A_ACCS(2,:)),...
    (A_ACCS(3,:)-A_ACCS(2,:)),'--','LineWidth',2.0,'Color',[0.5 0.5 0.5]);
set(h,'DisplayName','P-Spline');

xlim([150 450]);

set(gca,'xtick',OBS_COUNTS);
set(gca,'ytick',[0.80 0.85 0.90]);
set(gca,'YTickLabel',{'0.80', '0.85', '0.90'});
set(gca,'FontSize',14);
h = title('Housing: accuracy vs. training samples');
set(h,'FontSize',14);
h = xlabel('Training samples');
set(h,'FontSize',14);
h = ylabel('% variance recovered');
set(h,'FontSize',14);
legend('show','Location','SouthEast');




