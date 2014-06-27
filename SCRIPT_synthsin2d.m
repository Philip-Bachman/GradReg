clear;
warning off all;

OBS_COUNTS = zeros(1,11);
KMAX_ACCS = zeros(3,11);
K05_ACCS = zeros(3,11);
K2_ACCS = zeros(3,11);
K8_ACCS = zeros(3,11);
H_ACCS = zeros(3,11);
B_ACCS = zeros(3,11);
L_ACCS = zeros(3,11);

for II=1:11,
    load(sprintf('synthsin2d_final_%d.mat',(45+(II*5))));
    kern_accs = squeeze(max(kern_accs,[],2));
    OBS_COUNTS(II) = sample_count;
    % process all kernel regression results
    [val idx] = max(mean(kern_accs,2));
    KMAX_ACCS(2,II) = mean(kern_accs(idx,:));
    KMAX_ACCS(1,II) = quantile(kern_accs(idx,:),0.25);
    KMAX_ACCS(3,II) = quantile(kern_accs(idx,:),0.75);
    %KMAX_ACCS(1,II) = mean(kern_accs(idx,:)) - std(kern_accs(idx,:));
    %KMAX_ACCS(3,II) = mean(kern_accs(idx,:)) + std(kern_accs(idx,:));
    % gamma = 0.5
    K05_ACCS(1,II) = quantile(kern_accs(1,:),0.25);
    K05_ACCS(2,II) = mean(kern_accs(1,:));
    K05_ACCS(3,II) = quantile(kern_accs(1,:),0.75);
    % gamma = 2.0
    K2_ACCS(1,II) = quantile(kern_accs(2,:),0.25);
    K2_ACCS(2,II) = mean(kern_accs(2,:));
    K2_ACCS(3,II) = quantile(kern_accs(2,:),0.75);
    % gamma = 8.0
    K8_ACCS(1,II) = quantile(kern_accs(3,:),0.25);
    K8_ACCS(2,II) = mean(kern_accs(3,:));
    K8_ACCS(3,II) = quantile(kern_accs(3,:),0.75);
    % process hess-regularized regression results
    [val idx] = max(mean(hess_accs,2));
    H_ACCS(2,II) = mean(hess_accs(idx,:));
    H_ACCS(1,II) = quantile(hess_accs(idx,:),0.25);
    H_ACCS(3,II) = quantile(hess_accs(idx,:),0.75);
    % process (biasedly) hess-regularized regression results
    [val idx] = max(mean(bias_accs,2));
    B_ACCS(2,II) = mean(bias_accs(idx,:));
    B_ACCS(1,II) = quantile(bias_accs(idx,:),0.25);
    B_ACCS(3,II) = quantile(bias_accs(idx,:),0.75);
    %B_ACCS(1,II) = mean(bias_accs(idx,:)) - std(bias_accs(idx,:));
    %B_ACCS(3,II) = mean(bias_accs(idx,:)) + std(bias_accs(idx,:));
    % process L2-regularized regression results
    [val idx] = max(mean(l2_accs,2));
    L_ACCS(2,II) = mean(l2_accs(idx,:));
    L_ACCS(1,II) = quantile(l2_accs(idx,:),0.25);
    L_ACCS(3,II) = quantile(l2_accs(idx,:),0.75);
    %L_ACCS(1,II) = mean(l2_accs(idx,:)) - std(l2_accs(idx,:));
    %L_ACCS(3,II) = mean(l2_accs(idx,:)) + std(l2_accs(idx,:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT PERFORMANCE COMPARISON %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure();
hold on;

% Plot data
gray = [0.4 0.4 0.4];

% line = plot(OBS_COUNTS,K05_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
% hAnnotation = get(line,'Annotation');
% hLegendEntry = get(hAnnotation','LegendInformation');
% set(hLegendEntry,'IconDisplayStyle','off');
% 
% line = plot(OBS_COUNTS,K2_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
% hAnnotation = get(line,'Annotation');
% hLegendEntry = get(hAnnotation','LegendInformation');
% set(hLegendEntry,'IconDisplayStyle','off');
% 
% line = plot(OBS_COUNTS,K8_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
% hAnnotation = get(line,'Annotation');
% hLegendEntry = get(hAnnotation','LegendInformation');
% set(hLegendEntry,'IconDisplayStyle','off');

h = errorbar(OBS_COUNTS,KMAX_ACCS(2,:),(KMAX_ACCS(1,:)-KMAX_ACCS(2,:)),...
    (KMAX_ACCS(3,:)-KMAX_ACCS(2,:)),'LineWidth',2.0,'Color',[0.68 0.14 0.14]);
set(h,'DisplayName','RKHS-RBF (max)');

h = errorbar(OBS_COUNTS,H_ACCS(2,:),(H_ACCS(1,:)-H_ACCS(2,:)),...
    (H_ACCS(3,:)-H_ACCS(2,:)),'LineWidth',2.0,'Color',[0.51 0.77 0.48]);
set(h,'DisplayName','SAR2');

h = errorbar(OBS_COUNTS,B_ACCS(2,:),(B_ACCS(1,:)-B_ACCS(2,:)),...
    (B_ACCS(3,:)-B_ACCS(2,:)),'LineWidth',2.0,'Color',[0.16 0.82 0.82]);
set(h,'DisplayName','SAR2 (biased)');

h = errorbar(OBS_COUNTS,L_ACCS(2,:),(L_ACCS(1,:)-L_ACCS(2,:)),...
    (L_ACCS(3,:)-L_ACCS(2,:)),'LineWidth',2.0,'Color',[1.00 0.57 0.20]);
set(h,'DisplayName','L2');

% Set general axis properties, titles, etc.
xlim([50 100]);
ylim([0.75 1.0]);

set(gca,'xtick',OBS_COUNTS);
set(gca,'ytick',[0.8 0.9 1.0]);
set(gca,'YTickLabel',{'0.8', '0.9', '1.0'});
set(gca,'FontSize',14);
h = title('SynthSin2d: accuracy vs. training samples');
set(h,'FontSize',14);
h = xlabel('Training samples');
set(h,'FontSize',14);
h = ylabel('% variance recovered');
set(h,'FontSize',14);
legend('show','Location','SouthEast');

