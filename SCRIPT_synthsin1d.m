clear;
warning off all;

OBS_COUNTS = zeros(1,11);
KMAX_ACCS = zeros(3,11);
K2_ACCS = zeros(3,11);
K4_ACCS = zeros(3,11);
K8_ACCS = zeros(3,11);
K16_ACCS = zeros(3,11);
H_ACCS = zeros(3,11);
L_ACCS = zeros(3,11);

for II=1:11,
    load(sprintf('synthsin1d_final_%d.mat',(45+(II*5))));
    OBS_COUNTS(II) = sample_count;
    % process all kernel regression results
    [val idx] = max(mean(kern_accs,2));
    KMAX_ACCS(2,II) = mean(kern_accs(idx,:));
    KMAX_ACCS(1,II) = quantile(kern_accs(idx,:),0.25);
    KMAX_ACCS(3,II) = quantile(kern_accs(idx,:),0.75);
    %KMAX_ACCS(1,II) = mean(kern_accs(idx,:)) - std(kern_accs(idx,:));
    %KMAX_ACCS(3,II) = mean(kern_accs(idx,:)) + std(kern_accs(idx,:));
    % gamma = 2
    K2_ACCS(1,II) = quantile(kern_accs(1,:),0.25);
    K2_ACCS(2,II) = mean(kern_accs(1,:));
    K2_ACCS(3,II) = quantile(kern_accs(1,:),0.75);
    % gamma = 4
    K4_ACCS(1,II) = quantile(kern_accs(2,:),0.25);
    K4_ACCS(2,II) = mean(kern_accs(2,:));
    K4_ACCS(3,II) = quantile(kern_accs(2,:),0.75);
    % gamma = 8
    K8_ACCS(1,II) = quantile(kern_accs(3,:),0.25);
    K8_ACCS(2,II) = mean(kern_accs(3,:));
    K8_ACCS(3,II) = quantile(kern_accs(3,:),0.75);
    % gamma = 16
    K16_ACCS(1,II) = quantile(kern_accs(4,:),0.25);
    K16_ACCS(2,II) = mean(kern_accs(4,:));
    K16_ACCS(3,II) = quantile(kern_accs(4,:),0.75);
    % process hess-regularized regression results
    [val idx] = max(mean(hess_accs,2));
    H_ACCS(2,II) = mean(hess_accs(idx,:));
    H_ACCS(1,II) = quantile(hess_accs(idx,:),0.25);
    H_ACCS(3,II) = quantile(hess_accs(idx,:),0.75);
    %H_ACCS(1,II) = mean(hess_accs(idx,:)) - std(hess_accs(idx,:));
    %H_ACCS(3,II) = mean(hess_accs(idx,:)) + std(hess_accs(idx,:));
    % process L2-regularized regression results
    [val idx] = max(mean(l2_accs,2));
    L_ACCS(2,II) = mean(l2_accs(idx,:));
    L_ACCS(1,II) = quantile(l2_accs(idx,:),0.25);
    L_ACCS(3,II) = quantile(l2_accs(idx,:),0.75);
    %L_ACCS(1,II) = mean(l2_accs(idx,:)) - std(l2_accs(idx,:));
    %L_ACCS(3,II) = mean(l2_accs(idx,:)) + std(l2_accs(idx,:));
end

clear RME FFh FFk Kh Kk SVMh SVMk;

figure();
hold on;

% plot(OBS_COUNTS,KMAX_ACCS(1,:),'r--','LineWidth',0.5);
% plot(OBS_COUNTS,KMAX_ACCS(2,:),'r-','LineWidth',2.0);
% plot(OBS_COUNTS,KMAX_ACCS(3,:),'r--','LineWidth',0.5);
% 
% plot(OBS_COUNTS,H_ACCS(1,:),'b--','LineWidth',0.5);
% plot(OBS_COUNTS,H_ACCS(2,:),'b-','LineWidth',2.0);
% plot(OBS_COUNTS,H_ACCS(3,:),'b--','LineWidth',0.5);
% 
% plot(OBS_COUNTS,L_ACCS(1,:),'g--','LineWidth',0.5);
% plot(OBS_COUNTS,L_ACCS(2,:),'g-','LineWidth',2.0);
% plot(OBS_COUNTS,L_ACCS(3,:),'g--','LineWidth',0.5);

gray = [0.4 0.4 0.4];

line = plot(OBS_COUNTS,K2_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
hAnnotation = get(line,'Annotation');
hLegendEntry = get(hAnnotation','LegendInformation');
set(hLegendEntry,'IconDisplayStyle','off');

line = plot(OBS_COUNTS,K4_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
hAnnotation = get(line,'Annotation');
hLegendEntry = get(hAnnotation','LegendInformation');
set(hLegendEntry,'IconDisplayStyle','off');

line = plot(OBS_COUNTS,K8_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
hAnnotation = get(line,'Annotation');
hLegendEntry = get(hAnnotation','LegendInformation');
set(hLegendEntry,'IconDisplayStyle','off');


%hAnnotation = get(line,'Annotation');
%hLegendEntry = get(hAnnotation','LegendInformation');
%set(hLegendEntry,'IconDisplayStyle','off');

h = errorbar(OBS_COUNTS,KMAX_ACCS(2,:),(KMAX_ACCS(1,:)-KMAX_ACCS(2,:)),...
    (KMAX_ACCS(3,:)-KMAX_ACCS(2,:)),'LineWidth',2.0,'Color',[0.68 0.14 0.14]);
set(h,'DisplayName','RKHS-RBF (max)');

line = plot(OBS_COUNTS,K16_ACCS(2,:),'k--','LineWidth',2.0,'Color',gray);
set(line,'DisplayName','RKHS-RBF (each)');

h = errorbar(OBS_COUNTS,H_ACCS(2,:),(H_ACCS(1,:)-H_ACCS(2,:)),...
    (H_ACCS(3,:)-H_ACCS(2,:)),'LineWidth',2.0,'Color',[0.51 0.77 0.48]);
set(h,'DisplayName','SAR2');

h = errorbar(OBS_COUNTS,L_ACCS(2,:),(L_ACCS(1,:)-L_ACCS(2,:)),...
    (L_ACCS(3,:)-L_ACCS(2,:)),'LineWidth',2.0,'Color',[1.00 0.57 0.20]);
set(h,'DisplayName','L2');

xlim([50 100]);
ylim([0.7 1.0])

set(gca,'xtick',OBS_COUNTS);
set(gca,'ytick',[0.7 0.8 0.9 1.0]);
set(gca,'YTickLabel',{'0.7', '0.8', '0.9', '1.0'});
set(gca,'FontSize',14);
h = title('SynthSin1d: accuracy vs. training samples');
set(h,'FontSize',14);
h = xlabel('Training samples');
set(h,'FontSize',14);
h = ylabel('% variance recovered');
set(h,'FontSize',14);
legend('show','Location','SouthEast');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCATTER PLOT @ 75 SAMPLES %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('synthsin1d_final_75.mat');
[val idx] = max(mean(hess_accs,2));
X = 0.6:0.1:1.0;
Xt = cell(1,numel(X));
for i=1:numel(X),
    Xt{i} = sprintf('%.1f',X(i));
end

figure();
hold on;
h = scatter(kern_accs(1,:),hess_accs(idx,:),'s',...
    'MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 
set(h,'DisplayName','RBF-2');
h = scatter(kern_accs(2,:),hess_accs(idx,:),'^',...
    'MarkerEdgeColor',[0.51 0.77 0.48],'SizeData',64); 
set(h,'DisplayName','RBF-4');
h = scatter(kern_accs(3,:),hess_accs(idx,:),'v',...
    'MarkerEdgeColor',[0.16 0.82 0.82],'SizeData',64); 
set(h,'DisplayName','RBF-8');
h = scatter(kern_accs(4,:),hess_accs(idx,:),'o',...
    'MarkerEdgeColor',[1.00 0.57 0.20],'SizeData',64); 
set(h,'DisplayName','RBF-16');
h = plot(X,X,'k--','LineWidth',2.0,'Color',[0.25 0.25 0.25]);
set(h,'DisplayName','equality');

xlim([min(X) max(X)]); ylim([min(X) max(X)]);
axis square;
set(gca,'xtick',X);
set(gca,'XTickLabel',Xt);
set(gca,'ytick',X);
set(gca,'YTickLabel',Xt);
set(gca,'FontSize',14);
h = xlabel('RKHS accuracy');
set(h,'FontSize',14);
h = ylabel('SAR2 accuracy');
set(h,'FontSize',14);
h=title('SynthSin1d: 75 samples');
set(h,'FontSize',14);
legend('show','Location','SouthEast');



