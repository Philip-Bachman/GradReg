clear;
warning off all;

t_sizes = [200 400];

figure();
for tsn=1:numel(t_sizes),
    
    ts = t_sizes(tsn);
    
    load(sprintf('res_housing_ts%d.mat',ts));
    
    % process kernel regression results
    [val idx] = max(mean(k_accs,2));
    K_ACCS = k_accs(idx,:);
    
    % process l2 regression results
    [val idx] = max(mean(l_accs,2));
    L_ACCS = l_accs(idx,:);
    
    % process sar regression results
    [val idx] = max(mean(y1_accs,2));
    S_ACCS = y1_accs(idx,:);
    
    % process boost-trees regression results
    [val idx] = max(mean(t_accs,2));
    T_ACCS = t_accs(idx,:);

    X = 0.7:0.1:1.0;
    
    if (tsn == 2)
        os = 1;
    else
        os = 0;
    end
    
    subplot(2,2,tsn+os); hold on; axis square;
    plot(X,X,'k--','LineWidth',2.0,'Color',[0.25 0.25 0.25]);
    scatter(S_ACCS,K_ACCS,'o','MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 
    
    xlim([min(X) max(X)]); ylim([min(X) max(X)]);
    set(gca,'xtick',X);
    set(gca,'ytick',X);
    set(gca,'FontSize',14);
    h=title(sprintf('%d: SAR4 - RKHS',ts));
    set(h,'FontSize',14);
    
    subplot(2,2,tsn+os+1); hold on; axis square;
    plot(X,X,'k--','LineWidth',2.0,'Color',[0.25 0.25 0.25]); 
    scatter(S_ACCS,T_ACCS,'o','MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 
    xlim([min(X) max(X)]); ylim([min(X) max(X)]);
    set(gca,'xtick',X);
    set(gca,'ytick',X);
    set(gca,'FontSize',14);
    h=title(sprintf('%d: SAR4 - BOOST',ts));
    set(h,'FontSize',14);
    
end

%
% SINGLE SAMPLE SIZE PLOT
%

figure();
hold on;

ts = 300;

load(sprintf('res_housing_ts%d.mat',ts));

% process kernel regression results
[val idx] = max(mean(k_accs,2));
K_ACCS = k_accs(idx,:);

% process l2 regression results
[val idx] = max(mean(l_accs,2));
L_ACCS = l_accs(idx,:);

% process sar regression results
[val idx] = max(mean(y1_accs,2));
S_ACCS = y1_accs(idx,:);

% process boost-trees regression results
[val idx] = max(mean(t_accs,2));
T_ACCS = t_accs(idx,:);

X = 0.75:0.01:0.95;

subplot(2,1,1); hold on; axis square;
plot(X,X,'k--','LineWidth',2.0,'Color',[0.25 0.25 0.25]);
scatter(K_ACCS,S_ACCS,'o','MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 

xlim([min(X) max(X)]); ylim([min(X) max(X)]);
set(gca,'xtick',X);
set(gca,'ytick',X);
set(gca,'FontSize',14);
h=title('300 Samples');
set(h,'FontSize',14);
h = xlabel('RKHS-RBF');
set(h,'FontSize',14);
h = ylabel('SAR4');
set(h,'FontSize',14);

subplot(2,1,2); hold on; axis square;
plot(X,X,'k--','LineWidth',2.0,'Color',[0.25 0.25 0.25]); 
scatter(T_ACCS,S_ACCS,'o','MarkerEdgeColor',[1.00 0.57 0.20],'SizeData',64); 
xlim([min(X) max(X)]); ylim([min(X) max(X)]);
set(gca,'xtick',X);
set(gca,'ytick',X);
set(gca,'FontSize',14);
h = xlabel('BOOST');
set(h,'FontSize',14);
h = ylabel('SAR4');
set(h,'FontSize',14);



