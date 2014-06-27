clear;

load('result_approx_ss2d');

nz_std = 0.075;
show_count = 16;

% Get a "jittered" sample count for each matrix divergence measurement
scatter_scounts = repmat(sample_counts',1,size(mat_divr,2));
scatter_scounts = log(scatter_scounts) + (nz_std*randn(size(scatter_scounts)));
scatter_scounts = exp(scatter_scounts);

% Elide measurements for the converged matrices (their divergence is invalid)
scatter_scounts(end,:) = [];
mat_divr(end,:) = [];
fun_diff_fin(end,:) = [];
fun_diff_opt(end,:) = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO PLOTTING FOR REGULARIZER MATRIX DIVERGENCE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scat it up, a zippity do bop do wop, man.
figure();
hold on;

% Compute a mean before paring down samples
mean_divr = mean(mat_divr,2);
show_points = zeros(size(mat_divr,1),show_count);
for i=1:size(mat_divr,1);
    idx = randsample(size(mat_divr,2),show_count);
    show_points(i,:) = mat_divr(i,idx);
end
show_scounts = scatter_scounts(:,1:show_count);

% Scatter the divergences
scatter(show_scounts(:),show_points(:),...
    'o','MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 
% Plot the mean divergences
plot(mean(scatter_scounts,2),mean_divr,...
    '-','LineWidth',4.0,'Color',[0.16 0.82 0.82]);
%[0.51 0.77 0.48]

% Compute the theoretical convergence rate (up to constant factors)
theory_rate = 1 ./ sqrt(sample_counts);
% Scale the rate by a constant, to get it started near the divergences
theory_rate = theory_rate * (mean(mat_divr(1,:)) / (1/sqrt(sample_counts(1))));

% Plot the theoretical convergence rate
plot(sample_counts,theory_rate,'-','LineWidth',4.0,'Color',[0.25 0.25 0.25]);

xlim([min(scatter_scounts(:)) max(scatter_scounts(:))]);
ylim([min(mat_divr(:)) 1]);

set(gca,'YScale','log','YMinorTick','on','XScale','log','XMinorTick','on');
set(gca,'FontSize',14);
h=title('SynthSin2d matrix convergence');
set(h,'FontSize',14);
h = xlabel('Approximation samples');
set(h,'FontSize',14);
h = ylabel('Matrix divergence');
set(h,'FontSize',14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO PLOTTING FOR INDUCED FUNCTION DIVERGENCE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scat it up, a zippity do bop do wop, man.
figure();
hold on;

% Compute a mean before paring down samples
mean_diff = mean(fun_diff_fin,2);
show_points = zeros(size(fun_diff_fin,1),show_count);
for i=1:size(fun_diff_fin,1);
    idx = randsample(size(fun_diff_fin,2),show_count);
    show_points(i,:) = fun_diff_fin(i,idx);
end
show_scounts = scatter_scounts(:,1:show_count);
% Scatter the divergences
scatter(show_scounts(:),show_points(:),...
    'o','MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 
% Plot the mean divergences
plot(mean(scatter_scounts,2),mean_diff,...
    '-','LineWidth',4.0,'Color',[0.16 0.82 0.82]);
%[0.51 0.77 0.48]

% Compute the theoretical convergence rate (up to constant factors)
theory_rate = 1 ./ sqrt(sample_counts);
% Scale the rate by a constant, to get it started near the divergences
theory_rate = theory_rate*(mean(fun_diff_fin(1,:))/(1/sqrt(sample_counts(1))));

% Plot the theoretical convergence rate
plot(sample_counts,theory_rate,'-','LineWidth',4.0,'Color',[0.25 0.25 0.25]);

xlim([min(scatter_scounts(:)) max(scatter_scounts(:))]);
ylim([min(fun_diff_fin(:)) max(fun_diff_fin(:))]);

set(gca,'YScale','log','YMinorTick','on','XScale','log','XMinorTick','on');
set(gca,'FontSize',14);
h=title('SynthSin2d function convergence');
set(h,'FontSize',14);
h = xlabel('Approximation samples');
set(h,'FontSize',14);
h = ylabel('Function divergence');
set(h,'FontSize',14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO PLOTTING FOR OPTIMUM FUNCTION DIVERGENCE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scat it up, a zippity do bop do wop, man.
figure();
hold on;

% Compute a mean before paring down samples
mean_diff = mean(fun_diff_opt,2);
show_points = zeros(size(fun_diff_opt,1),show_count);
for i=1:size(fun_diff_opt,1);
    idx = randsample(size(fun_diff_opt,2),show_count);
    show_points(i,:) = fun_diff_opt(i,idx);
end
show_scounts = scatter_scounts(:,1:show_count);
% Scatter the divergences
scatter(show_scounts(:),show_points(:),...
    'o','MarkerEdgeColor',[0.68 0.14 0.14],'SizeData',64); 
% Plot the mean divergences
plot(mean(scatter_scounts,2),mean_diff,...
    '-','LineWidth',4.0,'Color',[0.16 0.82 0.82]);
%[0.51 0.77 0.48]

% Compute the theoretical convergence rate (up to constant factors)
theory_rate = 1 ./ sqrt(sample_counts);
% Scale the rate by a constant, to get it started near the divergences
theory_rate = theory_rate*(mean(fun_diff_opt(1,:))/(1/sqrt(sample_counts(1))));

theory_rate = mean(1-l2_accs)*ones(size(sample_counts));

% Plot the theoretical convergence rate
plot(sample_counts,theory_rate,'-','LineWidth',4.0,'Color',[0.25 0.25 0.25]);

xlim([min(scatter_scounts(:)) max(scatter_scounts(:))]);
ylim([min(fun_diff_opt(:)) max(fun_diff_opt(:))]);

set(gca,'YScale','log','YMinorTick','on','XScale','log','XMinorTick','on');
set(gca,'FontSize',14);
h=title('SynthSin2d optimum convergence');
set(h,'FontSize',14);
h = xlabel('Approximation samples');
set(h,'FontSize',14);
h = ylabel('Function divergence');
set(h,'FontSize',14);

