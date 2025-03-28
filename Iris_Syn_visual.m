clear;
format shortG;
warning off;
load fisheriris;
SetRNG(1);

% Iris data prep
inputs_ori = meas(:, 1:4);
inputs = inputs_ori - mean(inputs_ori);

% PCA on Iris data
[coeff_real,~,~,~,explained_1,~] = pca(inputs);
o = [0 0 0]; % Origin point for arrows
final_all_data = [inputs; coeff_real(:,1)'; coeff_real(:,2)'; coeff_real(:,3)'];
[coeff1,~,~,~,explained,~] = pca(final_all_data);
Z = final_all_data * coeff1(:, 1:3);

% Round coefficients and explained variance for better display
coeff_real = round(coeff_real, 2);
explained_1 = round(explained,1); % this is the explained variances by Iris: 92.2%, 5.4%, 1.8%, 0.5%
explained_1_cum = cumsum(explained_1); % this is the cumulative explained variances by Iris: 92.2%, 97.6%, 99.4%, 99.9%
explained = round(explained); 


% Load synthetic data
inputs_ori = readtable('syn_2.csv');
inputs_ori = inputs_ori(1:300, 1:end-1);
inputs_ori = table2array(inputs_ori);
inputs_ori = str2double(inputs_ori);
inputs = inputs_ori - mean(inputs_ori);

% PCA on synthetic data
[coeff_real_synthetic,~,~,~,explained_synthetic,~] = pca(inputs);
final_all_data_synthetic = [inputs; coeff_real_synthetic(:,1)'; coeff_real_synthetic(:,2)'; coeff_real_synthetic(:,3)'];
[coeff1_synthetic,~,~,~,explained_synthetic,~] = pca(final_all_data_synthetic);
Z_synthetic = final_all_data_synthetic * coeff1_synthetic(:, 1:3);

% Round coefficients and explained variance for synthetic data
coeff_real_synthetic = round(coeff_real_synthetic, 2);
explained_synthetic_1 = round(explained_synthetic,1); % this is the explained variances by synthetic: 31.5%, 12.9%, 8.2%, 6.8%, 6.1%, 4.5%, 2.9%...
explained_synthetic_1_cum = cumsum(explained_synthetic_1); % this is the cumulative explained variances by synthetic: 31.5%, 44.4%, 52.6%, 59.4%, 65.5%, 70%, 72.9%...
explained_synthetic = round(explained_synthetic);

% Create figure with 2 subplots
figure('Renderer', 'painters', 'Position', [100, 100, 800, 600]);

% Subplot 1: Iris Data and PCA
subplot(2, 2, 1);
view(3);
hold on;
plot3(Z(1:end-3, 1), Z(1:end-3, 2), Z(1:end-3, 3), 'r.', 'MarkerSize', 15);
arrow(o, Z(end-2, :) * 5, 'Color', 'b');
arrow(o, Z(end-1, :), 'Color', 'g');
arrow(o, Z(end, :), 'Color', 'r');
xlabel(['PC 1 (' num2str(explained(1)) '% variance expl)']);
ylabel(['PC 2 (' num2str(explained(2)) '% variance expl)']);
zlabel(['PC 3 (' num2str(explained(3)) '% variance expl)']);
xh = get(gca, 'XLabel'); % Handle of the x label
set(xh, 'Units', 'Normalized');
pos = get(xh, 'Position');
set(xh, 'Position', pos .* [1, -0.05, 1], 'Rotation', 15);
yh = get(gca, 'YLabel'); % Handle of the y label
set(yh, 'Units', 'Normalized');
pos = get(yh, 'Position');
set(yh, 'Position', pos .* [1, -0.07, 1], 'Rotation', -25);
title('Iris Data and the First 3 PCs');
legend('data', '1st', '2nd', '3rd');
set(gca, 'FontSize', 15, 'LineWidth', 1.5, 'FontName', 'Times New Roman');
grid on;
hold off;

% Subplot 2: Synthetic Data and PCA
subplot(2, 2, 2);
view(3);
hold on;
plot3(Z_synthetic(1:end-3, 1), Z_synthetic(1:end-3, 2), Z_synthetic(1:end-3, 3), 'r.', 'MarkerSize', 15);
arrow(o, Z_synthetic(end-2, :) * 5000, 'Color', 'b');
arrow(o, Z_synthetic(end-1, :) * 5000, 'Color', 'g');
arrow(o, Z_synthetic(end, :) * 5000, 'Color', 'r');
xlabel(['PC 1 (' num2str(explained_synthetic(1)) '% variance expl)']);
ylabel(['PC 2 (' num2str(explained_synthetic(2)) '% variance expl)']);
zlabel(['PC 3 (' num2str(explained_synthetic(3)) '% variance expl)']);
xh = get(gca, 'XLabel'); % Handle of the x label
set(xh, 'Units', 'Normalized');
pos = get(xh, 'Position');
set(xh, 'Position', pos .* [1, -0.05, 1], 'Rotation', 15);
yh = get(gca, 'YLabel'); % Handle of the y label
set(yh, 'Units', 'Normalized');
pos = get(yh, 'Position');
set(yh, 'Position', pos .* [1, -0.07, 1], 'Rotation', -25);
title('Synthetic Data and the First 3 PCs');
legend('data', '1st', '2nd', '3rd');
set(gca, 'FontSize', 15, 'LineWidth', 1.5, 'FontName', 'Times New Roman');
grid on;
hold off;
