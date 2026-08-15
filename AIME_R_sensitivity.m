%% ============================================================
%  AIME learning-rate sensitivity analysis
%
%  Datasets:
%    (1) Iris
%    (2) Synthetic
%
%  Analysis:
%    - Cycle 1 only
%    - 25 learning rates from 1e-6 to 1e-2
%    - 600 neural weight vectors
%    - 40% sparse initialization
%    - 400 epochs
%    - fixed random seed for every learning-rate condition
%
%  Metric:
%    Absolute cosine similarity between the best-aligned
%    post-hoc AIME component and the true PC1.
% ============================================================

clear;
clc;
format shortG;
warning off;


%% ============================================================
%  1. Learning-rate range
% =============================================================

% Broad sensitivity scan
learning_rates = logspace(-6,0,37);

% Ensure the manuscript value is included exactly
learning_rates = unique(sort([learning_rates, 1e-4]));

% Ensure the manuscript value r = 1e-4 is represented exactly
[~,main_lr_idx] = min(abs(log10(learning_rates) - log10(1e-4)));
learning_rates(main_lr_idx) = 1e-4;

n_rates = length(learning_rates);

epoch = 400;
n_dst = 600;
connectivity = 0.4;

iris_alignment = zeros(n_rates,1);
syn_alignment  = zeros(n_rates,1);


%% ============================================================
%  2. Load Iris data
% =============================================================

load fisheriris

iris_inputs_ori = meas(:,1:4);

% Preserve the preprocessing used in the primary Iris experiment
iris_inputs = ...
    iris_inputs_ori - mean(iris_inputs_ori);

[iris_PC,~,~,~,~,~] = ...
    pca(iris_inputs);

iris_PC1 = ...
    iris_PC(:,1) ./ norm(iris_PC(:,1));


%% ============================================================
%  3. Load synthetic data
% =============================================================

syn_table = readtable('syn_2.csv');

syn_inputs_ori = ...
    syn_table(1:300,1:end-1);

syn_inputs_ori = ...
    table2array(syn_inputs_ori);

syn_inputs_ori = ...
    str2double(syn_inputs_ori);

% Preserve the preprocessing used in the primary synthetic experiment
syn_inputs = ...
    syn_inputs_ori - mean(syn_inputs_ori);

[syn_PC,~,~,~,~,~] = ...
    pca(syn_inputs);

syn_PC1 = ...
    syn_PC(:,1) ./ norm(syn_PC(:,1));


%% ============================================================
%  4. Iris sensitivity analysis
% =============================================================

fprintf('\n============================================\n');
fprintf('IRIS LEARNING-RATE SENSITIVITY\n');
fprintf('============================================\n');


for r_idx = 1:n_rates

    lr = learning_rates(r_idx);

    fprintf('Iris: %2d/%2d   r = %.4e\n', ...
        r_idx,n_rates,lr);


    %% --------------------------------------------------------
    % Use the same seed for every learning-rate condition
    % ---------------------------------------------------------

    SetRNG(1);


    %% Initialize neural weight population

    n_src = size(iris_inputs,2);

    n_per_src = ...
        round(n_src * connectivity);

    synaptic_weights_mat = ...
        randn(n_src,n_dst);

    [srcIdx,dstIdx] = ...
        ConnectHypergeometric( ...
        n_dst,n_src,n_per_src);

    index = ...
        [srcIdx;dstIdx];

    for i = 1:n_dst

        nonzero_idx = ...
            index(2,index(1,:) == i);

        zero_idx = ...
            setdiff(1:n_src,nonzero_idx);

        synaptic_weights_mat(zero_idx,i) = 0;

    end

    cells = synaptic_weights_mat;


    %% --------------------------------------------------------
    % Run Cycle-1 AIME
    % ---------------------------------------------------------

    final_weights = ...
        run_aime_cycle1( ...
        iris_inputs, ...
        cells, ...
        epoch, ...
        lr);


    %% --------------------------------------------------------
    % Post-hoc AIME component
    %
    % Cycle 1 contains one dominant bidirectional component,
    % represented by K = 2 antipodal clusters.
    % ---------------------------------------------------------

    iris_alignment(r_idx) = ...
        get_best_pc_alignment( ...
        final_weights, ...
        iris_PC1);

end


%% ============================================================
%  5. Synthetic sensitivity analysis
% =============================================================

fprintf('\n============================================\n');
fprintf('SYNTHETIC LEARNING-RATE SENSITIVITY\n');
fprintf('============================================\n');


for r_idx = 1:n_rates

    lr = learning_rates(r_idx);

    fprintf('Synthetic: %2d/%2d   r = %.4e\n', ...
        r_idx,n_rates,lr);


    %% Same initialization seed for every r

    SetRNG(1);


    %% Initialize neural weight population

    n_src = size(syn_inputs,2);

    n_per_src = ...
        round(n_src * connectivity);

    synaptic_weights_mat = ...
        randn(n_src,n_dst);

    [srcIdx,dstIdx] = ...
        ConnectHypergeometric( ...
        n_dst,n_src,n_per_src);

    index = ...
        [srcIdx;dstIdx];

    for i = 1:n_dst

        nonzero_idx = ...
            index(2,index(1,:) == i);

        zero_idx = ...
            setdiff(1:n_src,nonzero_idx);

        synaptic_weights_mat(zero_idx,i) = 0;

    end

    cells = synaptic_weights_mat;


    %% Run Cycle-1 AIME

    final_weights = ...
        run_aime_cycle1( ...
        syn_inputs, ...
        cells, ...
        epoch, ...
        lr);


    %% Post-hoc PC1 alignment

    syn_alignment(r_idx) = ...
        get_best_pc_alignment( ...
        final_weights, ...
        syn_PC1);

end


%% ============================================================
%  6. Numerical result table
% =============================================================

T = table( ...
    learning_rates', ...
    iris_alignment, ...
    syn_alignment, ...
    'VariableNames', ...
    {'LearningRate', ...
     'Iris_PC1_Alignment', ...
     'Synthetic_PC1_Alignment'});

fprintf('\n============================================\n');
fprintf('LEARNING-RATE SENSITIVITY RESULTS\n');
fprintf('============================================\n');

disp(T);

writetable( ...
    T, ...
    'AIME_learning_rate_sensitivity.csv');


%% ============================================================
%  7. Side-by-side sensitivity figure
% =============================================================

figure;

tiledlayout(1,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');


%% Iris

nexttile;

semilogx( ...
    learning_rates, ...
    iris_alignment, ...
    '-o', ...
    'LineWidth',1.5, ...
    'MarkerSize',4);

hold on;

xline( ...
    1e-4, ...
    '--', ...
    'r = 10^{-4}', ...
    'LabelVerticalAlignment','bottom');

hold off;

xlabel('Learning rate, r');
ylabel('|cosine similarity|');

title('Iris: PC1 recovery');

ylim([0 1.02]);

grid on;

set(gca, ...
    'FontSize',11, ...
    'LineWidth',1.0, ...
    'FontName','Times New Roman');


%% Synthetic

nexttile;

semilogx( ...
    learning_rates, ...
    syn_alignment, ...
    '-o', ...
    'LineWidth',1.5, ...
    'MarkerSize',4);

hold on;

xline( ...
    1e-4, ...
    '--', ...
    'r = 10^{-4}', ...
    'LabelVerticalAlignment','bottom');

hold off;

xlabel('Learning rate, r');
ylabel('|cosine similarity|');

title('Synthetic data: PC1 recovery');

ylim([0 1.02]);

grid on;

set(gca, ...
    'FontSize',11, ...
    'LineWidth',1.0, ...
    'FontName','Times New Roman');


sgtitle( ...
    'Sensitivity of AIME to learning rate');


%% ============================================================
%  8. Report manuscript learning-rate result
% =============================================================

fprintf('\nAt manuscript learning rate r = 1e-4:\n');

fprintf('Iris PC1 alignment      = %.6f\n', ...
    iris_alignment(main_lr_idx));

fprintf('Synthetic PC1 alignment = %.6f\n', ...
    syn_alignment(main_lr_idx));



%% ============================================================
%  LOCAL FUNCTION
%  Run one Cycle-1 AIME simulation
% =============================================================

function final_weights = ...
    run_aime_cycle1(inputs,cells,epoch,lr)

    ori_cells = cells;

    for e = 1:epoch

        sampled_data = ...
            inputs(randperm(size(inputs,1)),:);

        for col = 1:size(sampled_data,1)

            input1 = ...
                sampled_data(col,:)';

            product = ...
                input1' * ori_cells;

            signs = ...
                sign(product);

            winning_idx = ...
                1:length(product);

            winning_cell = ...
                ori_cells(:,winning_idx);

            update_winner = ...
                winning_cell + ...
                (signs.*input1 - winning_cell).*lr;

            ori_cells(:,winning_idx) = ...
                update_winner;

        end

    end

    final_weights = ori_cells;

end



%% ============================================================
%  LOCAL FUNCTION
%  Post-hoc AIME-PC alignment
% =============================================================

function best_alignment = ...
    get_best_pc_alignment(weights,true_pc)

    %% Same centering used in the primary AIME analyses

    new_weight = ...
        weights - mean(weights,2);

    new_weight = ...
        normc(new_weight)';


    %% Cycle 1: one bidirectional component -> K = 2

    % Fix MATLAB K-means randomness so sensitivity reflects r,
    % rather than stochastic K-means initialization.
    rng(1);

    [~,C] = ...
        kmeans( ...
        new_weight, ...
        2, ...
        'Replicates',5);

    C = normr(C);


    %% Absolute cosine similarity
    %
    % The two centroids represent opposite orientations
    % of the same AIME axis.

    true_pc = ...
        true_pc ./ norm(true_pc);

    similarities = ...
        abs(C * true_pc);

    best_alignment = ...
        max(similarities);

end