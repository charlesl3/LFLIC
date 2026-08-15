%% AIME Synthetic Data: 100-seed reproducibility analysis
clear;
format shortG;
warning off;

%% =============================================================
%  Data preparation
% ==============================================================
inputs_ori = readtable('syn_2.csv');
inputs_ori = inputs_ori(1:300,1:end-1);
inputs_ori = table2array(inputs_ori);
inputs_ori = str2double(inputs_ori);

inputs = inputs_ori - mean(inputs_ori);

% True PCs of the original synthetic dataset
[coeff_real,~,~,~,~,~] = pca(inputs);
coeff_real = normc(coeff_real);


%% =============================================================
%  Fixed AIME settings
% ==============================================================
n_seed = 100;

n_dst = 600;
connectivity = 0.4;

epoch = 400;
lr = 1e-4;

% rows = seeds
% columns = cycles
aime_pc_similarity = zeros(n_seed,4);


%% =============================================================
%  100 independent random seeds
% ==============================================================
for seed = 1:n_seed

    fprintf('Seed %d / %d\n',seed,n_seed);

    %% ---------------------------------------------------------
    %  Initialize neural weight population
    % ----------------------------------------------------------
    SetRNG(seed);

    dim = size(inputs,2);
    n_src = dim;
    n_per_src = round(n_src*connectivity);

    synaptic_weights_mat = randn(n_src,n_dst);

    [srcIdx,dstIdx] = ...
        ConnectHypergeometric(n_dst,n_src,n_per_src);

    index = [srcIdx;dstIdx];

    for i = 1:n_dst

        nonzero_idx = ...
            index(2,index(1,:) == i);

        zero_idx = ...
            setdiff(1:n_src,nonzero_idx);

        synaptic_weights_mat(zero_idx,i) = 0;

    end

    cells = synaptic_weights_mat;


    %% =========================================================
    %  Cycle 1
    % ==========================================================
    cycle1_inputs = inputs;

    [weights1,center1] = ...
        run_aime_cycle( ...
        cycle1_inputs,cells,epoch,lr);

    % Post-hoc K = 2:
    % one bidirectional AIME component
    C1 = identify_components(weights1,2);

    aime_pc_similarity(seed,1) = ...
        best_pc_alignment(C1,coeff_real(:,1));


    %% =========================================================
    %  Cycle 2 input masking
    % ==========================================================
    norm_vec_c1 = normc(center1);

    cycle2_inputs = ...
        (cycle1_inputs' - ...
        norm_vec_c1 * ...
        (cycle1_inputs*norm_vec_c1 ./ ...
        norm(norm_vec_c1))')';

    cycle2_inputs = ...
        cycle2_inputs - mean(cycle2_inputs);


    %% Cycle 2 training
    [weights2,center2] = ...
        run_aime_cycle( ...
        cycle2_inputs,cells,epoch,lr);

    % Post-hoc K = 2:
    % one dominant bidirectional AIME component
    C2 = identify_components(weights2,2);

    aime_pc_similarity(seed,2) = ...
        best_pc_alignment(C2,coeff_real(:,2));


    %% =========================================================
    %  Cycle 3 input masking
    % ==========================================================
    norm_vec_c2 = normc(center2);

    cycle3_inputs = ...
        (cycle2_inputs' - ...
        norm_vec_c2 * ...
        (cycle2_inputs*norm_vec_c2 ./ ...
        norm(norm_vec_c2))')';

    cycle3_inputs = ...
        cycle3_inputs - mean(cycle3_inputs);


    %% Cycle 3 training
    [weights3,~] = ...
        run_aime_cycle( ...
        cycle3_inputs,cells,epoch,lr);

    % K = 4 because Cycle 3 may contain
    % two bidirectional AIME components
    C3 = identify_components(weights3,4);

    % Identify:
    %   (1) the Cycle-3 line most aligned with PC3
    %   (2) the other non-collinear Cycle-3 line
    %
    % This avoids relying on arbitrary K-means cluster labels.
    [c3_pc_line,c3_other_line,c3_best_alignment] = ...
        select_cycle3_lines(C3,coeff_real(:,3));

    aime_pc_similarity(seed,3) = ...
        c3_best_alignment;


    %% =========================================================
    %  Cycle 4 -- branch inherited from PC3-aligned Cycle-3 line
    % ==========================================================
    cycle4_inputs_A = ...
        mask_input(cycle3_inputs,c3_pc_line);

    [weights4_A,~] = ...
        run_aime_cycle( ...
        cycle4_inputs_A,cells,epoch,lr);

    % Original synthetic experiment:
    % this branch produces two bidirectional components
    % -> K = 4
    C4_A = identify_components(weights4_A,4);


    %% =========================================================
    %  Cycle 4 -- branch inherited from the other Cycle-3 line
    % ==========================================================
    cycle4_inputs_B = ...
        mask_input(cycle3_inputs,c3_other_line);

    [weights4_B,~] = ...
        run_aime_cycle( ...
        cycle4_inputs_B,cells,epoch,lr);

    % Original synthetic experiment:
    % this branch produces one bidirectional component
    % -> K = 2
    C4_B = identify_components(weights4_B,2);


    %% ---------------------------------------------------------
    %  Cycle 4 reproducibility metric
    %
    %  Combine all Cycle-4 post-hoc centroids from both branches.
    %  Record only the AIME direction with the strongest
    %  absolute alignment with true PC4.
    % ----------------------------------------------------------
    C4_all = [C4_A;C4_B];

    aime_pc_similarity(seed,4) = ...
        best_pc_alignment(C4_all,coeff_real(:,4));

end


%% =============================================================
%  Summary statistics
% ==============================================================
mean_similarity = ...
    mean(aime_pc_similarity,1);

sd_similarity = ...
    std(aime_pc_similarity,0,1);

% 95% confidence interval of the mean
sem = ...
    sd_similarity ./ sqrt(n_seed);

tcrit = ...
    tinv(0.975,n_seed-1);

ci_low = ...
    mean_similarity - tcrit.*sem;

ci_high = ...
    mean_similarity + tcrit.*sem;


%% =============================================================
%  Summary table
% ==============================================================
Cycle = (1:4)';

Mean = ...
    mean_similarity';

SD = ...
    sd_similarity';

CI95_Lower = ...
    ci_low';

CI95_Upper = ...
    ci_high';

T = table( ...
    Cycle, ...
    Mean, ...
    SD, ...
    CI95_Lower, ...
    CI95_Upper);

disp(T);


%% =============================================================
%  Raw 100 x 4 result table
% ==============================================================
raw_results = ...
    array2table( ...
    aime_pc_similarity, ...
    'VariableNames', ...
    {'Cycle1_PC1', ...
     'Cycle2_PC2', ...
     'Cycle3_PC3', ...
     'Cycle4_PC4'});

disp(raw_results);


%% =============================================================
%  Save results
% ==============================================================
writetable( ...
    T, ...
    'AIME_Synthetic_100seed_summary.csv');

writetable( ...
    raw_results, ...
    'AIME_Synthetic_100seed_raw.csv');


%% =============================================================
%  Local function:
%  Run one complete AIME learning cycle
% ==============================================================
function [final_weights,center] = ...
    run_aime_cycle(inputs,cells,epoch,lr)

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

    final_weights = ...
        update_winner;


    %% Mean direction used for masking in Cycles 1 and 2
    %
    % Preserve the same orientation-selection procedure
    % used in the original implementation.
    bench_v = ...
        ones(size(final_weights,1),1);

    normalized_weights = ...
        normc(final_weights);

    id = find( ...
        sign(bench_v' * normalized_weights) == 1);

    center = ...
        normc(mean(final_weights(:,id),2));

    % Preserve original implementation behavior
    center = round(center,2);

end


%% =============================================================
%  Local function:
%  Post-hoc K-means identification of AIME components
% ==============================================================
function C = ...
    identify_components(weights,K)

    % Center learned weight population
    new_weight = ...
        weights - mean(weights,2);

    % Normalize individual learned vectors
    new_weight = ...
        normc(new_weight)';

    % Post-hoc clustering only
    [~,C] = ...
        kmeans(new_weight,K);

    % Normalize centroid directions
    C = normr(C);

end


%% =============================================================
%  Local function:
%  Best absolute AIME-PC alignment
% ==============================================================
function best_alignment = ...
    best_pc_alignment(C,pc)

    C = normr(C);
    pc = normc(pc);

    alignment = ...
        abs(C * pc);

    best_alignment = ...
        max(alignment);

end


%% =============================================================
%  Local function:
%  Identify the two distinct Cycle-3 AIME axes
% ==============================================================
function [pc_line,other_line,best_alignment] = ...
    select_cycle3_lines(C,pc3)

    C = normr(C);
    pc3 = normc(pc3);

    %% Find the centroid direction most aligned with PC3
    alignment = ...
        abs(C * pc3);

    [best_alignment,best_idx] = ...
        max(alignment);

    pc_line = ...
        C(best_idx,:)';


    %% Because K = 4 represents two approximately antipodal
    %  pairs, find the centroid least collinear with the
    %  PC3-aligned axis. This identifies the second AIME axis
    %  without depending on arbitrary K-means cluster labels.
    collinearity = ...
        abs(C * pc_line);

    collinearity(best_idx) = inf;

    [~,other_idx] = ...
        min(collinearity);

    other_line = ...
        C(other_idx,:)';

end


%% =============================================================
%  Local function:
%  AIME input masking
% ==============================================================
function masked_inputs = ...
    mask_input(inputs,line_direction)

    line_direction = ...
        normc(line_direction);

    masked_inputs = ...
        (inputs' - ...
        line_direction * ...
        (inputs*line_direction ./ ...
        norm(line_direction))')';

    masked_inputs = ...
        masked_inputs - mean(masked_inputs);

end