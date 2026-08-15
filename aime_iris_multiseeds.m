%% AIME Iris: 100-seed reproducibility analysis
clear;
format shortG;
warning off;

load fisheriris

%% Data preparation
inputs_ori = meas(:,1:4);
inputs = inputs_ori - mean(inputs_ori);

% True PCs of the original Iris dataset
[coeff_real,~,~,~,~,~] = pca(inputs);
coeff_real = normc(coeff_real);

%% Fixed AIME settings
n_seed = 100;
n_dst = 600;
connectivity = 0.4;
epoch = 400;
lr = 1e-4;

% rows = seeds, columns = cycles
aime_pc_similarity = zeros(n_seed,4);

%% 100 independent seeds
for seed = 1:n_seed

    fprintf('Seed %d / %d\n',seed,n_seed);

    %% Initialize neural weight population
    SetRNG(seed);

    dim = size(inputs,2);
    n_src = dim;
    n_per_src = round(n_src*connectivity);

    synaptic_weights_mat = randn(n_src,n_dst);

    [srcIdx,dstIdx] = ConnectHypergeometric( ...
        n_dst,n_src,n_per_src);

    index = [srcIdx;dstIdx];

    for i = 1:n_dst
        nonzero_idx = index(2,index(1,:) == i);
        zero_idx = setdiff(1:n_src,nonzero_idx);
        synaptic_weights_mat(zero_idx,i) = 0;
    end

    cells = synaptic_weights_mat;


    %% =========================================================
    %  Cycle 1
    % ==========================================================
    cycle_inputs = inputs;

    [center1,line1] = run_aime_cycle( ...
        cycle_inputs,cells,epoch,lr);

    aime_pc_similarity(seed,1) = ...
        abs(line1' * coeff_real(:,1));


    %% =========================================================
    %  Cycle 2
    % ==========================================================
    norm_vec_c1 = normc(center1);

    cycle_inputs = ...
        (cycle_inputs' - ...
        norm_vec_c1 * ...
        (cycle_inputs*norm_vec_c1 ./ norm(norm_vec_c1))')';

    cycle_inputs = cycle_inputs - mean(cycle_inputs);

    [center2,line2] = run_aime_cycle( ...
        cycle_inputs,cells,epoch,lr);

    aime_pc_similarity(seed,2) = ...
        abs(line2' * coeff_real(:,2));


    %% =========================================================
    %  Cycle 3
    % ==========================================================
    norm_vec_c2 = normc(center2);

    cycle_inputs = ...
        (cycle_inputs' - ...
        norm_vec_c2 * ...
        (cycle_inputs*norm_vec_c2 ./ norm(norm_vec_c2))')';

    cycle_inputs = cycle_inputs - mean(cycle_inputs);

    [center3,line3] = run_aime_cycle( ...
        cycle_inputs,cells,epoch,lr);

    aime_pc_similarity(seed,3) = ...
        abs(line3' * coeff_real(:,3));


    %% =========================================================
    %  Cycle 4
    % ==========================================================
    norm_vec_c3 = normc(center3);

    cycle_inputs = ...
        (cycle_inputs' - ...
        norm_vec_c3 * ...
        (cycle_inputs*norm_vec_c3 ./ norm(norm_vec_c3))')';

    cycle_inputs = cycle_inputs - mean(cycle_inputs);

    [center4,line4] = run_aime_cycle( ...
        cycle_inputs,cells,epoch,lr);

    aime_pc_similarity(seed,4) = ...
        abs(line4' * coeff_real(:,4));

end


%% =============================================================
%  Summary statistics
% ==============================================================
mean_similarity = mean(aime_pc_similarity,1);
sd_similarity   = std(aime_pc_similarity,0,1);

% 95% confidence interval of the mean
tcrit = tinv(0.975,n_seed-1);
sem = sd_similarity ./ sqrt(n_seed);

ci_low  = mean_similarity - tcrit .* sem;
ci_high = mean_similarity + tcrit .* sem;


%% Summary table
Cycle = (1:4)';
Mean = mean_similarity';
SD = sd_similarity';
CI95_Lower = ci_low';
CI95_Upper = ci_high';

T = table( ...
    Cycle, ...
    Mean, ...
    SD, ...
    CI95_Lower, ...
    CI95_Upper);

disp(T);


%% Optional: full 100 x 4 raw result table
raw_results = array2table( ...
    aime_pc_similarity, ...
    'VariableNames', ...
    {'Cycle1_PC1','Cycle2_PC2','Cycle3_PC3','Cycle4_PC4'});

disp(raw_results);


%% Save results
writetable(T,'AIME_Iris_100seed_summary.csv');
writetable(raw_results,'AIME_Iris_100seed_raw.csv');


%% =============================================================
%  Local function: one AIME learning cycle
% ==============================================================
function [center,line_direction] = ...
    run_aime_cycle(inputs,cells,epoch,lr)

    ori_cells = cells;

    %% AIME training
    for e = 1:epoch

        sampled_data = ...
            inputs(randperm(size(inputs,1)),:);

        for col = 1:size(sampled_data,1)

            input1 = sampled_data(col,:)';

            product = input1' * ori_cells;
            signs = sign(product);

            winning_idx = 1:length(product);
            winning_cell = ori_cells(:,winning_idx);

            update_winner = ...
                winning_cell + ...
                (signs .* input1 - winning_cell) .* lr;

            ori_cells(:,winning_idx) = update_winner;

        end
    end


    %% Direction used for subsequent AIME masking
    bench_v = ones(size(update_winner,1),1);

    id = find( ...
        sign(bench_v' * normc(update_winner)) == 1);

    center = normc(mean(update_winner(:,id),2));

    % Preserve the original implementation behavior
    center = round(center,2);


    %% Post-hoc identification of the AIME component
    new_weight = ...
        update_winner - mean(update_winner,2);

    new_weight = normc(new_weight)';

    [~,C] = kmeans(new_weight,2);

    % Iris produces one bidirectional AIME component.
    % The two K-means centroids therefore represent opposite
    % orientations of the same line. Either centroid defines
    % the same axis after taking absolute cosine similarity.
    line_direction = normc(C(1,:)');

end