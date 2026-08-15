%% ============================================================
%  LI-HC vs Hierarchical Lloyd K-means
%  Synthetic hierarchical dataset
%
%  Hierarchy:
%    Cycle 1:  4 clusters
%    Cycle 2:  8 clusters
%    Cycle 3: 24 clusters
%
%  Metrics:
%    Purity
%    Normalized Mutual Information (NMI)
%    Rand Index (RI)
%    Silhouette Coefficient (SC)
%    Calinski-Harabasz Index (CHI)
%    Davies-Bouldin Index (DBI)
%
%  IMPORTANT:
%  Each cycle is evaluated against the corresponding
%  ground-truth hierarchy:
%       Cycle 1 -> assign_id1
%       Cycle 2 -> assign_id2
%       Cycle 3 -> assign_id3
% ============================================================

clear;
clc;
format shortG;
warning off;

SetRNG(111);


%% ============================================================
%  1. Generate hierarchical synthetic dataset
% =============================================================

% ----- Hierarchy levels 1 and 2 -----
k = 4;
dim = 100;

mu_range = [-1000 1000];
sigma_centroid = 100;
split_num = 2;

[center1, mu_mat] = ...
    subcluster_centroid( ...
    k, dim, split_num, mu_range, sigma_centroid);


% ----- Hierarchy level 3 -----
center2 = [];

for i = 1:size(center1,1)

    r = ...
        normrnd(0,20,[3,dim]) + center1(i,:);

    center2 = ...
        [center2; r]; %#ok<AGROW>

end

center = center2;


% ----- Generate observations -----
num_per_final_cluster = 100;
sigma_data = 5;

[inputs,~] = ...
    subcluster_simulate( ...
    center, dim, num_per_final_cluster, sigma_data);


%% ============================================================
%  2. Ground-truth hierarchical labels
% =============================================================

% 24 final clusters x 100 samples = 2400 observations

assign_id1 = ...
    repelem((1:4)',600);

assign_id2 = ...
    repelem((1:8)',300);

assign_id3 = ...
    repelem((1:24)',100);

data = inputs;       % samples x dimensions
data_T = inputs';    % dimensions x samples


%% ============================================================
%  3. Initialize LI-HC neural population
% =============================================================

SetRNG(111);

n_src = 100;
n_dst = 2500;
n_per_src = 20;

synaptic_weights_mat = ...
    randn(n_src,n_dst) * 1000;

[srcIdx,dstIdx] = ...
    ConnectHypergeometric( ...
    n_dst,n_src,n_per_src);

index = [srcIdx;dstIdx];

for i = 1:n_dst

    nonzero_idx = ...
        index(2,index(1,:) == i);

    zero_idx = ...
        setdiff(1:n_src,nonzero_idx);

    synaptic_weights_mat(zero_idx,i) = 0;

end

cells = synaptic_weights_mat;


%% ============================================================
%  4. LI-HC — Cycle 1
% =============================================================

cycle1_cells = normc(cells);
cycle1_cells_iter = normc(cells);

ori_cycle1_cells = cells;

cycle1_winners = [];

for epoch = 1:20

    sampled_data = ...
        data_T(:,randperm(size(data_T,2)));

    winner_len_mat = [];

    winner_average = ...
        zeros(n_src,1,n_dst);

    cycle1_winners = [];

    for col = 1:size(sampled_data,2)

        lr = 0.009;

        input1 = ...
            sampled_data(:,col);

        len_input = ...
            norm(input1);

        input1 = ...
            normc(input1);

        product = ...
            input1' * cycle1_cells;

        winning_value = ...
            max(product);

        winning_idx = ...
            find(product == winning_value);

        for cell_idx = winning_idx

            winner = ...
                cycle1_cells_iter(:,cell_idx);

            cycle1_winners = ...
                [cycle1_winners;cell_idx]; %#ok<AGROW>

            update_winner = ...
                winner + ...
                (input1-winner)*lr;

            winner_average(:,:,cell_idx) = ...
                winner_average(:,:,cell_idx) + ...
                update_winner;

            winner_len_mat = ...
                [winner_len_mat; ...
                 cell_idx,len_input]; %#ok<AGROW>

            cycle1_cells_iter(:,cell_idx) = ...
                update_winner;

            cycle1_cells(:,cell_idx) = ...
                normc(update_winner);

        end

    end

end


% Average winner weights
[~,~,ix] = ...
    unique(cycle1_winners,"stable");

winner_stats = ...
    [unique(cycle1_winners,"stable"), ...
     accumarray(ix,1)];

for i = 1:size(winner_stats,1)

    idx_cell = winner_stats(i,1);

    winner_average(:,:,idx_cell) = ...
        winner_average(:,:,idx_cell) ./ ...
        winner_stats(i,2);

    cycle1_cells(:,idx_cell) = ...
        winner_average(:,:,idx_cell);

end


% Assign all inputs to Cycle-1 winner neurons
cycle1_winners = ...
    unique(cycle1_winners,"stable");

[~,idx] = ...
    max( ...
    normr(data) * ...
    cycle1_cells(:,cycle1_winners), ...
    [],2);

lihc_c1_assign = ...
    cycle1_winners(idx);

cycle1_winners = ...
    cycle1_winners(unique(idx,"stable"));


% Restore magnitude information
for c = cycle1_winners'

    each_input = ...
        winner_len_mat( ...
        winner_len_mat(:,1) == c,2);

    ori_cycle1_cells(:,c) = ...
        cycle1_cells(:,c) * ...
        mean(each_input);

end


%% ============================================================
%  5. LI-HC — Cycle 2
% =============================================================

ori_cycle2_cells = ...
    ori_cycle1_cells;

cycle2_cells = ...
    cycle1_cells;

cycle2_cells_iter = ...
    cycle1_cells;

ori_cycle2_cells(:,cycle1_winners) = 0;
cycle2_cells(:,cycle1_winners) = 0;
cycle2_cells_iter(:,cycle1_winners) = 0;

lihc_c2_assign = ...
    zeros(size(data,1),1);

cycle2_winner_mat = [];


for parent_idx = 1:length(cycle1_winners)

    this_winner_idx = ...
        cycle1_winners(parent_idx);

    rawinput_idx = ...
        find( ...
        lihc_c1_assign == this_winner_idx);

    sampled_data_fix = ...
        data_T(:,rawinput_idx);

    this_winner = ...
        ori_cycle1_cells(:,this_winner_idx);

    sampled_data_fix = ...
        sampled_data_fix - this_winner;


    for epoch = 1:20

        lr = 0.02;

        sampled_data = ...
            sampled_data_fix(:, ...
            randperm(size(sampled_data_fix,2)));

        cycle2_winners = [];

        winner_len_mat = [];

        winner_average = ...
            zeros(n_src,1,n_dst);


        for col = 1:size(sampled_data,2)

            input1 = ...
                sampled_data(:,col);

            len_input = ...
                norm(input1);

            input1 = ...
                normc(input1);

            product = ...
                input1' * cycle2_cells;

            winning_value = ...
                max(product);

            winning_idx = ...
                find(product == winning_value);


            for cell_idx = winning_idx

                winning_cell = ...
                    cycle2_cells_iter(:,cell_idx);

                cycle2_winners = ...
                    [cycle2_winners;cell_idx]; %#ok<AGROW>

                winner_len_mat = ...
                    [winner_len_mat; ...
                     cell_idx,len_input]; %#ok<AGROW>

                update_winner = ...
                    winning_cell + ...
                    (input1-winning_cell)*lr;

                winner_average(:,:,cell_idx) = ...
                    winner_average(:,:,cell_idx) + ...
                    update_winner;

                cycle2_cells_iter(:,cell_idx) = ...
                    update_winner;

                cycle2_cells(:,cell_idx) = ...
                    normc(update_winner);

            end

        end

    end


    % Average winners
    [~,~,ix] = ...
        unique(cycle2_winners,"stable");

    winner_stats = ...
        [unique(cycle2_winners,"stable"), ...
         accumarray(ix,1)];

    for i = 1:size(winner_stats,1)

        idx_cell = ...
            winner_stats(i,1);

        winner_average(:,:,idx_cell) = ...
            winner_average(:,:,idx_cell) ./ ...
            winner_stats(i,2);

        cycle2_cells_iter(:,idx_cell) = ...
            winner_average(:,:,idx_cell);

    end


    cycle2_winners = ...
        unique(cycle2_winners);


    % Assign inputs within this parent cluster
    [~,idx] = ...
        max( ...
        normc(sampled_data_fix)' * ...
        cycle2_cells(:,cycle2_winners), ...
        [],2);

    lihc_c2_assign(rawinput_idx) = ...
        cycle2_winners(idx);


    cycle2_winners = ...
        cycle2_winners( ...
        unique(idx,"stable"));


    cycle2_winner_mat = ...
        [cycle2_winner_mat; ...
         cycle2_winners(:)]; %#ok<AGROW>


    % Restore magnitude
    for c = cycle2_winners'

        each_input = ...
            winner_len_mat( ...
            winner_len_mat(:,1) == c,2);

        ori_cycle2_cells(:,c) = ...
            cycle2_cells_iter(:,c) * ...
            mean(each_input);

    end


    % Prevent reuse of already selected neurons
    cycle2_cells(:,cycle2_winners) = 0;
    cycle2_cells_iter(:,cycle2_winners) = 0;

end


%% ============================================================
%  6. LI-HC — Cycle 3
% =============================================================

ori_cycle3_cells = ...
    ori_cycle2_cells;

cycle3_cells = ...
    cycle2_cells;

cycle3_cells_iter = ...
    cycle2_cells;


ori_cycle3_cells(:,cycle2_winner_mat) = 0;
cycle3_cells(:,cycle2_winner_mat) = 0;
cycle3_cells_iter(:,cycle2_winner_mat) = 0;


lihc_c3_assign = ...
    zeros(size(data,1),1);

cycle3_winner_mat = [];


for parent_idx = 1:length(cycle2_winner_mat)

    this_winner_idx = ...
        cycle2_winner_mat(parent_idx);

    rawinput_idx = ...
        find( ...
        lihc_c2_assign == this_winner_idx);

    sampled_data_fix = ...
        data_T(:,rawinput_idx);


    % Parent winner from Cycle 1
    this_idx_1 = ...
        unique( ...
        lihc_c1_assign(rawinput_idx));

    this_winner_1 = ...
        ori_cycle1_cells(:,this_idx_1);

    % Parent winner from Cycle 2
    this_winner_2 = ...
        ori_cycle2_cells(:,this_winner_idx);


    sampled_data_fix = ...
        sampled_data_fix - ...
        this_winner_1 - ...
        this_winner_2;


    for epoch = 1:20

        lr = 0.009;

        sampled_data = ...
            sampled_data_fix(:, ...
            randperm(size(sampled_data_fix,2)));

        winner_average = ...
            zeros(n_src,1,n_dst);

        cycle3_winners = [];

        winner_len_mat = [];


        for col = 1:size(sampled_data,2)

            input1 = ...
                sampled_data(:,col);

            len_input = ...
                norm(input1);

            input1 = ...
                normc(input1);

            product = ...
                input1' * cycle3_cells;

            winning_value = ...
                max(product);

            winning_idx = ...
                find(product == winning_value);


            for cell_idx = winning_idx

                winning_cell = ...
                    cycle3_cells_iter(:,cell_idx);

                cycle3_winners = ...
                    [cycle3_winners;cell_idx]; %#ok<AGROW>

                winner_len_mat = ...
                    [winner_len_mat; ...
                     cell_idx,len_input]; %#ok<AGROW>

                update_winner = ...
                    winning_cell + ...
                    (input1-winning_cell)*lr;

                winner_average(:,:,cell_idx) = ...
                    winner_average(:,:,cell_idx) + ...
                    update_winner;

                cycle3_cells_iter(:,cell_idx) = ...
                    update_winner;

                cycle3_cells(:,cell_idx) = ...
                    normc(update_winner);

            end

        end

    end


    % Average winners
    [~,~,ix] = ...
        unique(cycle3_winners,"stable");

    winner_stats = ...
        [unique(cycle3_winners,"stable"), ...
         accumarray(ix,1)];

    for i = 1:size(winner_stats,1)

        idx_cell = ...
            winner_stats(i,1);

        winner_average(:,:,idx_cell) = ...
            winner_average(:,:,idx_cell) ./ ...
            winner_stats(i,2);

        cycle3_cells_iter(:,idx_cell) = ...
            winner_average(:,:,idx_cell);

    end


    cycle3_winners = ...
        unique(cycle3_winners);


    % Assign observations
    [~,idx] = ...
        max( ...
        normc(sampled_data_fix)' * ...
        cycle3_cells(:,cycle3_winners), ...
        [],2);

    lihc_c3_assign(rawinput_idx) = ...
        cycle3_winners(idx);


    cycle3_winners = ...
        cycle3_winners( ...
        unique(idx,"stable"));


    cycle3_winner_mat = ...
        [cycle3_winner_mat; ...
         cycle3_winners(:)]; %#ok<AGROW>


    % Restore magnitude
    for c = cycle3_winners'

        each_input = ...
            winner_len_mat( ...
            winner_len_mat(:,1) == c,2);

        ori_cycle3_cells(:,c) = ...
            cycle3_cells_iter(:,c) * ...
            mean(each_input);

    end


    % Prevent reuse
    cycle3_cells(:,cycle3_winners) = 0;
    cycle3_cells_iter(:,cycle3_winners) = 0;

end


%% ============================================================
%  7. Hierarchical Lloyd K-means baseline
% =============================================================

% Fix K-means randomness for reproducibility
rng(111);


%% ---------------- Cycle 1: K = 4 ----------------

[km_c1_assign,~] = ...
    kmeans( ...
    data, ...
    4, ...
    'Replicates',10);


%% ---------------- Cycle 2: 2 children / parent ----------------

km_c2_assign = ...
    zeros(size(data,1),1);

parent_clusters = ...
    unique(km_c1_assign);

next_label = 1;

for p = parent_clusters'

    idx_parent = ...
        find(km_c1_assign == p);

    sub_data = ...
        data(idx_parent,:);

    local_assign = ...
        kmeans( ...
        sub_data, ...
        2, ...
        'Replicates',10);

    local_unique = ...
        unique(local_assign);

    for j = 1:length(local_unique)

        idx_local = ...
            idx_parent( ...
            local_assign == local_unique(j));

        km_c2_assign(idx_local) = ...
            next_label;

        next_label = ...
            next_label + 1;

    end

end


%% ---------------- Cycle 3: 3 children / parent ----------------

km_c3_assign = ...
    zeros(size(data,1),1);

parent_clusters = ...
    unique(km_c2_assign);

next_label = 1;

for p = parent_clusters'

    idx_parent = ...
        find(km_c2_assign == p);

    sub_data = ...
        data(idx_parent,:);

    local_assign = ...
        kmeans( ...
        sub_data, ...
        3, ...
        'Replicates',10);

    local_unique = ...
        unique(local_assign);

    for j = 1:length(local_unique)

        idx_local = ...
            idx_parent( ...
            local_assign == local_unique(j));

        km_c3_assign(idx_local) = ...
            next_label;

        next_label = ...
            next_label + 1;

    end

end


%% ============================================================
%  8. Evaluate LI-HC
% =============================================================

LIHC1 = evaluate_clustering( ...
    data, ...
    lihc_c1_assign, ...
    assign_id1);

LIHC2 = evaluate_clustering( ...
    data, ...
    lihc_c2_assign, ...
    assign_id2);

LIHC3 = evaluate_clustering( ...
    data, ...
    lihc_c3_assign, ...
    assign_id3);


%% ============================================================
%  9. Evaluate hierarchical K-means
% =============================================================

KM1 = evaluate_clustering( ...
    data, ...
    km_c1_assign, ...
    assign_id1);

KM2 = evaluate_clustering( ...
    data, ...
    km_c2_assign, ...
    assign_id2);

KM3 = evaluate_clustering( ...
    data, ...
    km_c3_assign, ...
    assign_id3);


%% ============================================================
%  10. Combined results table
% =============================================================

Method = { ...
    'LI-HC'; ...
    'LI-HC'; ...
    'LI-HC'; ...
    'K-means'; ...
    'K-means'; ...
    'K-means'};

Cycle = [1;2;3;1;2;3];

NumClusters = [ ...
    numel(unique(lihc_c1_assign)); ...
    numel(unique(lihc_c2_assign)); ...
    numel(unique(lihc_c3_assign)); ...
    numel(unique(km_c1_assign)); ...
    numel(unique(km_c2_assign)); ...
    numel(unique(km_c3_assign))];


results = [ ...
    LIHC1; ...
    LIHC2; ...
    LIHC3; ...
    KM1; ...
    KM2; ...
    KM3];


T = table( ...
    Method, ...
    Cycle, ...
    NumClusters, ...
    results(:,1), ...
    results(:,2), ...
    results(:,3), ...
    results(:,4), ...
    results(:,5), ...
    results(:,6), ...
    'VariableNames', ...
    {'Method', ...
     'Cycle', ...
     'Clusters', ...
     'Purity', ...
     'NMI', ...
     'RI', ...
     'SC', ...
     'CHI', ...
     'DBI'});


fprintf('\n');
fprintf('============================================================\n');
fprintf('LI-HC vs hierarchical Lloyd K-means\n');
fprintf('============================================================\n');

disp(T);


%% ============================================================
%  11. Optional method-level summary
% =============================================================

LIHC_mean = ...
    mean(results(1:3,:),1);

KM_mean = ...
    mean(results(4:6,:),1);


SummaryMethod = ...
    {'LI-HC'; 'K-means'};

Summary = table( ...
    SummaryMethod, ...
    [LIHC_mean(1); KM_mean(1)], ...
    [LIHC_mean(2); KM_mean(2)], ...
    [LIHC_mean(3); KM_mean(3)], ...
    [LIHC_mean(4); KM_mean(4)], ...
    [LIHC_mean(5); KM_mean(5)], ...
    [LIHC_mean(6); KM_mean(6)], ...
    'VariableNames', ...
    {'Method', ...
     'MeanPurity', ...
     'MeanNMI', ...
     'MeanRI', ...
     'MeanSC', ...
     'MeanCHI', ...
     'MeanDBI'});


fprintf('\n');
fprintf('============================================================\n');
fprintf('Mean metrics across the three hierarchical cycles\n');
fprintf('============================================================\n');

disp(Summary);


%% Save tables
writetable( ...
    T, ...
    'LIHC_vs_Kmeans_by_cycle.csv');

writetable( ...
    Summary, ...
    'LIHC_vs_Kmeans_summary.csv');


%% ============================================================
%  LOCAL FUNCTION
%  Compute clustering metrics
% =============================================================

function metrics = ...
    evaluate_clustering(data,assigned_cluster,true_label)

    assigned_cluster = ...
        assigned_cluster(:);

    true_label = ...
        true_label(:);


    % External metrics
    PTY = ...
        purity( ...
        assigned_cluster, ...
        true_label);

    NMI = ...
        nmi( ...
        true_label, ...
        assigned_cluster);

    [RI,~] = ...
        randindex( ...
        true_label, ...
        assigned_cluster);


    % Internal metrics
    SC = ...
        ClusterEvalSilhouette( ...
        data, ...
        assigned_cluster, ...
        'cosine');

    CHI = ...
        ClusterEvalCalinskiHarabasz( ...
        data, ...
        assigned_cluster);

    DBI = ...
        ClusterEvalDaviesBouldin( ...
        data, ...
        assigned_cluster);


    metrics = ...
        [PTY,NMI,RI,SC,CHI,DBI];

end