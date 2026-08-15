%% ============================================================
%  LI-HC Synthetic Hierarchical Demonstration
%
%  Purpose:
%  Reproduce the synthetic LI-HC hierarchy and generate:
%
%    Figure 1: Representative 3D LI-HC hierarchy
%    Figure 2: Dendrogram of learned LI-HC centroids
%    Figure 3: Dendrogram of original synthetic data
%
%  Synthetic hierarchy:
%    Level 1:  4 clusters
%    Level 2:  8 clusters
%    Level 3: 24 clusters
%
%  LI-HC:
%    Cycle 1 -> coarse clusters
%    Cycle 2 -> sub-clusters
%    Cycle 3 -> finest sub-clusters
% ============================================================

clear;
clc;
format shortG;
warning off;

SetRNG(111);


%% ============================================================
%  1. Generate hierarchical synthetic data
% =============================================================

% ------------------------------------------------------------
% Hierarchical levels 1 and 2
% ------------------------------------------------------------

k = 4;
dim = 100;

mu_range = [-1000 1000];
sigma_parent = 100;
split_num = 2;

[center1, mu_mat] = ...
    subcluster_centroid( ...
    k, ...
    dim, ...
    split_num, ...
    mu_range, ...
    sigma_parent);


% ------------------------------------------------------------
% Hierarchical level 3
%
% Each of the 8 level-2 centroids is split into 3 children,
% producing 24 finest-level centroids.
% ------------------------------------------------------------

center2 = [];

for i = 1:size(center1,1)

    r = ...
        normrnd(0,20,[3,dim]) + ...
        center1(i,:);

    center2 = ...
        [center2; r]; %#ok<AGROW>

end

center = center2;


% ------------------------------------------------------------
% Generate observations around the 24 finest centroids
% ------------------------------------------------------------

num_per_cluster = 100;
sigma_data = 5;

[inputs,~] = ...
    subcluster_simulate( ...
    center, ...
    dim, ...
    num_per_cluster, ...
    sigma_data);


% Ground-truth hierarchical labels
assign_id1 = ...
    repelem((1:4)',600);

assign_id2 = ...
    repelem((1:8)',300);

assign_id3 = ...
    repelem((1:24)',100);


% LI-HC code operates on dimensions x observations
data = inputs';


fprintf('Synthetic dataset generated.\n');
fprintf('Samples:    %d\n',size(inputs,1));
fprintf('Dimensions: %d\n',size(inputs,2));
fprintf('Hierarchy:  4 -> 8 -> 24 clusters\n');


%% ============================================================
%  2. Initialize LI-HC neural population
% =============================================================

SetRNG(111);

n_src = 100;
n_dst = 2500;

n_per_src = 20;

synaptic_weights_mat = ...
    randn(n_src,n_dst) * 1000;


[srcIdx,dstIdx] = ...
    ConnectHypergeometric( ...
    n_dst, ...
    n_src, ...
    n_per_src);


index = ...
    [srcIdx;dstIdx];


for i = 1:n_dst

    nonzero_idx = ...
        index(2,index(1,:) == i);

    zero_idx = ...
        setdiff(1:n_src,nonzero_idx);

    synaptic_weights_mat(zero_idx,i) = 0;

end


cells = ...
    synaptic_weights_mat;


% Store cluster assignments for each LI-HC cycle
cluster_assign_cycle = ...
    zeros(size(data,2),3);


%% ============================================================
%  3. LI-HC Cycle 1
% =============================================================

fprintf('\nRunning LI-HC Cycle 1...\n');

cycle = 1;

ori_cycle1_cells = ...
    cells;

cycle1_cells = ...
    normc(cells);

cycle1_cells_iter = ...
    normc(cells);


for epoch = 1:20

    cycle1_winners = [];

    sampled_data = ...
        data(:,randperm(size(data,2)));

    winner_len_mat = [];

    winner_average = ...
        zeros(n_src,1,n_dst);


    for col = 1:size(sampled_data,2)

        lr = 0.009;

        input1 = ...
            sampled_data(:,col);

        len_input = ...
            norm(input1);

        input1 = ...
            normc(input1);


        % ----------------------------------------------------
        % Competitive winner selection
        % ----------------------------------------------------

        product = ...
            input1' * cycle1_cells;

        winning_value = ...
            max(product);

        winning_idx = ...
            find(product == winning_value);


        % ----------------------------------------------------
        % Update winning neuron
        % ----------------------------------------------------

        for cell_idx = winning_idx

            winner = ...
                cycle1_cells_iter(:,cell_idx);

            cycle1_winners = ...
                [cycle1_winners;cell_idx]; %#ok<AGROW>


            update_winner_ori = ...
                winner + ...
                (input1-winner)*lr;


            winner_average(:,:,cell_idx) = ...
                winner_average(:,:,cell_idx) + ...
                update_winner_ori;


            winner_len_mat = ...
                [winner_len_mat; ...
                 cell_idx,len_input]; %#ok<AGROW>


            cycle1_cells_iter(:,cell_idx) = ...
                update_winner_ori;


            cycle1_cells(:,cell_idx) = ...
                update_winner_ori ./ ...
                norm(update_winner_ori);

        end

    end

end


% ------------------------------------------------------------
% Average learned winner vectors
% ------------------------------------------------------------

[~,~,ix] = ...
    unique(cycle1_winners,"stable");

winner_stats = ...
    [unique(cycle1_winners,"stable"), ...
     accumarray(ix,1)];


for i = 1:size(winner_stats,1)

    cell_idx = ...
        winner_stats(i,1);

    winner_average(:,:,cell_idx) = ...
        winner_average(:,:,cell_idx) ./ ...
        winner_stats(i,2);

    cycle1_cells(:,cell_idx) = ...
        winner_average(:,:,cell_idx);

end


% ------------------------------------------------------------
% Assign every observation to a Cycle-1 winner
% ------------------------------------------------------------

cycle1_winners = ...
    unique(cycle1_winners,"stable");


[~,idx] = ...
    max( ...
    normr(inputs) * ...
    cycle1_cells(:,cycle1_winners), ...
    [],2);


cluster_assign_cycle(:,cycle) = ...
    cycle1_winners(idx);


cycle1_winners = ...
    cycle1_winners(unique(idx,"stable"));


% ------------------------------------------------------------
% Restore winner magnitudes
% ------------------------------------------------------------

for c = cycle1_winners'

    each_input = ...
        winner_len_mat( ...
        winner_len_mat(:,1) == c,2);

    ori_cycle1_cells(:,c) = ...
        cycle1_cells(:,c) * ...
        mean(each_input);

end


cycle1_winner_demask = ...
    ori_cycle1_cells(:,cycle1_winners);


fprintf('Cycle 1 recovered clusters: %d\n', ...
    length(cycle1_winners));


%% ============================================================
%  4. LI-HC Cycle 2
% =============================================================

fprintf('\nRunning LI-HC Cycle 2...\n');

cycle = 2;

ori_cycle2_cells = ...
    ori_cycle1_cells;

cycle2_cells = ...
    cycle1_cells;

cycle2_cells_iter = ...
    cycle1_cells;


% Prevent reuse of Cycle-1 winners
ori_cycle2_cells(:,cycle1_winners) = 0;
cycle2_cells(:,cycle1_winners) = 0;
cycle2_cells_iter(:,cycle1_winners) = 0;


cycle2_winner_mat = [];

% Needed later to reconstruct full-space centroids
cycle2_parent_mat = [];


for parent_idx = 1:length(cycle1_winners)

    lr = 0.02;


    % --------------------------------------------------------
    % Select current parent cluster
    % --------------------------------------------------------

    this_winner_idx = ...
        cycle1_winners(parent_idx);


    rawinput_idx = ...
        find( ...
        cluster_assign_cycle(:,cycle-1) == ...
        this_winner_idx);


    sampled_data_fix = ...
        data(:,rawinput_idx);


    this_winner = ...
        ori_cycle1_cells(:,this_winner_idx);


    % Residualize by parent centroid
    sampled_data_fix = ...
        sampled_data_fix - ...
        this_winner;


    % Two Cycle-2 children expected for each Cycle-1 parent
    cycle2_parent_mat = ...
        [cycle2_parent_mat; ...
         this_winner'; ...
         this_winner']; %#ok<AGROW>


    % --------------------------------------------------------
    % Train within current parent cluster
    % --------------------------------------------------------

    for epoch = 1:20

        winner_len_mat = [];
        cycle2_winners = [];

        sampled_data = ...
            sampled_data_fix(:, ...
            randperm(size(sampled_data_fix,2)));

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


    % --------------------------------------------------------
    % Average winners
    % --------------------------------------------------------

    [~,~,ix] = ...
        unique(cycle2_winners,"stable");


    winner_stats = ...
        [unique(cycle2_winners,"stable"), ...
         accumarray(ix,1)];


    for i = 1:size(winner_stats,1)

        cell_idx = ...
            winner_stats(i,1);

        winner_average(:,:,cell_idx) = ...
            winner_average(:,:,cell_idx) ./ ...
            winner_stats(i,2);

        cycle2_cells_iter(:,cell_idx) = ...
            winner_average(:,:,cell_idx);

    end


    cycle2_winners = ...
        unique(cycle2_winners);


    % --------------------------------------------------------
    % Assign parent observations to Cycle-2 winners
    % --------------------------------------------------------

    [~,idx] = ...
        max( ...
        normc(sampled_data_fix)' * ...
        cycle2_cells(:,cycle2_winners), ...
        [],2);


    cluster_assign_cycle(rawinput_idx,cycle) = ...
        cycle2_winners(idx);


    cycle2_winners = ...
        cycle2_winners(unique(idx,"stable"));


    cycle2_winner_mat = ...
        [cycle2_winner_mat; ...
         cycle2_winners(:)]; %#ok<AGROW>


    % --------------------------------------------------------
    % Restore magnitudes
    % --------------------------------------------------------

    for c = cycle2_winners'

        each_input = ...
            winner_len_mat( ...
            winner_len_mat(:,1) == c,2);

        ori_cycle2_cells(:,c) = ...
            cycle2_cells_iter(:,c) * ...
            mean(each_input);

    end


    % Prevent winner reuse in later branches
    cycle2_cells(:,cycle2_winners) = 0;
    cycle2_cells_iter(:,cycle2_winners) = 0;

end


% Reconstruct Cycle-2 centroids in original coordinate space
cycle2_winner_demask = ...
    ori_cycle2_cells(:,cycle2_winner_mat) + ...
    cycle2_parent_mat';


fprintf('Cycle 2 recovered clusters: %d\n', ...
    length(cycle2_winner_mat));


%% ============================================================
%  5. LI-HC Cycle 3
% =============================================================

fprintf('\nRunning LI-HC Cycle 3...\n');

cycle = 3;

ori_cycle3_cells = ...
    ori_cycle2_cells;

cycle3_cells = ...
    cycle2_cells;

cycle3_cells_iter = ...
    cycle2_cells;


% Prevent reuse of Cycle-2 winners
ori_cycle3_cells(:,cycle2_winner_mat) = 0;
cycle3_cells(:,cycle2_winner_mat) = 0;
cycle3_cells_iter(:,cycle2_winner_mat) = 0;


cycle3_winner_mat = [];

cycle3_parent1_mat = [];
cycle3_parent2_mat = [];


for parent_idx = 1:length(cycle2_winner_mat)

    this_winner_idx = ...
        cycle2_winner_mat(parent_idx);


    rawinput_idx = ...
        find( ...
        cluster_assign_cycle(:,cycle-1) == ...
        this_winner_idx);


    sampled_data_fix = ...
        data(:,rawinput_idx);


    % --------------------------------------------------------
    % Identify parent centroids from Cycles 1 and 2
    % --------------------------------------------------------

    this_idx_1 = ...
        unique( ...
        cluster_assign_cycle(rawinput_idx,1));


    this_winner_1 = ...
        ori_cycle1_cells(:,this_idx_1);


    this_winner_2 = ...
        ori_cycle2_cells(:,this_winner_idx);


    % Three Cycle-3 children expected per Cycle-2 parent
    cycle3_parent1_mat = ...
        [cycle3_parent1_mat; ...
         this_winner_1'; ...
         this_winner_1'; ...
         this_winner_1']; %#ok<AGROW>


    cycle3_parent2_mat = ...
        [cycle3_parent2_mat; ...
         this_winner_2'; ...
         this_winner_2'; ...
         this_winner_2']; %#ok<AGROW>


    % Residualize by both parent levels
    sampled_data_fix = ...
        sampled_data_fix - ...
        this_winner_1 - ...
        this_winner_2;


    % --------------------------------------------------------
    % Train Cycle 3
    % --------------------------------------------------------

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


    % --------------------------------------------------------
    % Average Cycle-3 winners
    % --------------------------------------------------------

    [~,~,ix] = ...
        unique(cycle3_winners,"stable");


    winner_stats = ...
        [unique(cycle3_winners,"stable"), ...
         accumarray(ix,1)];


    for i = 1:size(winner_stats,1)

        cell_idx = ...
            winner_stats(i,1);

        winner_average(:,:,cell_idx) = ...
            winner_average(:,:,cell_idx) ./ ...
            winner_stats(i,2);

        cycle3_cells_iter(:,cell_idx) = ...
            winner_average(:,:,cell_idx);

    end


    cycle3_winners = ...
        unique(cycle3_winners);


    % --------------------------------------------------------
    % Assign observations to Cycle-3 winners
    % --------------------------------------------------------

    [~,idx] = ...
        max( ...
        normc(sampled_data_fix)' * ...
        cycle3_cells(:,cycle3_winners), ...
        [],2);


    cluster_assign_cycle(rawinput_idx,cycle) = ...
        cycle3_winners(idx);


    cycle3_winners = ...
        cycle3_winners(unique(idx,"stable"));


    cycle3_winner_mat = ...
        [cycle3_winner_mat; ...
         cycle3_winners(:)]; %#ok<AGROW>


    % --------------------------------------------------------
    % Restore magnitudes
    % --------------------------------------------------------

    for c = cycle3_winners'

        each_input = ...
            winner_len_mat( ...
            winner_len_mat(:,1) == c,2);

        ori_cycle3_cells(:,c) = ...
            cycle3_cells_iter(:,c) * ...
            mean(each_input);

    end


    % Prevent winner reuse
    cycle3_cells(:,cycle3_winners) = 0;
    cycle3_cells_iter(:,cycle3_winners) = 0;

end


% ------------------------------------------------------------
% Reconstruct Cycle-3 centroids in original coordinate system
% ------------------------------------------------------------

cycle3_winner_demask = ...
    ori_cycle3_cells(:,cycle3_winner_mat) + ...
    cycle3_parent1_mat' + ...
    cycle3_parent2_mat';


fprintf('Cycle 3 recovered clusters: %d\n', ...
    length(cycle3_winner_mat));


%% ============================================================
%  6. Sanity checks
% =============================================================

fprintf('\n============================================\n');
fprintf('LI-HC hierarchy summary\n');
fprintf('============================================\n');

fprintf('Cycle 1 clusters: %d\n', ...
    numel(unique(cluster_assign_cycle(:,1))));

fprintf('Cycle 2 clusters: %d\n', ...
    numel(unique(cluster_assign_cycle(:,2))));

fprintf('Cycle 3 clusters: %d\n', ...
    numel(unique(cluster_assign_cycle(:,3))));


%% ============================================================
%  7. Common PCA projection for 3D visualization
% =============================================================

% Combine original observations, true finest-level centroids,
% and learned LI-HC centroids in the same PCA coordinate system.

final_all_data = [ ...
    inputs; ...
    center; ...
    cycle1_winner_demask'; ...
    cycle2_winner_demask'; ...
    cycle3_winner_demask' ...
    ];


[coeff,~,~,~,explained,~] = ...
    pca(final_all_data);


Z = ...
    final_all_data * ...
    coeff(:,1:3);


%% Index bookkeeping

n_input = ...
    size(inputs,1);

n_true = ...
    size(center,1);

n_c1 = ...
    size(cycle1_winner_demask,2);

n_c2 = ...
    size(cycle2_winner_demask,2);

n_c3 = ...
    size(cycle3_winner_demask,2);


idx_input = ...
    1:n_input;

idx_true = ...
    n_input + ...
    (1:n_true);

idx_c1 = ...
    n_input + n_true + ...
    (1:n_c1);

idx_c2 = ...
    n_input + n_true + n_c1 + ...
    (1:n_c2);

idx_c3 = ...
    n_input + n_true + n_c1 + n_c2 + ...
    (1:n_c3);


%% ============================================================
%  8. FIGURE 2(a)
%  Representative 3D LI-HC hierarchy
%
%  You can manually rotate this MATLAB figure and screenshot it.
% =============================================================

sample_parent = 4;


% ------------------------------------------------------------
% Observations belonging to selected Cycle-1 branch
% ------------------------------------------------------------

parent_winner = ...
    cycle1_winners(sample_parent);


data_idx = ...
    find( ...
    cluster_assign_cycle(:,1) == ...
    parent_winner);


% ------------------------------------------------------------
% True finest-level centroids belonging to this branch
%
% Dataset construction gives six finest-level centroids
% beneath each Cycle-1 cluster.
% ------------------------------------------------------------

true_local_idx = ...
    (sample_parent-1)*6 + ...
    (1:6);


% ------------------------------------------------------------
% Cycle-2 centroids belonging to selected branch
% ------------------------------------------------------------

cycle2_local_idx = [];

for i = 1:length(cycle2_winner_mat)

    member_idx = ...
        find( ...
        cluster_assign_cycle(:,2) == ...
        cycle2_winner_mat(i));

    if any(ismember(member_idx,data_idx))

        cycle2_local_idx = ...
            [cycle2_local_idx,i]; %#ok<AGROW>

    end

end


% ------------------------------------------------------------
% Cycle-3 centroids belonging to selected branch
% ------------------------------------------------------------

cycle3_local_idx = [];

for i = 1:length(cycle3_winner_mat)

    member_idx = ...
        find( ...
        cluster_assign_cycle(:,3) == ...
        cycle3_winner_mat(i));

    if any(ismember(member_idx,data_idx))

        cycle3_local_idx = ...
            [cycle3_local_idx,i]; %#ok<AGROW>

    end

end


%% Plot 3D hierarchy

figure('Color','w');

hold on;
grid on;
box on;
view(3);


% Original observations
plot3( ...
    Z(data_idx,1), ...
    Z(data_idx,2), ...
    Z(data_idx,3), ...
    'r.', ...
    'MarkerSize',8);


% True finest-level centroids
plot3( ...
    Z(idx_true(true_local_idx),1), ...
    Z(idx_true(true_local_idx),2), ...
    Z(idx_true(true_local_idx),3), ...
    'g*', ...
    'MarkerSize',13, ...
    'LineWidth',1.5);


% Cycle-1 learned centroid
plot3( ...
    Z(idx_c1(sample_parent),1), ...
    Z(idx_c1(sample_parent),2), ...
    Z(idx_c1(sample_parent),3), ...
    'bp', ...
    'MarkerSize',18, ...
    'MarkerFaceColor','b');


% Cycle-2 learned centroids
plot3( ...
    Z(idx_c2(cycle2_local_idx),1), ...
    Z(idx_c2(cycle2_local_idx),2), ...
    Z(idx_c2(cycle2_local_idx),3), ...
    'bo', ...
    'MarkerSize',10, ...
    'LineWidth',1.5);


% Cycle-3 learned centroids
plot3( ...
    Z(idx_c3(cycle3_local_idx),1), ...
    Z(idx_c3(cycle3_local_idx),2), ...
    Z(idx_c3(cycle3_local_idx),3), ...
    'bs', ...
    'MarkerSize',9, ...
    'LineWidth',1.5);


xlabel( ...
    sprintf('PC1 (%.1f%% variance)', ...
    explained(1)));

ylabel( ...
    sprintf('PC2 (%.1f%% variance)', ...
    explained(2)));

zlabel( ...
    sprintf('PC3 (%.1f%% variance)', ...
    explained(3)));


title('Representative LI-HC hierarchy');


legend( ...
    'Original data', ...
    'True centroids', ...
    'Cycle 1 centroid', ...
    'Cycle 2 centroids', ...
    'Cycle 3 centroids', ...
    'Location','best');


set(gca, ...
    'FontSize',11, ...
    'LineWidth',1);


axis vis3d;

hold off;


%% ============================================================
%  9. FIGURE 2(b-c)
%  Side-by-side dendrogram comparison
% =============================================================

% ------------------------------------------------------------
% LI-HC learned finest-level centroids
% ------------------------------------------------------------

tree_lihc = ...
    linkage( ...
    cycle3_winner_demask', ...
    'average', ...
    'cosine');


% ------------------------------------------------------------
% Original synthetic data
% ------------------------------------------------------------

tree_original = ...
    linkage( ...
    data', ...
    'average', ...
    'cosine');


% ------------------------------------------------------------
% Plot side by side
% ------------------------------------------------------------

figure( ...
    'Color','w', ...
    'Position',[100 100 1200 450]);


tiledlayout( ...
    1,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');


% ============================================================
% LEFT: LI-HC learned centroids
% ============================================================

nexttile;

dendrogram( ...
    tree_lihc, ...
    24);

title( ...
    'LI-HC learned centroids', ...
    'FontWeight','normal');

xlabel('Cluster leaves');

ylabel('Average-linkage distance (cosine)');

ylim([0 0.06]);

set(gca, ...
    'FontSize',11, ...
    'LineWidth',1);

box off;


% ============================================================
% RIGHT: Original synthetic data
% ============================================================

nexttile;

dendrogram( ...
    tree_original, ...
    24);

title( ...
    'Original data', ...
    'FontWeight','normal');

xlabel('Cluster leaves');

ylabel('Average-linkage distance (cosine)');

ylim([0 0.06]);

set(gca, ...
    'FontSize',11, ...
    'LineWidth',1);

box off;