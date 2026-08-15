%% ============================================================
%  WDBC benchmark: PCA vs AIME vs Oja vs Sanger GHA
%
%  Dataset:
%  Wisconsin Diagnostic Breast Cancer (WDBC)
%
%  Goal:
%  Recover the first three principal-component directions and
%  compare absolute cosine similarity with PCA reference PCs.
%
%  AIME:
%    - 600 neural weight vectors
%    - 40% sparse initialization
%    - learning rate = 1e-4
%    - 400 epochs / cycle
%    - K = 4 post-hoc clusters, allowing multiple AIME lines
%    - best PC-aligned AIME line is reported for each cycle
%
%  Oja:
%    - vanilla Oja learning rule
%    - sequential deflation for PC1, PC2, PC3
%
%  Sanger GHA:
%    - three output weight vectors trained simultaneously
%    - generalized Hebbian update for PC1, PC2, PC3
% ============================================================

clear;
clc;
format shortG;
warning off;


%% ============================================================
%  1. Load WDBC data
% =============================================================

% wdbc.data format:
% Column 1      = ID
% Column 2      = Diagnosis ('M' or 'B')
% Columns 3:32  = 30 numerical features

rawData = readcell('wdbc.data', ...
    'FileType','text', ...
    'Delimiter',',');

IDs = rawData(:,1);          %#ok<NASGU>
diagnosis = rawData(:,2);    %#ok<NASGU>

% Convert feature columns to numerical matrix
X_raw = cell2mat(rawData(:,3:end));


%% Standardize features
% WDBC features have substantially different native scales.

X = zscore(X_raw);

% Remove residual numerical mean
X = X - mean(X,1);

fprintf('WDBC dataset: %d samples x %d features\n', ...
    size(X,1), size(X,2));


%% ============================================================
%  2. Reference PCA
% =============================================================

[coeff_real,~,~,~,explained,~] = pca(X);

% Normalize PCA directions
coeff_real = normc(coeff_real);

fprintf('\nVariance explained by first three PCs:\n');
disp(explained(1:3));

fprintf('Cumulative variance explained by first three PCs:\n');
disp(cumsum(explained(1:3)));


%% ============================================================
%  3. AIME settings
% =============================================================

SetRNG(1);

n_src = size(X,2);
n_dst = 600;

connectivity = 0.4;
n_per_src = round(n_src * connectivity);

aime_epoch = 400;
aime_lr = 1e-4;

% Allow multiple candidate AIME directions
K = 4;


%% ============================================================
%  4. Initialize AIME neural weight population
% =============================================================

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


%% ============================================================
%  5. AIME: recover first three PC-aligned components
% =============================================================

aime_alignment = zeros(1,3);
aime_components = zeros(n_src,3);

current_inputs = X;


for cycle = 1:3

    fprintf('\nAIME cycle %d...\n',cycle);


    %% Train AIME

    final_weights = ...
        run_aime( ...
        current_inputs, ...
        cells, ...
        aime_epoch, ...
        aime_lr);


    %% Post-hoc identification of candidate AIME lines

    candidate_lines = ...
        identify_aime_lines( ...
        final_weights, ...
        K);


    %% Compare candidate lines with corresponding PCA direction

    target_pc = ...
        coeff_real(:,cycle);

    similarities = ...
        abs(candidate_lines * target_pc);

    [best_similarity,best_idx] = ...
        max(similarities);

    best_line = ...
        candidate_lines(best_idx,:)';

    best_line = ...
        best_line ./ norm(best_line);


    %% Save result

    aime_alignment(cycle) = ...
        best_similarity;

    aime_components(:,cycle) = ...
        best_line;

    fprintf('Best AIME-PC%d alignment: %.6f\n', ...
        cycle,best_similarity);


    %% Mask selected AIME direction before next cycle

    if cycle < 3

        current_inputs = ...
            mask_component( ...
            current_inputs, ...
            best_line);

    end

end


%% ============================================================
%  6. Vanilla Oja + sequential deflation
% =============================================================

oja_alignment = zeros(1,3);
oja_components = zeros(n_src,3);

oja_epoch = 200;
oja_lr = 1e-3;

rng(1);

oja_inputs = X;


for component = 1:3

    fprintf('\nOja component %d...\n',component);


    %% Random initialization

    w = randn(n_src,1);
    w = w ./ norm(w);


    %% Vanilla Oja learning
    %
    % y = w' * x
    %
    % w <- w + eta * y * (x - y*w)

    for e = 1:oja_epoch

        order = ...
            randperm(size(oja_inputs,1));

        for ii = 1:length(order)

            x = ...
                oja_inputs(order(ii),:)';

            y = ...
                w' * x;

            w = ...
                w + ...
                oja_lr * y * ...
                (x - y*w);

            % Numerical normalization
            w = ...
                w ./ norm(w);

        end

    end


    %% Compare Oja direction with corresponding true PCA direction

    target_pc = ...
        coeff_real(:,component);

    oja_alignment(component) = ...
        abs(w' * target_pc);

    oja_components(:,component) = ...
        w;

    fprintf('Oja-PC%d alignment: %.6f\n', ...
        component, ...
        oja_alignment(component));


    %% Sequential deflation

    if component < 3

        oja_inputs = ...
            oja_inputs - ...
            (oja_inputs*w)*w';

        oja_inputs = ...
            oja_inputs - ...
            mean(oja_inputs,1);

    end

end


%% ============================================================
%  7. Sanger Generalized Hebbian Algorithm (GHA)
% =============================================================

fprintf('\nSanger GHA: learning PC1-PC3 simultaneously...\n');

n_components = 3;

gha_epoch = 200;
gha_lr = 1e-3;

rng(1);

% Columns of W are the three learned component directions
W = randn(n_src,n_components);

% Normalize initial directions
for j = 1:n_components

    W(:,j) = ...
        W(:,j) ./ norm(W(:,j));

end


%% ------------------------------------------------------------
%  GHA training
%
%  For output j:
%
%  y_j = w_j' x
%
%  w_j <- w_j + eta*y_j *
%         (x - sum_{k=1}^{j} y_k w_k)
%
%  The first output reduces to an Oja-like update.
%  Higher outputs progressively remove contributions from the
%  earlier learned directions.
% ------------------------------------------------------------

for e = 1:gha_epoch

    order = ...
        randperm(size(X,1));

    for ii = 1:length(order)

        x = ...
            X(order(ii),:)';

        % Neural outputs from current weights
        y = ...
            W' * x;


        %% Update all GHA outputs

        for j = 1:n_components

            reconstructed = ...
                W(:,1:j) * y(1:j);

            W(:,j) = ...
                W(:,j) + ...
                gha_lr * y(j) * ...
                (x - reconstructed);

        end


        %% Numerical stabilization

        for j = 1:n_components

            norm_w = ...
                norm(W(:,j));

            if norm_w > 0

                W(:,j) = ...
                    W(:,j) ./ norm_w;

            end

        end

    end

end


%% ============================================================
%  8. Match GHA components to reference PCs
% =============================================================

gha_alignment = zeros(1,3);
gha_components = zeros(n_src,3);


% Absolute pairwise cosine similarities:
%
% rows    = GHA components
% columns = true PCA components

alignment_matrix = ...
    abs(W' * coeff_real(:,1:3));


% Use one-to-one matching so one GHA output cannot be
% assigned to multiple PCA components.

used_rows = [];

for pc = 1:3

    candidate_alignment = ...
        alignment_matrix(:,pc);

    candidate_alignment(used_rows) = ...
        -Inf;

    [best_similarity,best_idx] = ...
        max(candidate_alignment);

    gha_alignment(pc) = ...
        best_similarity;

    gha_components(:,pc) = ...
        W(:,best_idx);

    used_rows = ...
        [used_rows,best_idx]; %#ok<AGROW>

    fprintf('Sanger GHA-PC%d alignment: %.6f\n', ...
        pc,best_similarity);

end


%% ============================================================
%  9. Final benchmark table
% =============================================================

Method = ...
    {'AIME'; ...
     'Oja'; ...
     'Sanger GHA'};

PC1 = ...
    [aime_alignment(1); ...
     oja_alignment(1); ...
     gha_alignment(1)];

PC2 = ...
    [aime_alignment(2); ...
     oja_alignment(2); ...
     gha_alignment(2)];

PC3 = ...
    [aime_alignment(3); ...
     oja_alignment(3); ...
     gha_alignment(3)];


T = table( ...
    Method, ...
    PC1, ...
    PC2, ...
    PC3);


fprintf('\n=================================================\n');
fprintf('WDBC benchmark: absolute cosine similarity\n');
fprintf('=================================================\n');

disp(T);


%% ============================================================
%  10. Publication-ready rounded table
% =============================================================

T_round = T;

T_round.PC1 = ...
    round(T_round.PC1,4);

T_round.PC2 = ...
    round(T_round.PC2,4);

T_round.PC3 = ...
    round(T_round.PC3,4);

fprintf('\nPublication-ready values:\n');

disp(T_round);


%% Save

writetable( ...
    T, ...
    'WDBC_AIME_Oja_Sanger_benchmark.csv');


%% ============================================================
%  LOCAL FUNCTION
%  Run one AIME learning cycle
% =============================================================

function final_weights = ...
    run_aime(inputs,cells,epoch,lr)

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
        ori_cells;

end


%% ============================================================
%  LOCAL FUNCTION
%  Identify candidate AIME lines using post-hoc K-means
% =============================================================

function lines = ...
    identify_aime_lines(weights,K)

    %% Center learned weight population

    new_weight = ...
        weights - mean(weights,2);


    %% Normalize each learned vector

    new_weight = ...
        normc(new_weight)';


    %% Post-hoc K-means

    [~,C] = ...
        kmeans( ...
        new_weight, ...
        K, ...
        'Replicates',5);

    C = ...
        normr(C);


    %% Remove antipodal duplicates
    %
    % +v and -v represent the same AIME component.

    lines = [];

    for i = 1:size(C,1)

        candidate = ...
            C(i,:);

        if isempty(lines)

            lines = ...
                candidate;

        else

            similarity_to_existing = ...
                abs(lines * candidate');

            if max(similarity_to_existing) < 0.95

                lines = ...
                    [lines;candidate]; %#ok<AGROW>

            end

        end

    end

end


%% ============================================================
%  LOCAL FUNCTION
%  Remove one recovered AIME component from input
% =============================================================

function masked_inputs = ...
    mask_component(inputs,line_direction)

    line_direction = ...
        line_direction ./ ...
        norm(line_direction);

    projection = ...
        (inputs * line_direction) * ...
        line_direction';

    masked_inputs = ...
        inputs - projection;

    masked_inputs = ...
        masked_inputs - ...
        mean(masked_inputs,1);

end