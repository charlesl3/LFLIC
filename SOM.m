% Author: Charles Liu, 01/31/2024
% SOM: use the self-organizing map to generate lines of AIME

% Inputs: 
%   - new_weight = resulting weights after AIME training
% Output: 
%   - C = updated weight vectors, each two of them form an AIME line
%   - winner_idx = index of the winning weight 

function [C,winner_idx] = SOM(new_weight);  
    [row,col] = size(new_weight);
    cells = normc(rand(col,10));
    epoch = 30;
    for k = 1:epoch;
        cycle1_winners = [];
        sampled_data = new_weight(randperm(size(new_weight, 1)),:);
        winner_average = zeros(col,1,10);
        for row = 1:size(sampled_data,1); % loop over all inputs
            lr = 0.001;
            input1 = sampled_data(row,:);% each input
            input1 = normr(input1);
            product = input1*cells; % the dot products of the input and all cells
            winning_value = max(product); % max dot product value
            cell_idx = find(product == winning_value); % index(indices) of winning cell(s)
            cycle1_winners = [cycle1_winners;cell_idx];
            winner = cells(:,cell_idx);
            update_winner_ori = winner+(input1'-winner)*lr; %update_winner_ori is NOT norm
            winner_average(:,:,cell_idx) = update_winner_ori + winner_average(:,:,cell_idx);
            cells(:,cell_idx) = update_winner_ori;
        end
        winner_idx = unique(cycle1_winners);
        all_cells = winner_average./row;
        C = reshape(all_cells(:,:,winner_idx),col,length(winner_idx))';
        C = normr(C);
    end
end