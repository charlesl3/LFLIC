clear
format shortG
warning off;

load ('mnist.mat')
colormap(gray(256))
images = rescale(test.images(:,:,:),0,255);
data = reshape(images,784,10000);

inputs_ori = data';
inputs = inputs_ori-mean(inputs_ori);


[coeff_real,~,~,~,explained_1,~] = pca(inputs);
o = [0 0 0];
final_all_data = [inputs;coeff_real(:,1)';coeff_real(:,2)';coeff_real(:,3)'];
[coeff1,~,~,~,explained_real,~] = pca(final_all_data);
Z=final_all_data*coeff1(:,1:3);
%Z = round(Z,4);
coeff_real = round(coeff_real,2);
explained = round(explained_real);



figure;
view(3)
hold on
plot3(Z(1:end-3,1),Z(1:end-3,2),Z(1:end-3,3),'r.','MarkerSize',15)
arrow(o,Z(end-2,:)*5,'Color','b');
arrow(o,Z(end-1,:),'Color','g');
arrow(o,Z(end,:),'Color','r');
xlabel('PC 1(' + string(explained(1))+"% variance expl)")
ylabel('PC 2(' + string(explained(2))+"% variance expl)")
zlabel('PC 3(' + string(explained(3))+"% variance expl)")
xh = get(gca,'XLabel'); % Handle of the x label
set(xh, 'Units', 'Normalized')
pos = get(xh, 'Position');
set(xh, 'Position',pos.*[1,-0.05,1],'Rotation',15)
yh = get(gca,'YLabel'); % Handle of the y label
set(yh, 'Units', 'Normalized')
pos = get(yh, 'Position');
set(yh, 'Position',pos.*[1,-0.07,1],'Rotation',-25)
title('ori data and ori true PCs before cycle 1')
legend('data','1st','2nd','3rd')

set(gca, 'FontSize', 15);% Increase font size
set(gca, 'LineWidth', 1.5); % Make lines thicker
set(gca, 'FontName', 'Times New Roman'); % Set preferred font
grid on
hold off

% initiate neural weight vectors
SetRNG(1);
dim = size(inputs,2);
n_src = dim;
n_dst = 70;
n_per_src = round(n_src*0.4);
synaptic_weights_mat = randn(n_src,n_dst);
[srcIdx,dstIdx] = ConnectHypergeometric(n_dst, n_src, n_per_src);
index = [srcIdx;dstIdx];
for i = 1:n_dst;
    nonzero_idx = index(2,find(index(1,:) == i));
    zero_idx = setdiff(1:n_src,nonzero_idx);
    synaptic_weights_mat(zero_idx,i) = 0;
end
cells = synaptic_weights_mat; %original

%cycle 1, find the first PC
cycle = 1;
ori_cycle1_cells = cells;
mean_sum = [];
final_weight = [];
epoch = 500;%300 good



mean1 = [];
for e = 1:epoch;
    imterim_weight = [];
    sampled_data = inputs;
    sampled_data = inputs(randperm(size(inputs, 1)),:);
    mean_sum = [];
    for col = 1:size(sampled_data,1); % loop over all inputs
        lr = 1e-7;
        input1_ori = sampled_data(col,:);% each input
        input1 = input1_ori';
        product = input1'*ori_cycle1_cells;
        signs = sign(product);
        winning_idx = 1:length(product);
        winning_cell = ori_cycle1_cells(:,winning_idx); % the winning cell set, which may contain more than one winning cell
        update_winner_ori = winning_cell+(signs.*input1-winning_cell)*lr;
        update_winner_norm = update_winner_ori;
        ori_cycle1_cells(:,winning_idx) = update_winner_norm;
    end
     final_weight = [final_weight,normc(ori_cycle1_cells(:,1))];
end

new_weight = update_winner_norm-mean(update_winner_norm,2);
new_weight = normc(new_weight)';
[idx,C,sumd,D] = kmeans(new_weight,4);
combine_line_data = normr([new_weight;C]);
[coeff_c,~,~,~,explained,~] = pca(combine_line_data);
combine_line_data = normr([new_weight;C]);
[coeff_c,~,~,~,explained,~] = pca(combine_line_data);
Z = combine_line_data*coeff_c(:,1:3);
figure;
view(3)
plot3(Z(1:end-4,1),Z(1:end-4,2),Z(1:end-4,3),'r*','MarkerSize',20)
for i=1:4;
    arrow(o,Z(end-(i-1),:),'Color','b');
end


line1 = normr(C(4,:))';
line2 = normr(C(3,:))';

w1_real = line1'*normc(coeff_real);
w1_real_2 = line2'*normc(coeff_real);

line1_var = round(var(inputs*line1)./sum(var(inputs*[line1,line2,coeff_real(:,1)])),4)
line2_var = round(var(inputs*line2)./sum(var(inputs*[line1,line2,coeff_real(:,1)])),4)
PC1_var = round(var(inputs*coeff_real(:,1))./sum(var(inputs*[line1,line2,coeff_real(:,1)])),4)
