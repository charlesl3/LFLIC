function p = purity(assigned_cluster,true_label)
pure_mat = [assigned_cluster,true_label];
total_TP = 0;
id = unique(assigned_cluster);
for i = 1:length(id);
    sub_mat = pure_mat(find(assigned_cluster == id(i)),:);
    maxid = mode(sub_mat(:,2));
    total_TP = total_TP + sum(sub_mat(:,2) == maxid);
end 
p = total_TP/length(assigned_cluster);
end