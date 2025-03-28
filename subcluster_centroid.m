% Author: Charles Liu
% subcluster_centroid: simulate the center points for the use of
% the function subcluster_simulate.m

% Inputs: 
%   - k = the number of centroids of "big" clusters
%   - dim = dimensions of the desired data
%   - mu_range = the range of the centroids of "big" clusters
%   - sigma = the standard deviation of the gaussian noise that is added to
%   each centroid "big" clusters for generating sub-centroids

% Outputs: 
%   - center = the centriod points of each subcluster as an
%   n-by-dim matrix. where n is the number of all subclusters
%   - mu_mat = the centriod points of "big" clusters

function [center,mu_mat] = subcluster_centroid(k,dim,num,mu_range,sigma)

%[mu_mat, ] = random_mu_sigma(k,dim,mu_range,[0 1]);

a = mu_range(1);
b = mu_range(2);
mu_mat = a + (b-a)*rand(k,dim);
center = [];
for i = 1:size(mu_mat,1)
    ran = abs(normrnd(0,sigma,[num,dim]));
    ran1 = ran ;
    idx = randperm(numel(ran),round(numel(ran)*50/100)) ; % get 60% of indices randomly
    ran1(idx) = -ran1(idx) ;
    %ran1 = normrnd(0,sigma,[num,dim]);
    r = ran1 + mu_mat(i,:);
    center = [center;r];
end

end