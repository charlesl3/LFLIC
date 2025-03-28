% Author: Charles Liu, 01/15/2022
% subcluster_simulate: simulate data for hierarchical clustering

% Inputs: 
%   - center = the centriod points of each subcluster as an
%   n-by-dim matrix. where n is the number of all subclusters
%   - dim = dimensions of the desired data
%   - num = randomly generate data for each centroid
%   - sigma = the standard deviation of the gaussian noise that is added to
%   each centroid for generating the data

% Outputs: 
%   - data = resulting matrix with all simulated data

function [data,c] = subcluster_simulate(center,dim,num,sigma)
c = center;
data = [];
for i = 1:size(center,1)
    r = normrnd(0,sigma,[num,dim])+center(i,:);
    data = [data;r];
end
end