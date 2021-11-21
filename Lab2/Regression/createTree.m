%% Create the tree recursively

% Description: Use the best split index and value of all features to create the tree recursively.
% Args:
%      dataSet: The dataset to train/build the tree
%      tolS: Tolerate(Min) decreased sum of variances
%      tolN: Tolerate(Min) number of nodes in dataSet
%      feature_used: store the index of features which have been used 
%      algori: only support 'CART' and 'ID3'
% Return:
%      tree: The decision tree in struct type
    

function  [tree]  = createTree(data,label,tolS,tolN,feature_used )
    
    feature_name = {'Displacement','HorsePower','Weight','Acceleration'};
    tree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', []);
    
    [fIndex,val] = chooseBestSplit(data,label, tolS, tolN, feature_used);
    
    % fIndex == 0 means it is a leaf node
    if fIndex == 0
        tree.op = [];
        tree.attribute = [];
        tree.prediction = val;
        tree.threshold = [];
        tree.kids = cell(0);
        return
    else
        tree.op = feature_name{fIndex};
        tree.attribute = fIndex;
        tree.prediction = [];
        tree.threshold = val;
        feature_used = [feature_used, fIndex];
    end
    
    % Use the best split index and value to split the dataset to left/right DataSet 
    [lSet,llab,rSet,rlab] = binSplitDataSet(data,label, fIndex, val);
    % Use tree.kids save the kid nodes recursively
    tree.kids = {createTree( lSet,llab,tolS,tolN,feature_used ), createTree( rSet,rlab,tolS,tolN,feature_used)};
    
end