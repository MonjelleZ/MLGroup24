function  [tree]  = createTree(data,label,RN,feature )
    
    feature_name = {'Displacement','HorsePower','Weight','Acceleration'};
    tree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', []);
    entropy = getEntropy(data,label);
    
    [Index,value] =Getgain(data,label,RN,entropy );
    

    if Index == 0
        tree.op = [];
        tree.attribute = [];
        tree.prediction = value;
        tree.threshold = [];
        tree.kids = cell(0);
        return
    else
        tree.op = feature_name{Index};
        tree.attribute = Index;
        tree.prediction = [];
        tree.threshold = value;
        feature = [feature, Index];
    end
    
    [lSet,llab,rSet,rlab] = Split(data,label, Index, value);

    tree.kids = {createTree( lSet,llab,RN,feature ), createTree( rSet,rlab,RN,feature)};
    
end