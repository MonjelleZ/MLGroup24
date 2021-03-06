function [tree] = CreateTreeClassification(data,feature_used)

    %{
     feature name of column 1-6:

     Incidence of pelvis : IOP
     pelvic inclination : PI
     lumbar lordosis Angle : LLA
     sacral slope : SS
     pelvic radius : PR
     degree of lumbar spondylolisthesis : DOLS

    %}
    feature_name = {'IOP','PI','LLA','SS','PR','DOLS'};
    [m,n] = size(data);
    allFeature = 1:(n-1);
    unusedFeature = setdiff(allFeature, feature_used);
    [~,mf] = size(unusedFeature);
    tree = struct('op', [], 'kids', [], 'prediction', [], 'attribute', [], 'threshold', []);
    entropy = getEntropy(data);
    
    temp_type=data(1,n);
    temp_b=true;

    for i=1:m
       if temp_type ~= data(i,n)
        temp_b=false;
       end
    end

    if temp_b==true || mf == 1
        res = mostType(data);
        tree.prediction=res;
        tree.kids=cell(0);
        tree.op = [];
        tree.attribute = [];
        tree.threshold = [];
        return
            
    else
        bestGain = 0;
        bestfeature = 0;
        bestsplitnum = 0;
        leftdata = [];
        rightdata = [];
        for j = 1:mf % traverse unusedFeature columns(features)
            [gain,split_num,feature_data1,feature_data2] = getGain(entropy,data,unusedFeature(j));
            if gain > bestGain
                bestGain = gain;
                bestfeature = unusedFeature(j);
                bestsplitnum = split_num;
                leftdata = feature_data1;
                rightdata = feature_data2;
            end
                
        end
        if bestGain == 0
            bestfeature = unusedFeature(1);
            bestsplitnum = split_num;
            leftdata = feature_data1;
            rightdata = feature_data2;
        end
        feature_used =[feature_used,bestfeature];
        tree.op = feature_name{bestfeature};
        tree.threshold = bestsplitnum;
        tree.attribute = bestfeature;
        tree.prediction=[];
        tree.kids = {CreateTreeClassification(leftdata,feature_used), CreateTreeClassification(rightdata,feature_used)};
    end
  

end


