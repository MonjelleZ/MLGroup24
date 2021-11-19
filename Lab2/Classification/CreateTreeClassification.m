function [tree] = CreateTreeClassification(data)
    %feature_name = {'X1','X2','X3','X4','X5','X6'};
    [m,n] = size(data);
    %tree = struct('op', [], 'kids', [], 'type', [], 'attribute', []);
    tree = struct('op', [], 'kids', [], 'class', [], 'attribute', [], 'threshold', []);
    entropy = getEntropy(data);

    temp_type=data(1,n);
    temp_b=true;

    for i=1:m
       if temp_type ~= data(i,n)
        temp_b=false;
       end
    end

    if temp_b==true || n ==2
        %tree.class=data(1,end);
        res = mostType(data);
        tree.class=res;
        tree.kids=cell(0);
        tree.op = [];
        tree.attribute = [];
        tree.threshold = [];
        disp('------ end ----- ')
        return
            
    else
        bestGain = 0;
        bestfeature = 0;
        for i = 1:n-1
            [gain,split_num,feature_data1,feature_data2] = getGain(entropy,data,i);
            if gain > bestGain
                bestGain = gain;
                bestfeature = i;
            end
        end
        disp(bestfeature)
        disp(bestGain)
        feature_data1(:,bestfeature) = [];
        feature_data2(:,bestfeature) = [];
        disp('-------')
        disp(feature_data1)
        disp('__')
        disp(feature_data2)
        [~, l ] = size(feature_data1);
        [~, r ] = size(feature_data2);
        tree.op = bestfeature;
        tree.threshold = split_num;
        tree.attribute = [];
        tree.class=[];
        tree.kids = {CreateTreeClassification(feature_data1), CreateTreeClassification(feature_data2)};
    end
  

end


