function [ Index, Value ] = chooseBestSplit( data,label, tolS, tolN, feature_used )

    % (m(1), m(2)) == (row, column)
    m = length(data);
    n= size(data,2);

    if length(unique(label)) == 1% only one lable
        Index = 0;
        Value = regLeaf(label);
        return;
    end
    
    % Variance of original dataSet
    originalVar = varianceErr(label);
    bestVar = inf;
    bestIndex = 0;
    bestValue = 0;

    
    Feature = 1:n;
    [~,mf] = size(Feature);
    % Find the best split index and value
    for j = 1:mf
        uniqueValue = unique(data(:,j));
        lenCharacter = length(uniqueValue);
        
        for i = 1:lenCharacter
            tempValue = uniqueValue(i,:);
            [matLeft,labLeft,matRight,labRight] = binSplitDataSet(data,label,j,tempValue);
            mLeft = size(matLeft);
            mRight = size(matRight);
            if mLeft(1) < tolN || mRight(1) < tolN
                continue;
            end
            newVar = varianceErr(labLeft) + varianceErr(labRight);
            if newVar < bestVar
                bestVar = newVar;
                bestIndex = j;
                bestValue = tempValue;
            end
        end
    end
    
    if (originalVar - bestVar) < tolS
        Index = 0;
        Value = regLeaf(label);
        return;
    end

    
    [matLeft,labLeft,matRight,labRight] = binSplitDataSet(data,label, bestIndex ,bestValue);

    mLeft = size(matLeft);
    mRight = size(matRight);
    if mLeft(1) < tolN || mRight(1) < tolN
        Index = 0;
        Value = regLeaf(label);
        return;
    end
    Index = bestIndex;
    Value = bestValue;
end