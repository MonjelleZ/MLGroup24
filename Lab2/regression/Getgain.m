function [ Index, Value ] = Getgain( data,label,RN,entropy )

    m = length(data);
    n = size(data,2);

    if length(unique(label)) == 1
        Index = 0;
        Value = mean(label);
        return;
    end
    

    bestGain = 0;
    bestIndex = 0;
    bestValue = 0;

    
    Feature = 1:n;
    [~,mf] = size(Feature);

    for j = 1:mf
        uniqueValue = unique(data(:,j));
        lenCharacter = length(uniqueValue);
        
        for i = 1:lenCharacter
            tempValue = uniqueValue(i,:);
            [matLeft,labLeft,matRight,labRight] = Split(data,label,j,tempValue);
            mLeft = size(matLeft);
            mRight = size(matRight);
            if mLeft(1) < RN || mRight(1) < RN
                continue;
            end
            newGain =entropy - ((mLeft(1))/m)*getEntropy(matLeft,labLeft) - ((mRight(1))/m)*getEntropy(matRight,labRight);
            if newGain > bestGain
                bestGain = newGain;
                bestIndex = j;
                bestValue = tempValue;
            end
        end
    end

    if bestGain==0
        
    
    [matLeft,labLeft,matRight,labRight] = Split(data,label, bestIndex ,bestValue);

    mLeft = size(matLeft);
    mRight = size(matRight);
    if mLeft(1) < RN || mRight(1) < RN
        Index = 0;
        Value = mean(label);
        return;
    end
    Index = bestIndex;
    Value = bestGain;
end