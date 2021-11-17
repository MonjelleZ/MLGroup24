function [ DLeft, DRight ] = BinSpilt( dataSet, feature, value )
  
    [m,~] = size(dataSet);
    DataTemp = dataSet(:,feature)';
    
    indexAll = 1:m;
    indexLeft = indexAll(DataTemp > value);
    indexRight = indexAll(DataTemp <= value);
    
    [~,nLeft] = size(indexLeft);
    [~,nRight] = size(indexRight);
    
    if nLeft>0 && nRight>0
        DLeft = dataSet(indexLeft,:);
        DRight = dataSet(indexRight,:);
    elseif nLeft == 0
         DLeft = [];
         DRight = dataSet;
    elseif nRight == 0
         DRight = [];
         DLeft = dataSet;
    end
end
