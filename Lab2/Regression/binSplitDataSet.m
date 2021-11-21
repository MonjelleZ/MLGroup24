function [ dataSetLeft,labelSetLeft, dataSetRight,labelSetRight] = binSplitDataSet( data,label, feature, value )
  
    m=length(data);
    DataTemp = data(:,feature)';% tansform to row model
    
    indexAll = 1:m;
    indexLeft = indexAll(DataTemp > value);
    indexRight = indexAll(DataTemp <= value);
    
    nLeft = size(indexLeft,2);
    nRight = size(indexRight,2);
    
    if nLeft>0 && nRight>0
        dataSetLeft = data(indexLeft,:);
        labelSetLeft= label(indexLeft,:);
        dataSetRight = data(indexRight,:);
        labelSetRight = label(indexRight,:);
    elseif nLeft == 0
            dataSetLeft = [];
            labelSetLeft= [];
            dataSetRight = data;
            labelSetRight =label;
    elseif nRight == 0
            dataSetRight = [];
            labelSetRight = [];
            dataSetLeft = data;
            labelSetLeft = label;
    end
end
