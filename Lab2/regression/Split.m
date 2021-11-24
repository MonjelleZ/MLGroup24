function [ dataSetLeft,labelSetLeft, dataSetRight,labelSetRight] = Split( data,label, feature, value )
  
    m=length(data);
    DataTemp = data(:,feature)';
    
    index = 1:m;
    Left = index(DataTemp > value);
    Right = index(DataTemp <= value);
    
    nLeft = size(Left,2);
    nRight = size(Right,2);
    
    if nLeft>0 && nRight>0
        dataSetLeft = data(Left,:);
        labelSetLeft= label(Left,:);
        dataSetRight = data(Right,:);
        labelSetRight = label(Right,:);
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