%calculate the gain
function [gain,split_num,feature_data1,feature_data2] = getGain(entropy,data,column)
    [m,n]=size(data);
    feature=data(:,column);
    %feature_proc=zeros(2,2);
    split_num = mean(feature);
    feature_data1=[];
    feature_data2=[];
    feature_row1 = 1;
    feature_row2 = 1;
    for i = 1:m 
        if feature(i) > split_num
            feature_data1(feature_row1,:)=data(i,:);
            feature_row1 = feature_row1 +1 ;

        else
            feature_data2(feature_row2,:)=data(i,:);
            feature_row2 = feature_row2 +1 ;
        end
    
    end
    gain=entropy -  (feature_row1-1)/m*getEntropy(feature_data1) - (feature_row2-1)/m*getEntropy(feature_data2);
    
end

