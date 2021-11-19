%计算信息增益
function [gain,split_num,feature_data1,feature_data2] = getGain(entropy,data,column)
    [m,n]=size(data);
    feature=data(:,column);
    %feature_proc=zeros(2,2);
    split_num = mean(feature);
    big_num = 0;
    feature_data1=[];
    feature_data2=[];
    small_num  = 0;
    feature_row1 = 1;
    feature_row2 = 1;
    for i = 1:m 
        if feature(i) > split_num
            big_num = big_num + 1;
            feature_data1(feature_row1,:)=data(i,:);
            feature_row1 = feature_row1 +1 ;

        else
            small_num = small_num + 1;
            feature_data2(feature_row2,:)=data(i,:);
            feature_row2 = feature_row2 +1 ;
        end
    
    end
    gain=entropy -  feature_row1/m*getEntropy(feature_data1) - feature_row2/m*getEntropy(feature_data2);

    
end



%{
function [gain] = getGain(entropy,data,column)
  [m,n]=size(data);
  feature=data(:,column);
  feature_distinct=unique(feature);
  feature_num=length(feature_distinct);
  feature_proc=zeros(feature_num,2);
  feature_proc(:,1)=feature_distinct(:,1);
  f_entropy=0;
  for i=1:feature_num
    feature_data=[];
    feature_proc(:,2)=0;
    feature_row=1;
    for j=1:m
      if feature_proc(i,1)==data(j,column)
        feature_proc(i,2)=feature_proc(i,2)+1;
      end
      if feature_distinct(i,1)==data(j,column)
        feature_data(feature_row,:)=data(j,:);
        feature_row=feature_row+1;
      end
    end
    f_entropy=f_entropy+feature_proc(i,2)/m*getEntropy(feature_data);
  end
  gain=entropy-f_entropy;
  end
  %}