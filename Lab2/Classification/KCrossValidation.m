%% K-cross validation
function ClassificationRate= KCrossValidation(data,K)
    %ClassificationRate=0;
    accuracysum=0;
    for i = 1:K
        feature_used = [];
        begin = ['******************************* K-fold cross-validation ： ' , num2str(i), ' ******************************* '];
        disp(begin)
        Knum=size(data,1)/K;
        vaildset=data(1+(i-1)*Knum:i*Knum,:);
        %vaildlabel=labels(1+(i-1)*Knum:i*Knum,:);
        if i==1
            trainset=data(1+i*Knum:end,:);
            %trainlabel=labels(1+i*Knum:end,:);
        elseif i==10
            trainset=data(1:(i-1)*Knum,:);
            %trainlabel=labels(1:(i-1)*Knum,:);
        else
            trainset=data([1:(i-1)*Knum i*Knum+1:end],:);
            %trainlabel=labels([1:(i-1)*Knum i*Knum+1:end],:);
        end
        [tree] = CreateTreeClassification(trainset,feature_used);
        predictedSet = predictTree(tree,vaildset(:,1:6))';
        realSet = vaildset(:,7);
        accuracy = sum(predictedSet == realSet)/length(predictedSet)*100;
        accuracysum = accuracy + accuracysum;
    end
    ClassificationRate = accuracysum / K;
    answer = ['The average classification rate : ', num2str(ClassificationRate),'%' ];
    disp(answer)
end
