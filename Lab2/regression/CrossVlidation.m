%cross validation
function [RMSEtrain,RMSEtest]=CrossVlidation(traindata,labels, tolS,tolN,k,feature_used)
    RMSEtrain=0;
    RMSEtest=0;
    for i = 1:k
        Knum=size(traindata,1)/k;
        vaildset=traindata(1+(i-1)*Knum:i*Knum,:);
        vaildlabel=labels(1+(i-1)*Knum:i*Knum,:);
        if i==1
            trainset=traindata(1+i*Knum:end,:);
            trainlabel=labels(1+i*Knum:end,:);
        elseif i==10
            trainset=traindata(1:(i-1)*Knum,:);
            trainlabel=labels(1:(i-1)*Knum,:);
        else
            trainset=traindata([1:(i-1)*Knum i*Knum+1:end],:);
            trainlabel=labels([1:(i-1)*Knum i*Knum+1:end],:);
        end

        fprintf('==Cross Validation: %d\n', i);
        tree = createTree(trainset,trainlabel, tolS, tolN, feature_used);
        %% trainset
        Dif1= predictTree(tree,trainset)- trainlabel';
        m1= find(isnan(Dif1));
        k1= length(trainlabel);
        Dif1(:,m1)=[];
        RMSE1 = sqrt(sum(Dif1.*Dif1)/k1);
        RMSEtrain=RMSEtrain+RMSE1;
        fprintf('RMSE on TrainDataSet %f\n', RMSE1);


        %testset
        Dif2=predictTree(tree,vaildset)- vaildlabel';
        m2= find(isnan(Dif2));
        k2 =length(vaildlabel);
        Dif2(:,m2)=[];
        RMSE2 = sqrt(sum(Dif2.*Dif2)/k2);
        RMSEtest=RMSEtrain+RMSE2;
        fprintf('RMSE on testDataSet %f\n', RMSE2);

    end


end