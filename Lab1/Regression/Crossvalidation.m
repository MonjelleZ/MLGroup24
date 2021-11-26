%cross validation
function [lostfunction,BM]=Crossvalidation(traindata,labels,K,Kernel)
    lostfunction=0;
    for i = 1:K
        RMSEt=0;
        Knum=size(traindata,1)/K;
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
        if Kernel=='rbf'
            bestModel_inner = InnerOptimiseReg(trainset,trainlabel,'rbf');
            bestModel_inner_rdf = bestModel_inner;
        
        elseif Kernel=='Pol'
            bestModel_inner = InnerOptimiseReg(trainset,trainlabel,'Polynomial');
            bestModel_inner_Polynomial = bestModel_inner;
        
        elseif Kernel=='lin'
            bestModel_inner = InnerOptimiseReg(trainset,trainlabel,'linear');
            bestModel_inner = bestModel_inner;
        end
        BM=bestModel_inner;
        Dif= predict(BM,vaildset)- vaildlabel;
        m= find(isnan(Dif));
        Dif(m,:)=[];
        RMSE = sqrt(sum(Dif.*Dif));
        RMSEt=RMSEt+RMSE;
    end
    lostfunction= (1/K)*RMSEt;

end
