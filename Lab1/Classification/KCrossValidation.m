%% K-cross validation
function ClassificationRate= KCrossValidation(traindata,labels,K,Kernel)
    %ClassificationRate=0;
    accurysum=0;
    for i = 1:K
        begin = ['******************************* K-fold cross-validation ： ' , num2str(i), ' ******************************* '];
        disp(begin)
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
        if strcmp(Kernel,'rbf')
            bestModel_final = InnerOptimiseClass(trainset,trainlabel,'rbf');        
        elseif strcmp(Kernel,'Pol')
            bestModel_final = InnerOptimiseClass(trainset,trainlabel,'Polynomial');        
        elseif strcmp(Kernel,'lin')
            bestModel_final = InnerOptimiseClass(trainset,trainlabel,'Linear');
        end
        accuracy = sum(predict(bestModel_final,vaildset) == vaildlabel)/length(vaildlabel)*100;
        accurysum = accuracy + accurysum;
    end
    ClassificationRate = accurysum / K;
    answer = ['The average classification rate for ', Kernel , ' : ', num2str(ClassificationRate) ];
    disp(answer)
end
