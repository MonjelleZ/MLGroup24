%% function inner-fold cross-validation for Classification
%  Task : finding the optimal hyperparameters. 
%    BoxConstraint    KernelScale
%    _____________    ___________
%
%        16.01          0.71858 


%% The optimal model is selected based on inner-fold cross-validation
function bestModel_inner = InnerOptimiseClass(dataSet,lebal,kernel)
    
    X1_martix=dataSet(1:length(dataSet)/2,:);
    Y1_martix=lebal(1:length(lebal)/2,:);

    X2_martix=dataSet(length(dataSet)/2+1:end,:);
    Y2_martix=lebal(length(lebal)/2+1:end,:);

    bestModel = ClassOptimise(X1_martix,Y1_martix, kernel);
    BM1 = bestModel;
    accuracy1 = sum(predict(BM1,X2_martix) == Y2_martix)/length(Y2_martix)*100;
    SP_Per_BM1 = length(BM1.SupportVectors)/length(X1_martix)*100;

    bestModel = ClassOptimise(X2_martix,Y2_martix, kernel);
    BM2 = bestModel;
    accuracy2 = sum(predict(BM2,X1_martix) == Y1_martix)/length(Y1_martix)*100;
    SP_Per_BM2 = length(BM2.SupportVectors)/length(X2_martix)*100;

    if accuracy1 > accuracy2
        bestModel_inner = BM1;
        bestaccuracy = accuracy1;
    elseif accuracy1 < accuracy2
        bestModel_inner = BM2;
        bestaccuracy = accuracy2;
    else 
        bestModel_inner = BM1;
        bestaccuracy = accuracy1;
    end

end

% According to kernel method, the optimal hyperparameter of the model is obtained
function bestModel = ClassOptimise(dataSet,lebal,kernel)

  %BoxConstraint  [1e-3:0.1:1e3];
  c = [1:1:20];
  %KernelScale , sigma [1e-3:0.1:1e3];
  sigma = [0.1:0.1:1];
  %PolynomialOrder , q
  q =[2:4];
  resultLoss = 100;

  for i = 1:length(c)
       if strcmp(kernel,'rbf')
           for s = 1:length(sigma)
               Mdl = fitcsvm(dataSet,lebal,'KernelFunction',kernel,'KernelScale',sigma(s),'BoxConstraint',c(i));
               rsl = Mdl.resubLoss;
               ANSWER = ['C :',num2str(c(i)),' , Sigma: ', num2str(sigma(s)),' , resultLoss: ' , num2str(resultLoss),' , Support Vectors: ',num2str(length(Mdl.SupportVectors)) ];
               disp(ANSWER);
               if rsl<resultLoss
                   resultLoss = rsl;
                   bestModel=Mdl;
               end
           end
       elseif strcmp(kernel,'Polynomial')
            for k = 1:length(q)
                Mdl = fitcsvm(dataSet,lebal,'KernelFunction',kernel,'BoxConstraint',c(i),'PolynomialOrder',q(k));
                rsl = Mdl.resubLoss;
                ANSWER = ['C : ',num2str(c(i)),' , PolynomialOrder: ',num2str(q(k)),' , resultLoss: ',num2str(resultLoss) , ' , Support Vectors: ',num2str(length(Mdl.SupportVectors))];
                disp(ANSWER);
                if rsl<resultLoss
                    resultLoss = rsl;
                    bestModel=Mdl;
                end
            end
       elseif strcmp(kernel,'Linear')
            Mdl = fitcsvm(dataSet,lebal,'KernelFunction',kernel,'BoxConstraint',c(i));
            rsl = Mdl.resubLoss;
            ANSWER = ['C : ',num2str(c(i)), ', resultLoss: ', num2str(resultLoss),' , Support Vectors: ', num2str(length(Mdl.SupportVectors)) ];
            disp(ANSWER);
            if rsl<resultLoss
                resultLoss = rsl;
                bestModel=Mdl;
            end
        else
           fprintf(" kernel method error")
           break;

        end % end of kernel function
  end
end
