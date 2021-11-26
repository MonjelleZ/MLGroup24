%% 
%% function inner-fold cross-validation
%  Task : finding the optimal hyperparameters.
%    BoxConstraint    KernelScale
%    _____________    ___________
%
%        16.01          0.71858 


%% function Regression
function bestModel_inner = InnerOptimiseReg(dataSet,lebal,Kernel)
    X1_martix=dataSet(1:length(dataSet)/2,:);
    Y1_martix=lebal(1:length(lebal)/2,:);

    X2_martix=dataSet(length(dataSet)/2+1:end,:);
    Y2_martix=lebal(length(lebal)/2+1:end,:);

    bestModel = RegOptimise(X1_martix,Y1_martix,X2_martix,Y2_martix,Kernel);
    BM1 = bestModel;
    Dif1= predict(BM1,X2_martix)- Y2_martix;
    m= find(isnan(Dif1));
    Dif1(m,:)=[];
    RMSE1 = sqrt(sum(Dif1.*Dif1));
    SP_Per_BM1 = length(BM1.SupportVectors)/length(X1_martix)*100;

    bestModel = RegOptimise(X2_martix,Y2_martix,X1_martix,Y1_martix, Kernel);
    BM2 = bestModel;
    Dif2= predict(BM1,X2_martix)- Y2_martix;
    m= find(isnan(Dif2));
    Dif2(m,:)=[];
    RMSE2 = sqrt(sum(Dif2.*Dif2));
    SP_Per_BM2 = length(BM2.SupportVectors)/length(X2_martix)*100;

    if RMSE1 < RMSE2
        bestModel_inner = BM1;
        bestRMSE = RMSE1_rbf;
    elseif RMSE1 > RMSE2
        bestModel_inner = BM2;
        bestRMSE = RMSE2;
    else 
        bestModel_inner = BM1;
        bestRMSE = RMSE1;
    end
    ANSWER1 = ['Use ',Kernel, ' kernel method, the number of support vecorts in BM1 is :  ',num2str(length(BM1.SupportVectors)),', absolute terms and in terms of a % training data available is: ',num2str(SP_Per_BM1),'%'];
    disp(ANSWER1)
    ANSWER2 = ['Use ',Kernel, ' kernel method, the number of support vecorts in BM2 is :  ',num2str(length(BM2.SupportVectors)), ', absolute terms and in terms of a % training data available is: ',num2str(SP_Per_BM2),'%'];
    disp(ANSWER2)
    ANSWER = ['the best hyperparameters of ',Kernel, ' Model : C :',num2str(bestModel_inner.BoxConstraints(1)),', Sigma:', num2str(bestModel_inner.KernelParameters.Scale),', RMSE: ',num2str(bestRMSE(1)),', Epsilon:',num2str(bestModel_inner.Epsilon)];
    disp(ANSWER)
    %disp('the best hyperparameters of RBF Model : C : %f, Sigma: %f \n' bestModel_inner_rbf.BoxConstraints(1) bestModel_inner_rbf.KernelParameters.Scale)

end


%% 

function bestModel = RegOptimise(dataSet,label,testset,testlabel, Kernel)
  %BoxConstraint  [1e-3:0.1:1e3];
  c = [35:1:45];
  %KernelScale  [1e-3:0.1:1e3];
  d = [0.1:0.2:2];
  %PolynomialOrder
  p =[1:2];
  %Epsilon
  e=[0.5:0.1:1];
  resultLoss = 100;
  for i = 1:length(c)
      for o= 1:length(e)
          if strcmp(Kernel,'rbf')
              for j = 1:length(d)
                    Mdl = fitrsvm(dataSet,label,'KernelFunction',Kernel,'KernelScale',d(j),'BoxConstraint',c(i),'Epsilon',e(o));
                    Dif11= predict(Mdl,testset)- testlabel;
                    m= find(isnan(Dif11));
                    Dif11(m,:)=[];
                    RMSE11 = sqrt(sum(Dif11.*Dif11));
                    %ANSWER = ['C :',num2str(c(i)),' , Sigma: ', num2str(d(j)),' , resultLoss: ',num2str(resultLoss),' , Support Vectors: ',num2str(length(Mdl.SupportVectors)) ,', Epsilon:',num2str(e(m))];
                    %disp(ANSWER);
                    if RMSE11<resultLoss
                        resultLoss = RMSE11;
                        bestModel=Mdl;
                    end
              end
          elseif strcmp(Kernel,'Polynomial')
               for k = 1:length(p)
                   Mdl = fitrsvm(dataSet,label,'KernelFunction',Kernel,'BoxConstraint',c(i),'PolynomialOrder',p(k),'Epsilon',e(o));
                   Dif11= predict(Mdl,testset)- testlabel;
                   m= find(isnan(Dif11));
                   Dif11(m,:)=[];
                   RMSE11 = sqrt(sum(Dif11.*Dif11));
                   %ANSWER = ['C : ',num2str(c(i)),' , Sigma: ', num2str(d(j)),' ,PolynomialOrder: ',num2str(p(k)),' , resultLoss: ',num2str(resultLoss) , ' , Support Vectors: ',num2str(length(Mdl.SupportVectors)),', Epsilon:',num2str(e(m))];
                   %disp(ANSWER);
                   if RMSE11<resultLoss
                       resultLoss = RMSE11;
                       bestModel=Mdl;
                   end
               end

           elseif strcmp(Kernel,'linear')
                Mdl = fitrsvm(dataSet,label,'KernelFunction',Kernel,'BoxConstraint',c(i),'Epsilon',e(o));
                Dif11= predict(Mdl,testset)- testlabel;
                m= find(isnan(Dif11));
                Dif11(m,:)=[];
                RMSE11 = sqrt(sum(Dif11.*Dif11));
                %ANSWER = ['C : ',num2str(c(i)), ', resultLoss: ', num2str(resultLoss),' , Support Vectors: ', num2str(length(Mdl.SupportVectors)),', Epsilon:',num2str(e(m)) ];
                %disp(ANSWER);
                if RMSE11<resultLoss
                   resultLoss = RMSE11;
                   bestModel=Mdl;
                end
            else
                fprintf(" kernel method error")
                break;
           end % end of kernel function
       end
   end
end

