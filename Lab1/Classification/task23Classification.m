%% Initialize
clear;
close all;
clc;

%% Divide training data and test data
column_2c=readtable('column_2C.dat');
categories=table2cell(column_2c(:,end));
categories_num=grp2idx(categories);

featuren=table2array(column_2c(:,1:6));

% Normalized between 0 and 1
temp=mapminmax(featuren',0,1);
feature=temp';

X=feature(1:300,5:6);
Y=categories_num(1:300);

rand_num=randperm(300);
X_train=X(rand_num(1:240),:);
Y_train=Y(rand_num(1:240),:);

X_test=X(rand_num(241:end),:);
Y_test=Y(rand_num(241:end),:);

%% Task 2 : inner-fold cross-validation 

%% inner-fold cross-validation best optimal hyperparameters : kernel == rbf 
%% Write the log in the file
diary rbfInnerOptimiseClassification.log
bestModel_inner = InnerOptimiseClass(X_train,Y_train,'rbf');
bestModel_inner_rbf = bestModel_inner;

bestModel_inner_accuracy_rbf = sum(predict(bestModel_inner_rbf,X_test) == Y_test)/length(Y_test)*100;
support_vectors_num_rbf = length(bestModel_inner_rbf.SupportVectors);
data_availabl_rate_rbf = support_vectors_num_rbf/length(X_train)*100;
ANSWER = ['Use RBF kernel method, the number of support vecorts is :  ',num2str(support_vectors_num_rbf), ', absolute terms and in terms of a % training data available is: ',num2str(data_availabl_rate_rbf),'% , accuracy: ',num2str(bestModel_inner_accuracy_rbf)];
disp(ANSWER)
diary off

%% inner-fold cross-validation best optimal hyperparameters : kernel == Polynomial
diary PolInnerOptimiseClassification.log
bestModel_inner = InnerOptimiseClass(X_train,Y_train,'Polynomial');
bestModel_inner_Polynomial = bestModel_inner;

bestModel_inner_accuracy_Polynomial = sum(predict(bestModel_inner_Polynomial,X_test) == Y_test)/length(Y_test)*100;
support_vectors_num_Polynomial = length(bestModel_inner_Polynomial.SupportVectors);
data_availabl_rate_Polynomial = support_vectors_num_Polynomial/length(X_train)*100;
ANSWER = ['Use Polynomial kernel method, the number of support vecorts is :  ',num2str(support_vectors_num_Polynomial), ', absolute terms and in terms of a % training data available is: ',num2str(data_availabl_rate_Polynomial),'% , accuracy: ',num2str(bestModel_inner_accuracy_Polynomial)];
disp(ANSWER)
diary off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Task 3 : 10-fold cross-validation
%% cross-validation best optimal hyperparameters : kernel == rbf 
diary rbf10CorssFoldClassification.log
ClassificationRate= KCrossValidation(X_train,Y_train,10,'rbf');
ClassificationRate_rbf = ClassificationRate;
ANSWER = ['Use RBF kernel method, the ClassificationRate is :  ',num2str(ClassificationRate_rbf)];
disp(ANSWER)

diary off

%%  cross-validation best optimal hyperparameters : kernel == Polynomial
diary Pol10CorssFoldClassification.log
ClassificationRate = KCrossValidation(X_train,Y_train,10,'Pol');
ClassificationRate_Pol = ClassificationRate;
ANSWER = ['Use Polynomial kernel method, the ClassificationRate is :  ',num2str(ClassificationRate_Pol)];
disp(ANSWER)
diary off

%% cross-validation best optimal hyperparameters : kernel == linear
diary lin10CorssFoldClassification.log
ClassificationRate = KCrossValidation(X_train,Y_train,10,'lin');
ClassificationRate_lin = ClassificationRate;
ANSWER = ['Use linear kernel method, the ClassificationRate is :  ',num2str(ClassificationRate_lin)];
disp(ANSWER)
diary off




