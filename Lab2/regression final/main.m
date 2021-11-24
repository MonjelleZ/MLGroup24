%% Initialize
warning off 
clear;
close all;
clc;

%% Divide training data and test data
autompg=readtable('auto-mpg.dat');
categories=table2array(autompg(:,1));
categories_num=length(categories);

featuren=table2array(autompg(:,2:8));

temp=mapminmax(featuren',0,1);%
feature=temp';

X=feature(:,2:5);
Y=categories(:);


X_train=X(1:390,:);
Y_train=Y(1:390,:);

RN =30;

k = 10;

feature = [];
tree = createTree(X_train,Y_train,RN, feature);


DrawDecisionTree(tree);


[RMSEtrain,RMSEtest]=CrossVlidation(X_train,Y_train,RN,k,feature);

RMSETrain = RMSEtrain*(1/k);
RMSETest = RMSEtest*(1/k);
fprintf('===========RMSE=============\n');
fprintf('RMSE on TrainSet %f\n', RMSETrain);
fprintf('RMSE on TestSet %f\n', RMSETest);