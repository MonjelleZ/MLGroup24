% the categories Disk Hernia and Spondylolisthesis were merged into a single category labelled as 'abnormal'.
%   NO:   Normal (100 patients) 
%   AB:   Abnormal (210 patients). 

%% Initialize
clear;
close all;
clc;

%% Divide training data and test data
column_2c=readtable('column_2C.dat');
categories=table2cell(column_2c(:,end));
categories_num=grp2idx(categories);

feature=table2array(column_2c(:,1:6));

% Normalized between 0 and 1
%temp=mapminmax(feature',0,1);
%featuren=temp';

X=feature(1:300,5:6);
Y=categories_num(1:300);

rand_num=randperm(300);

X_train=X(rand_num(1:240),:);
Y_train=Y(rand_num(1:240),:);

X_test=X(rand_num(241:end),:);
Y_test=Y(rand_num(241:end),:);

%% Task 1: Train SVMs with the linear kernel， Use column 5, 6 of feature
SVMModel=fitcsvm(X,Y,'KernelFunction','linear', 'BoxConstraint',1);
sv=SVMModel.SupportVectors;
accuracy2_rbf = sum(predict(SVMModel,X_test) == Y_test)/length(Y_test)*100;

%% figure
figure
gscatter(X(:,1),X(:,2),Y);
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
hold on

X_ = 0:1:200;
a = - SVMModel.Beta(1)/SVMModel.Beta(2); 
b = - SVMModel.Bias/SVMModel.Beta(2); 
Y_ = a*X_ + b;   
Y2 = Y_+12;
Y3 = Y_-12;
plot(X_,Y_,'k-',X_,Y2,'k--',X_,Y3,'k--','MarkerSize',10)
legend('AB','NO','Support Vector')
title('SVM Liner ')




