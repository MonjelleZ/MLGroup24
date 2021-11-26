%% Initialize
warning off 
clear;
close all;
clc;

%% Write the log in the file
diary InnerOptimiseRegression.log

%% Divide training data and test data
autompg=readtable('auto-mpg.dat');
categories=table2array(autompg(:,1));
categories_num=length(categories);

featuren=table2array(autompg(:,2:8));

temp=mapminmax(featuren',0,1);%
feature=temp';

X=feature(:,3:4);
Y=categories(:);

rand_num=randperm(398);
X_train=X(rand_num(1:350),:);
Y_train=Y(rand_num(1:350),:);

X_test=X(rand_num(351:398),:);
Y_test=Y(rand_num(351:398),:);

RMSEtotal1=0;
RMSEtotal2=0;
RMSEtotal3=0;
%% cross-validation best optimal hyperparameters : kernel == rbf 
[lostfunction,BM]=Crossvalidation(X_train,Y_train,10,'rbf');
RMSEtr1=lostfunction;
Dif= predict(BM,X_test)- Y_test;
m= find(isnan(Dif));
Dif(m,:)=[];
RMSE = sqrt(sum(Dif.*Dif));
RMSEtotal1=RMSEtotal1+RMSE;

%%  cross-validation best optimal hyperparameters : kernel == Polynomial
[lostfunction,BM]=Crossvalidation(X_train,Y_train,10,'Pol');
RMSEtr2=lostfunction;
Dif= predict(BM,X_test)- Y_test;
m= find(isnan(Dif));
Dif(m,:)=[];
RMSE = sqrt(sum(Dif.*Dif));
RMSEtotal2=RMSEtotal2+RMSE;

%% cross-validation best optimal hyperparameters : kernel == liner
[lostfunction,BM]=Crossvalidation(X_train,Y_train,10,'lin');
RMSEtr3=lostfunction;
Dif= predict(BM,X_test)- Y_test;
m= find(isnan(Dif));
Dif(m,:)=[];
RMSE = sqrt(sum(Dif.*Dif));
RMSEtotal3=RMSEtotal3+RMSE;


%% close log file

diary off


