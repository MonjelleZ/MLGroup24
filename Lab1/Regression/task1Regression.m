%% Initialize
warning off 
clear;
close all;
clc;
%% testing Epsilon

autompg=readtable('auto-mpg.dat');
categories=table2array(autompg(:,1));
categories_num=length(categories);

featuren=table2array(autompg(:,2:8));

temp=mapminmax(featuren',0,1); 
feature=temp';

X=feature(:,3:4);
Y=categories(:);

rand_num=randperm(398);
X_train=X(rand_num(1:398),:);
Y_train=Y(rand_num(1:398),:);

for i=1:length(X_train)
    e = [0.1:0.5:20];
    Mdl = fitrsvm(X_train,Y_train,'Standardize',true,'KernelFunction','linear','Epsilon',e(i));
    Dif= predict(Mdl,X_train)- Y_train;
    m= find(isnan(Dif));
    Dif(m,:)=[];
    RMSE = sqrt(sum(Dif.*Dif));
    ANSWER1 = ['Epsilon:',num2str(e(i)),' RMSE: ',num2str(RMSE)];
    SP_Per_Mdl = length(Mdl.SupportVectors)/length(X_train)*100;
    ANSWER1 = ['Epsilon:',num2str(e(i)),' RMSE: ',num2str(RMSE), ' the number of support vecorts in Mdl is : ',num2str(length(Mdl.SupportVectors)),', absolute terms and in terms of a % training data available is: ',num2str(SP_Per_Mdl),'%'];
    disp(ANSWER1)
end
