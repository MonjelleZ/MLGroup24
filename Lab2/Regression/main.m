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

rand_num=randperm(398);
X_train=X(rand_num(1:350),:);
Y_train=Y(rand_num(1:350),:);

X_test=X(rand_num(351:398),:);
Y_test=Y(rand_num(351:398),:);


regPurging = struct('tolS', 1, 'tolN', 50);

k = 10;

% The Entrance function
regMain(X_train,Y_train,X_test,Y_test, regPurging);