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

dateset=table2array(column_2c(:,1:6));

dateset(:,7)=categories_num(:,1);

rand_num=randperm(300);
trainset=dateset(rand_num(1:250),:);
testset = dateset(rand_num(251:end),:);



%% decision tree
[tree] = CreateTreeClassification(trainset);

%[entropy] = getEntropy(trainset);
%[gain] = getGain(entropy,trainset,1);





