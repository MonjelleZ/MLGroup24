% the categories Disk Hernia and Spondylolisthesis were merged into a single category labelled as 'abnormal'.
%   NO:   Normal (100 patients)  0
%   AB:   Abnormal (210 patients). 1

%
%     feature name of column 1-6:
%           1. Incidence of pelvis : IOP
%           2. pelvic inclination : PI
%           3. lumbar lordosis Angle : LLA
%           4. sacral slope : SS
%           5. pelvic radius : PR
%           6. degree of lumbar spondylolisthesis : DOLS


%% Initialize
clear;
close all;
clc;

%% Divide training data and test data
column_2c=readtable('column_2C.dat');
categories=table2cell(column_2c(:,end));
categories_num=grp2idx(categories);
for i=1:length(categories_num(:,end))
    if categories_num(i) == 2
        categories_num(i) = 0;
    end
end


dateset=table2array(column_2c(:,1:6));

dateset(:,7)=categories_num(:,1);

rand_num=randperm(310);
trainset=dateset(rand_num(1:250),:);
testset = dateset(rand_num(251:end),:);



%% generate decision tree
feature_used = [];
[tree] = CreateTreeClassification(trainset,feature_used);

%% draw decidion tree
DrawDecisionTree(tree,'Decision Tree of Vertebral Classification')

%% cross validatioin, K = 10
ClassificationRate= KCrossValidation(dateset(rand_num(1:end),:),10);



