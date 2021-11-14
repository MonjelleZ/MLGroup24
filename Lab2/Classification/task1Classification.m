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

X=feature(1:300,1:6);
Y=categories_num(1:300);

rand_num=randperm(300);

X_train=X(rand_num(1:240),:);
Y_train=Y(rand_num(1:240),:);

X_test=X(rand_num(241:end),:);
Y_test=Y(rand_num(241:end),:);

%% decision tree and view
ctree = ClassificationTree.fit(X_train,Y_train);
view(ctree);
view(ctree,'Mode','graph');


%% predict
pre = predict(ctree,X_test);

%% evalute of predict result 
accury = sum(pre == Y_test)/length(Y_test)*100;


%% 叶子节点含有的最小样本数对决策树性能的影响 :22
leafs = logspace(1,2,10);
N = numel(leafs);
err = zeros(N,1);
for n = 1:N
    t = ClassificationTree.fit(X_train,Y_train,'crossval','on','minleaf',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('叶子节点含有的最小样本数');
ylabel('交叉验证误差');
title('叶子节点含有的最小样本数对决策树性能的影响');


%% decision tree and view
Optimaltree = ClassificationTree.fit(X_train,Y_train,'minleaf',13);
view(Optimaltree,'mode','graph');

% 优化后决策树的重采样误差和交叉验证误差
resubOpt = resubLoss(Optimaltree);
lossOpt = kfoldLoss(crossval(Optimaltree));
% 没有优化的误差
resubDefault = resubLoss(ctree);
lossDefault = kfoldLoss(crossval(ctree));

%% 减枝
[~,~,~,bestlevel] = cvloss(ctree,'subtrees','all','treesize','min');
cptree = prune(ctree,'Level',bestlevel);
view(cptree,'mode','graph');


