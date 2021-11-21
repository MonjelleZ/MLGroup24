% Regression Tree Main Scripts
function regMain(X_train,Y_train,X_test,Y_test, regPurging,K)

% Set Parameters
tolS = regPurging.tolS;
tolN = regPurging.tolN;

feature_used = [];
m=length(X_train);
n=size(X_train,2);
tree = createTree(X_train,Y_train, tolS, tolN, feature_used);


DrawDecisionTree(tree);

% Show the RMSE
fprintf('======calculate the RMSE of the Reg Tree======\n');
trainedTree = tree;
% RMSE on TrainSet
predictedTrainSet = predictTree( trainedTree,X_train );
realSet = Y_train;
rmseValueTrain = calRMSE( realSet,predictedTrainSet' );
fprintf('RMSE on TrainDataSet %f\n', rmseValueTrain);

% RMSE on testSet
predictedTestSet = predictTree( trainedTree,X_test );
realSet = Y_test;
rmseValueTest = calRMSE( realSet,predictedTestSet' );
fprintf('RMSE on TestDataSet %f\n', rmseValueTest);



fprintf('======10-fold Cross validation Start======\n');
[RMSEtrain,RMSEtest]=kCrossV(X_train,Y_train,K);



% Calculate the mean RMSE value from cross validation 
finalRMSETrain = mean(RMSEtrain);
finalRMSETest = mean(RMSEtest);
fprintf('======Mean RMSE======\n');
fprintf('Mean RMSE on TrainSet %f\n', finalRMSETrain);
fprintf('Mean RMSE on TestSet %f\n', finalRMSETest);

end


fprintf('======Mission Complete======\n');

end