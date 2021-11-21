function [predictedSet] = predictTree( trainedTree,testSet )

    m= length(testSet);

    if isempty(trainedTree.kids)
        predictedSet = trainedTree.prediction;
        return
    end

    for i = 1:m
        sample = testSet(i,:);
        if sample(trainedTree.attribute) > trainedTree.threshold
            predictedSet(i) = predictTree(trainedTree.kids{1,1},sample );
        else
            predictedSet(i) = predictTree(trainedTree.kids{1,2},sample );
        end 
    end
end