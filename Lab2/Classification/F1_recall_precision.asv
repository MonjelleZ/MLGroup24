function [F1,Recall, Precision,Accuracy] = Measure(targets, outputs)
[c,cm,ind,per] = confusion(targets,outputs);
for i = 1:size(cm,1)
    tp = cm(i,i);
    fp = 0;
    tn = 0;
    fn = 0;
    for j = 1:size(cm,1)
        if j ~= i
            tn = tn+cm(j,j);
            fp = fp+cm(j,i);
            fn = fn+cm(i,j);
        end
    end
    accuracy = (tp+fp)/(tp+fp+tn+fn);
    recall = tp/(tp+fn)
    precision = tp/(tp+fp); 
    f1 = 2*precision*recall/(precision+recall);
    
    % add results for class i to the finall result list
    F1 = [F1,f1];
    Recall = [Recall,recall];
    Precision = [Precision,Precision];
    Accuracy = [Accuracy,accuracy];
end

end