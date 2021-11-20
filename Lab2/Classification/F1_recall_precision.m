function [F1,recal, preci,accu] = F1_recall_precision(targets, outputs,class)
tp = 0;
fp = 0;
tn = 0;
fn = 0;
if ischar(class)
    for i = 1:size(targets,2)
        if strcmp(targets(i),class) && strcmp(outputs(i).calss)
            tp = tp+1;
        elseif strcmp(targets(i),class) && ~strcmp(outputs(i).calss)
            fn = fn+1;
        elseif ~strcmp(targets(i),class) && strcmp(outputs(i).calss)
            fp = fp+1;
        else
            tn = tn+1
        end
    end
end
if isnumeric(class)
    s = size(targets,2)
    for i = 1:s
        if targets(i) == class && outputs(i) == class
            tp = tp+1;
        elseif targets(i) == class && outputs(i) ~= class
            fn = fn+1;
        elseif targets(i) ~= class && outputs(i) == class
            fp = fp+1;
        else
            tn = tn+1
        end
    end

end
    accu = (tp+fp)/(tp+fp+tn+fn);
    recal = tp/(tp+fn)
    preci = tp/(tp+fp); 
    F1 = 2*preci*recal/(preci+recal);
end


%% Test Code
%targett = [1,2,3,4,5,4,5,4,5]
%predict = [1,2,5,4,5,5,5,4,5]
%[f1,recall,precision,accuracy] = F1_recall_precision(targett, predict, 5)
