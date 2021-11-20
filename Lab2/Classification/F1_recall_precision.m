function [F1,recal, preci,accu] = F1_recall_precision(targets, outputs)

%  targets : real dataset
%  outputs : predict dataset

tp = 0;
fp = 0;
tn = 0;
fn = 0;


s = size(targets,1);
  for i = 1:s
        if targets(i) == 1 && outputs(i) == 1
            tp = tp+1;
        elseif targets(i) == 1 && outputs(i) ~= 1
            fn = fn+1;
        elseif targets(i) ~= 1 && outputs(i) == 1
            fp = fp+1;
        else
            tn = tn+1;
        end
  end

    accu = (tp+tn)/(tp+fp+tn+fn);
    recal = tp/(tp+fn);
    preci = tp/(tp+fp); 
    F1 = 2*preci*recal/(preci+recal);
end


%% Test Code
%targett = [1,2,3,4,5,4,5,4,5]
%predict = [1,2,5,4,5,5,5,4,5]
%[f1,recall,precision,accuracy] = F1_recall_precision(targett, predict, 5)
