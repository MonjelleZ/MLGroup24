function [ variances ] = var( label )
    m = length(label);
    dataVar = var(label);
    variances = dataVar * m;
end