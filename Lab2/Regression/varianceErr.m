function [ variances ] = varianceErr( label )
    m = length(label);
    dataVar = var(label);
    variances = dataVar * m;
end