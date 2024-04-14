function scaling = compute_scaling(FeatsMatrix, type)
    FeatsMatrix = full(FeatsMatrix);
    scaling = [];

    if(strcmp(type, 'zero_one'))
        scaling = zero_one_scaling(FeatsMatrix);
        scaling.type = 'zero_one';
    elseif(strcmp(type,'norm_1'))
        scaling = [];
    elseif(strcmp(type, 'zscore'))
        m = mean(FeatsMatrix,2);
        sigma = std(FeatsMatrix,[],2);
        scaling.to_subtract = m;
        scaling.to_divide = 1./(sigma');
        scaling.total_mean = m * size(FeatsMatrix,2);
        scaling.total_std = sigma .* sigma * (size(FeatsMatrix,2)-1);
        scaling.type = 'zscore';
        scaling.n = size(FeatsMatrix,2);
    elseif(strcmp(type, 'none'))
        scaling = [];
    end
end
