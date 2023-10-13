function H = ComputeH(X,rule)
    [N, d] = size(X);
    nRules = size(rule.width,1);
    a = zeros(nRules, N, d);
    
    for ii = 1:nRules
        for jj = 1:d
                result=exp(-((X(:,jj)-rule.center(ii,jj))/rule.width(ii,jj)).^2);
                result(result<1e-6)=1e-6;
                a(ii, :, jj) = result;
        end
    end
    
    w = prod(a, 3); %Computes the unnormalized firing strengths
    
    w_hat = w./(repmat(sum(w, 1), nRules, 1)); %Normalizes the firing strengths
    
    H = [];
    for c = 1:size(w_hat, 1)
        H = [H, repmat(w_hat(c, :)', 1, d+1).*[ones(N, 1) X]]; %Computes the hidden matrix
    end
end
