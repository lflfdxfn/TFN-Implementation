function [P,center,U,obj_fcn] = ComputeP(X,nRules)
    [n_sample,n_dim] = size(X);
    [center,U,obj_fcn] = fcm(X,nRules,[NaN, NaN, NaN, 0]);
    U = U';
    P = [];
    for i = 1:nRules
        P = [P, repmat(U(:,i), 1, n_dim+1).*[ones(n_sample,1), X]];
    end
end