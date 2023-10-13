function [P,U] = ComputeP_test(X,n_rule,center)
    [n_sample,n_dim] = size(X);
    U = update_u(X, center, n_rule, 2);
    U = U';
    P = []; % (n_sample,(n_dim+1)*nRules)
    for i = 1:n_rule
        P = [P, repmat(U(:,i), 1, n_dim+1).*[ones(n_sample,1), X]];
    end
end

function u = update_u(X, center, n_rule, beta)
    var_x = zeros(n_rule, size(X,1));
    u_ini = zeros(n_rule, size(X, 1));

    for i=1:n_rule
        var_x(i,:)=sqrt(sum((X - center(i,:)).^2, 2));
    end

    for i=1:n_rule
        u_tmp = zeros(1, size(X,1));
        for j=1:n_rule
            var_x_i = var_x(i,:);
            var_x_j = var_x(j,:);
            u_tmp = u_tmp+(var_x_i./var_x_j).^(2/(beta-1));
        end

        u_ini(i,:) = 1./u_tmp;
    end
    u = u_ini;
end
