function rule = kmeans_rule(X,nRules)
    [idx, center] = kmeans(X,nRules);
    rule = struct('center', [], 'width', [], 'conq', []);
    rule.center = center;

    for i=1:nRules
        cind=find(idx==i)';
        [n,m]=size(X(cind,:));
        
        if n<=1
            new_width=zeros(1,m);
        else
            new_width=std(X(cind,:));
        end
        rule.width = [rule.width; new_width];
    end
    
    rule.width(rule.width<=1e-3)=1e-3;
end
