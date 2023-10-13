function rule = fcm_rule(X,nRules,alpha)
    [center,U] = fcm(X,nRules,[alpha,100,1e-5,0]);
    [~,N]=size(U);
    rule = struct('center', [], 'width', [], 'conq', []);
    rule.center=center;
    
    for i=1:nRules
        mk=center(i,:);
        uk=U(i,:);
        sigma=((uk*(X-repmat(mk,N,1)).^2)./sum(uk)).^0.5;
        rule.width=[rule.width;sigma];
    end
    
    rule.width(rule.width<=1e-3)=1e-3;%%%%%%%%%%
end
