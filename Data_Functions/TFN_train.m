function [test_auc, test_pr, train_time, test_time] = TFN_train(n_rule, trainData, testData, TRAIN, REGU, MIXUP, AUG)
    n_feature=size(trainData.X,2);
    
    % Augmentation part
    [X1,X2,Y]=augmentation(trainData, AUG.num_train, AUG.c_values(1), AUG.c_values(2), AUG.c_values(3));  
    % Prepare for mixup
    unlabel_X=trainData.X(trainData.Y==0,:);
    
    % train time begin
    tic;
    
    % Calculate the parameters w in consequent layer    
    if strcmp(TRAIN.cluster, 'k-means')
        rule = kmeans_rule(trainData.X, n_rule);

        HL=ComputeH(X1,rule);
        HR=ComputeH(X2,rule);
        H=HL+HR;
        
        % Mixup-term, B
        s=rng;% seperate the mixup part from this whole experiment
        if strcmp(MIXUP.type,'No')
            B=0;   
        elseif strcmp(MIXUP.type, 'ICR')
            [LM1,LM2,LM_mixup,RM1,RM2,RM_mixup,ratio] = ICR_mixup(unlabel_X,MIXUP.M);            
            HM_L0=ComputeH(LM_mixup,rule);
            H_L1=ComputeH(LM1,rule);
            H_L2=ComputeH(LM2,rule);
            HM_R0=ComputeH(RM_mixup,rule);
            H_R1=ComputeH(RM1,rule);
            H_R2=ComputeH(RM2,rule);
            BL=HM_L0-ratio.*H_L1-(1-ratio).*H_L2;
            BR=HM_R0-ratio.*H_R1-(1-ratio).*H_R2;
            B=BL+BR;
        end
        
        rng(s);
        
        % Calculate the w
        [~,p]=size(H);
        w=(H'*H+REGU.lambda*eye(p)+MIXUP.gamma*(B'*B))\(H'*Y);
        train_time=toc;
        
        tic;
        rule.conq=reshape(w,n_feature+1,n_rule)';        
        y_hat_test=KTFN_test(trainData,testData,rule,AUG.E_test);
        [~,~,~,test_auc]=perfcurve(testData.Y,y_hat_test,1);
        [~,~,~,test_pr]=perfcurve(testData.Y,y_hat_test,1,'xCrit','reca','yCrit','prec');
        test_time=toc;

    elseif strcmp(TRAIN.cluster, 'p_fcm')
        % Calculate the shared parameters, center.
        [~,center]=ComputeP([X1;X2],n_rule);
        [P1,~] = ComputeP_test(X1,n_rule,center);
        [P2,~] = ComputeP_test(X2,n_rule,center);
        H=P1+P2;
        
        % Mixup-term, B
        s=rng;% seperate the mixup part from this whole experiment
        if strcmp(MIXUP.type,'No')
            B=0;   
        elseif strcmp(MIXUP.type, 'ICR')
            [LM1,LM2,LM_mixup,RM1,RM2,RM_mixup,ratio] = ICR_mixup(unlabel_X,MIXUP.M);        
            HM_L0=ComputeP_test(LM_mixup,n_rule,center);
            H_L1=ComputeP_test(LM1,n_rule,center);
            H_L2=ComputeP_test(LM2,n_rule,center);
            HM_R0=ComputeP_test(RM_mixup,n_rule,center);
            H_R1=ComputeP_test(RM1,n_rule,center);
            H_R2=ComputeP_test(RM2,n_rule,center);
            BL=HM_L0-ratio.*H_L1-(1-ratio).*H_L2;
            BR=HM_R0-ratio.*H_R1-(1-ratio).*H_R2;
            B=BL+BR;
        end
        
        rng(s);
        
        if strcmp(MIXUP.type, 'mixup')
            [~,p]=size(H);
            w=(H'*H+REGU.lambda*eye(p)+MIXUP.gamma*(B'*B))\(H'*Y+MIXUP.gamma*B'*mix_Y');
        elseif strcmp(MIXUP.type, 'ICR')
            [~,p]=size(H);
            w=(H'*H+REGU.lambda*eye(p)+MIXUP.gamma*(B'*B))\(H'*Y);
        end
        train_time=toc;

        tic;
        y_hat_test=PTFN_test(trainData,testData,n_rule,center,AUG.E_test,w);
        [~,~,~,test_auc]=perfcurve(testData.Y,y_hat_test,1);   
        [~,~,~,test_pr]=perfcurve(testData.Y,y_hat_test,1,'xCrit','reca','yCrit','prec');
        test_time=toc;
        
    end    
end

function [LM1,LM2,LM_mixup,RM1,RM2,RM_mixup,ratio] = ICR_mixup(X,M)
    ratio=betarnd(0.5,0.5,[M,1]);
    LM1=randSampBack(X,M);
    LM2=randSampBack(X,M);
    LM_mixup=LM1.*ratio+LM2.*(1-ratio);
    
    RM1=randSampBack(X,M);
    RM2=randSampBack(X,M);
    RM_mixup=RM1.*ratio+RM2.*(1-ratio);
end

function [y_hat]=PTFN_test(trainData,testData,n_rule,center,E,w)
    %randomly sample out E unlabeled and labeled data instances
    unlabeled=trainData.X(trainData.Y==0,:);
    labeled=trainData.X(trainData.Y==1,:);        
    E_u=randSamp(unlabeled,E);
    
    if size(labeled,1)<E        
        E_a=randSampBack(labeled,E);
    else        
        E_a=randSamp(labeled,E);
    end
    
    %calculate y_hat for each instances
    [N,~]=size(testData.X);
    
    Eas_H=ComputeP_test(E_a,n_rule,center);
    Eus_H=ComputeP_test(E_u,n_rule,center);
    
    xs_H1=ComputeP_test(testData.X,n_rule,center);
    xs_H2=xs_H1;
    
    y_hat_list=zeros(N,E);
    for i=1:E
        Ea_H=Eas_H(i,:);
        Eu_H=Eus_H(i,:);
        
        repEa_H=repmat(Ea_H,N,1);
        repEu_H=repmat(Eu_H,N,1);
        
        y_hat_list(:,i)=((repEa_H+xs_H2+xs_H1+repEu_H)*w)/2;
    end    
    
    y_hat=mean(y_hat_list,2);   
end

function [y_hat]=KTFN_test(trainData,testData,rule,E)
    %randomly sample out E unlabeled and labeled data instances
    unlabeled=trainData.X(trainData.Y==0,:);
    labeled=trainData.X(trainData.Y==1,:);   
    E_u=randSamp(unlabeled,E);
    
    if size(labeled,1)<E        
        E_a=randSampBack(labeled,E);
    else        
        E_a=randSamp(labeled,E);
    end
    
    %calculate y_hat for each instances
    [N,~]=size(testData.X);
    
    Eas_H=ComputeH(E_a,rule);
    Eus_H=ComputeH(E_u,rule);
    w=reshape(rule.conq',numel(rule.conq),1);   
    
    xs_H1=ComputeH(testData.X,rule);
    xs_H2=ComputeH(testData.X,rule);
    
    y_hat_list=zeros(N,E);
    for i=1:E
        Ea_H=Eas_H(i,:);
        Eu_H=Eus_H(i,:);
        
        repEa_H=repmat(Ea_H,N,1);
        repEu_H=repmat(Eu_H,N,1);
        
        y_hat_list(:,i)=((repEa_H+xs_H2+xs_H1+repEu_H)*w)/2;
    end    
    
    y_hat=mean(y_hat_list,2);   
end

function [sample_data]=randSamp(datamatrix,num_sample)
% Sample specific number of data lines from datamatrix randomly (num_sample<N)
% input: datamatrix, input datamatrix
%        num_sample, required number of sample data
% output: data matrix composed of sampled data lines
    [N,~]=size(datamatrix);
    
    rand_lines=randperm(N);
    sampled=rand_lines(1:num_sample);
    
    sample_data=datamatrix(sampled,:);
end

function [sample_data]=randSampBack(datamatrix,num_sample)
% Sample specific number of data lines from datamatrix randomly
% input: datamatrix, input datamatrix
%        num_sample, required number of sample data
% output: data matrix composed of sampled data lines
    [N,~]=size(datamatrix);
    sampled_index=randi(N,1,num_sample);
    sample_data=datamatrix(sampled_index,:);
end