function [X1,X2,Y] = augmentation(dataset,num_sample,c1,c2,c3)
% Data augmentation for mixup data pairs
% input: dataset, object of Dataset class 
%        num_sample, desired number of data pairs
%        c1, num of anomaly-anomaly pairs
%        c2, num of anomaly-unlabeled pairs
%        c3, num of unlabeled-unlabeled pairs
% output:pair_dataset, with value of X, Y, name, task
    X=dataset.X;
    Y=dataset.Y;
    
    unlabeled=X(Y==0,:);
    anomaly=X(Y==1,:);
    num_uu=floor(num_sample/2);
    num_au=floor(num_sample/4);
    num_aa=floor(num_sample/4);
    
    X1_u=randSamp(unlabeled,num_uu);
    X1_a=randSamp(anomaly,num_au+num_aa);
    X2_u=randSamp(unlabeled,num_uu+num_au);
    X2_a=randSamp(anomaly,num_aa);
    
    X1=[X1_u;X1_a];
    X2=[X2_u;X2_a];
    
    Y=[c3*ones(num_uu,1);c2*ones(num_au,1);c1*ones(num_aa,1)];
    
    shuffle_X=round(rand(size(X1,1),1));
    X1_raw=X1.*shuffle_X+X2.*(1-shuffle_X);
    X2_raw=X1.*(1-shuffle_X)+X2.*shuffle_X;
    
    shuffle_Y=randperm(size(X1,1));
    X1=X1_raw(shuffle_Y,:);
    X2=X2_raw(shuffle_Y,:);
    Y=Y(shuffle_Y,:);
end

function [sample_data]=randSamp(datamatrix,num_sample)
% Sample specific number of data lines from datamatrix randomly
% input: datamatrix, input datamatrix
%        num_sample, required number of sample data
% output: data matrix composed of sampled data lines
    [N,~]=size(datamatrix);
    sampled_index=randi(N,1,num_sample);
    sample_data=datamatrix(sampled_index,:);
end

