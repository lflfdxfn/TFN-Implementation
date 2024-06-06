%% Initialization
clc;clear;f=genpath(pwd);addpath(f);
warning('off','all');echo anfis off;

%% Parameters
%1.Experimental settings
EXP=struct('seed',42, 'runs',10);
%2.Datasets
datasets = ["campaign","celeba","donors","fraud","thyroid",...
    "DoS","reconnaissance","fuzzers","backdoor","exploits"];
DATA=struct('datasets', datasets, 'start',1, 'end', length(datasets));
%3.WSAD scenario settings
WSAD=struct('known_outlier',60, 'contamination',0.02);
%4.Pretrain settings
PCA=struct('pca',1, 'min_dim',100, 'threshold',0.99);
%5."search": search for number of rules, or "fixed": just use the fixed numbers in fix_rule
fix_rule=[];
PTRAIN=struct('method','search', 'min_rule',2, 'max_rule', 10, 'fix_rule',fix_rule);
%6.Data Augmentation module settings
AUG=struct('num_train',10240, 'c_values',[8,4,0], 'E_test', 30);
%7.Consistency Regularization Term settings, "No", "ICR"
MIXUP=struct('type','ICR', 'M', 20000, 'gamma',10);
%8.L2 Regularization setting
REGU=struct('lambda',1);
%9.Cluster methods used: "p_fcm", or "k-means"
TRAIN=struct('cluster','p_fcm');

%% Preparation
loc='./Datasets/';
% Record the experiment
[log_file,result_file, index]=log_name();
diary(log_file);
diary on;
FPN_setting(EXP,DATA,WSAD,PCA,PTRAIN,AUG,MIXUP,REGU,TRAIN);

%% Experiment
result1_cell={};
result2_cell={};


for i=DATA.start:DATA.end
    [ROC,PR,TIME, best_k] = TFN_experiment(loc,i,EXP,DATA,WSAD,PCA,PTRAIN,AUG,MIXUP,REGU,TRAIN);
    
    roc_mean_var=sprintf('%f / %f',ROC.mean,ROC.var);
    pr_mean_var=sprintf('%f / %f',PR.mean,PR.var);
    
    result1_cell(:,i)={DATA.datasets{i},best_k,ROC.mean,ROC.var,roc_mean_var,TIME.ptrain,TIME.train,TIME.test};
    result2_cell(:,i)={DATA.datasets{i},best_k,PR.mean,PR.var,pr_mean_var,TIME.ptrain,TIME.train,TIME.test};
end

fid1=fopen(result_file,'w');
fprintf(fid1,'AUC-ROC\ndataset,n_rule,mean,var,mean/var,Pretrain,Train,Test\n');
fprintf(fid1,'%s,%d,%f,%f,%s,%f,%f,%f,%f\n',result1_cell{:,DATA.start:DATA.end});
fprintf(fid1,'AUC-PR\ndataset,n_rule,mean,var,mean/var,Pretrain,Train,Test\n');
fprintf(fid1,'%s,%d,%f,%f,%s,%f,%f,%f,%f\n',result2_cell{:,DATA.start:DATA.end});
fclose(fid1);

diary off;

%% Functions
function [ROC,PR,TIME, best_k] = TFN_experiment(loc,i,EXP,DATA,WSAD,PCA,PTRAIN,AUG,MIXUP,REGU,TRAIN)
    % dataset
    name=DATA.datasets{i};    
    fprintf('\nDataset %s is running..,\n',name);
    train_path=strcat(loc,name,'_weakly_train_',string(WSAD.contamination),'_',string(WSAD.known_outlier),'.mat');
    test_path=strcat(loc,name,'_weakly_test_',string(WSAD.contamination),'_',string(WSAD.known_outlier),'.mat');
    PATH=struct('trainPath',train_path, 'testPath',test_path);    
    PTRAIN.cur_dataset=i;
    
    [ROC, PR, TIME, best_k] = TFN_runs(PATH,EXP,PCA,PTRAIN,AUG,MIXUP,REGU,TRAIN);
    fprintf('n_rule: %d, %s in %d runs: %f / %f\n',best_k,name,EXP.runs,ROC.mean,ROC.var);
    fprintf('n_rule: %d, %s in %d runs: %f/ %f\n',best_k,name,EXP.runs,PR.mean,PR.var);
    fprintf('pre_train_time: %f, train_time: %f, test_time: %f\n',TIME.ptrain,TIME.train,TIME.test);        

end

function [ROC, PR, TIME, best_k] = TFN_runs(PATH,EXP,PCA,PTRAIN,AUG,MIXUP,REGU,TRAIN)  
    auc_list=zeros(EXP.runs,1);
    pr_list=zeros(EXP.runs,1);
    train_sumtime=0;
    test_sumtime=0;
    rng(EXP.seed,'twister');
    
    trainData=load(PATH.trainPath);
    trainData.X=double(trainData.X);
    trainData.Y=double(trainData.Y);
    testData=load(PATH.testPath);
    testData.X=double(testData.X);
    testData.Y=double(testData.Y);
    %PCA
    [~,dim]=size(trainData.X);
    if PCA.pca==1 && dim>=PCA.min_dim
        [trainData, testData] = pcaTrans(trainData,testData,PCA.threshold);
    end
    %Pre-train   
    tic;
    best_k=search_k(trainData,PTRAIN);
    n_rule=best_k;
    ptrain_time=toc;
    
    rng(EXP.seed,'twister');
    for j=1:EXP.runs
        [auc_roc,auc_pr,train_time, test_time] = TFN_train(n_rule, trainData, testData, TRAIN, REGU, MIXUP, AUG);

        auc_list(j)=auc_roc;
        pr_list(j)=auc_pr;
        train_sumtime=train_sumtime+train_time;
        test_sumtime=test_sumtime+test_time;
    end

    ROC=struct('mean',mean(auc_list), 'var',std(auc_list));
    PR=struct('mean',mean(pr_list), 'var',std(pr_list));
    TIME=struct('ptrain',ptrain_time, 'train',train_sumtime/EXP.runs, 'test', test_sumtime/EXP.runs);
end

function [logfile,resultfile,i]=log_name()
    for i=1:100
        logfile=sprintf('Logs/log%d.txt',i);
        resultfile=sprintf('Logs/log_Result%d.csv',i);
        if ~exist(logfile,'file') && ~exist(resultfile,'file')
            break
        end
    end
end

function [] = FPN_setting(EXP,DATA,WSAD,PCA,PTRAIN,AUG,MIXUP,REGU,TRAIN)
    fprintf('SETTINGS\n')
    % Dataset info
    fprintf('\tDATASETS:\n\t\t')
    fprintf(repmat(' %s',1,DATA.end-DATA.start+1),DATA.datasets(DATA.start:DATA.end))
    fprintf('\n')
    % Rule mode info
    fprintf('\tPCA to reducing dims: %d\n',PCA.pca)
    if PCA.pca==1
        fprintf('\t\tDim threshold: %d\n',PCA.min_dim)
        fprintf('\t\tPCA threshold: %f\n',PCA.threshold)
    end
    fprintf('\tRUNS: %d\n',EXP.runs)
    % Pretrain method
    fprintf('\tPRETRAIN: %d~%d using %s\n',PTRAIN.min_rule,PTRAIN.max_rule,PTRAIN.method)
    if strcmp(PTRAIN.method,'fixed')
        fprintf('\tALPHA of FCM: %f with fixed n_rules:\n\t\t',2)
        fprintf(' %d ',PTRAIN.fix_rule)
        fprintf('\n')
    else
        fprintf('\tALPHA of FCM: %f\n', 2)
    end
    % Cluster mode info
    fprintf('\tCLUSTER: %s\n',TRAIN.cluster)
    % REGULARIZATION
    fprintf('\tREGULARIZATION: \n')
    fprintf('\t\tlambda: %d\n',REGU.lambda)
    fprintf('\t\tmixup_term: %s\n',MIXUP.type)
    % SCENRIO
    fprintf('\tSCENRIO: weakly-supervised\n')
    fprintf('\t\tnum of labeled: %d\n',WSAD.known_outlier)
    fprintf('\t\trate of contamination in unlabeled: %f\n',WSAD.contamination)
    % AUGMENTATION mode info
    fprintf('\tAUGMENTATION: \n')
    fprintf('\t\tnum of sampled in training: %d\n',AUG.num_train)
    fprintf('\t\tnum of sampled in testing:  %d\n',AUG.E_test)
    fprintf('\t\tassigned target C: %d, %d, %d\n',AUG.c_values)
    fprintf('！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！\n')
end

function [best_k]=search_k(train_data,PTRAIN)
    best_k = -1;

    if strcmp(PTRAIN.method,'fixed')
        best_k=PTRAIN.fix_rule(PTRAIN.cur_dataset);
    elseif strcmp(PTRAIN.method,'search')
        for n_rule=PTRAIN.min_rule:PTRAIN.max_rule
            [~,U]=fcm(train_data.X,n_rule, [NaN, NaN, NaN, 0]);
            
            best_k=n_rule;
            if prod(max(U,[],2)>0.5)==0
                best_k=max(2,best_k-1);
                break
            end
        end
    end
end