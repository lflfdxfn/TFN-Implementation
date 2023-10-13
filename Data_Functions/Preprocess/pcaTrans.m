function [transTrainData, transTestData,idx] = pcaTrans(trainData,testData,threshold)
%PCATRANS 此处显示有关此函数的摘要
%   此处显示详细说明
    XTrain=trainData.X;
    YTrain=trainData.Y;
    XTest=testData.X;
    YTest=testData.Y;
    
    [coeff,scoreTrain,~,~,explained,mu] = pca(XTrain);    
    sum_explained = 0;
    idx = 0;
    while sum_explained < 100*threshold
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    transXTrain = scoreTrain(:,1:idx);
    transXTest = (XTest-mu)*coeff(:,1:idx);
    
    transTrainData=Dataset(strcat(trainData.name,'_pca'),transXTrain,'R',YTrain);
    transTestData=Dataset(strcat(testData.name,'_pca'),transXTest,'R',YTest);
end

