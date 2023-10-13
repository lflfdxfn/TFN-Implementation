classdef KFoldPartition < PartitionStrategy
    
    properties
        K;
        partition_struct;
    end
    
    methods
        function obj = KFoldPartition(K)
            obj.K = K;
            obj.num_folds = K;
        end
        
        function obj = partition(obj, Y)
            warning('off','stats:cvpartition:KFoldMissingGrp');
            obj.partition_struct = cvpartition(Y,'kfold',obj.K);
            warning('on','stats:cvpartition:KFoldMissingGrp');
            obj = obj.setCurrentFold(1);
        end
        
        function [logTrainIdx, numTrainIdx] = getTrainIndexes(obj)
            logTrainIdx = obj.partition_struct.training(obj.current_fold);
            numTrainIdx = find(logTrainIdx);
        end
        
        function [logTestIdx, numTestIdx] = getTestIndexes(obj)
            logTestIdx = obj.partition_struct.test(obj.current_fold);
            numTestIdx = find(logTestIdx);
        end
        
        function d = getDescription(obj)
            d = sprintf('%i-fold partition',obj.K);
        end
    end
    
end

