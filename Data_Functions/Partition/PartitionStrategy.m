classdef PartitionStrategy
    
    properties
        current_fold;
        num_folds;
    end
    
    methods (Abstract)
        obj = partition(obj, Y);
        
        [logTrainIdx, numTrainIdx] = getTrainIndexes(obj);
        
        [logTestIdx, numTestIdx] = getTestIndexes(obj);
        
        d = getDescription(obj);
    end
    
    methods
        function n = getNumFolds(obj)
            n = obj.num_folds;
        end
        
        function obj = setCurrentFold(obj,ii)
            obj.current_fold = ii;
        end
    end
end

