classdef NoPartition < PartitionStrategy
    
    properties
        y_length;
    end
    
    methods
        function obj = NoPartition()
            obj.num_folds = 1;
        end
        
        function obj = partition(obj, Y)
            obj.y_length = size(Y,1);
            obj.setCurrentFold(1);
        end
        
        function [logTrainIdx, absTrainIdx] = getTrainIndexes(obj)
            logTrainIdx = true(obj.y_length,1);
            absTrainIdx = [1:obj.y_length]';
        end
        
        function [logTestIdx, absTestIdx] = getTestIndexes(obj)
            logTestIdx = true(obj.y_length,1);
            absTestIdx =  [1:obj.y_length]';
        end
        
        function d = getDescription(obj)
            d = sprintf('Both Training and Test sets include all data (%i samples)', obj.y_length);
        end
    end
    
end

