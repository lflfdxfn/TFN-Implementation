classdef Dataset
    
    properties
        name;
        X;
        Y;
        task;
    end
    
    properties
        currentPartition;
        partitions;
        ss_partitions;
        ss_strategy;
        shuffles;
        N_nodes;
        distrPartition;
    end
    
    methods
        function obj = Dataset(name, X, task, Y)
            obj.name = name;
            obj.X = X;
            obj.task = task;
            if nargin == 4
                obj.Y = Y;
            end
        end
        
        function obj = shuffle(obj, N)
            obj.shuffles = cell(N,1);
            for ii = 1:N
                obj.shuffles{ii} = randperm(size(obj.Y,1));
            end
        end
        
        function obj = normalize(obj, l_range, u_range, flag)
            obj.X = mapminmax(obj.X',l_range, u_range);
            obj.X = obj.X';
            if (nargin == 4)
                d = sum(obj.X');
                d = repmat(d',1,size(obj.X,2));
                obj.X = obj.X./d;
            end
        end
        
        function obj = generateNPartitions(obj, N, partitionStrategy, ss_partitionStrategy)
            % Generate N partitions of the dataset using a given
            % partitioning strategy. In semi-supervised mode, two
            % strategies must be provided.
            
            if(nargin < 4)
                obj.ss_strategy = [];
            else
                obj.ss_strategy = ss_partitionStrategy;
            end
            
            obj.partitions = cell(N, 1);
            obj.currentPartition = 1;
            obj.ss_partitions = cell(N, 1);
            
            for ii=1:N
                if(~isempty(obj.shuffles))
                    % Shuffle the dataset
                    currentY = obj.Y(obj.shuffles{ii});
                else
                    currentY = obj.Y;
                end
                
                if(~isempty(obj.ss_strategy))
                    obj.partitions{ii} = partitionStrategy.partition(currentY);
                    obj.ss_partitions{ii} =obj.ss_strategy.partition(currentY(obj.partitions{ii}.getTrainIndexes, :));
                else
                    obj.partitions{ii} = partitionStrategy.partition(currentY);
                end
            end
            
        end
        
        function obj = generateSinglePartition(obj, partitionStrategy, ss_partitionStrategy)
            % Commodity method for generating a single partition
           if(nargin == 3)
               obj = obj.generateNPartitions(1, partitionStrategy, ss_partitionStrategy);
           else
               obj = obj.generateNPartitions(1, partitionStrategy);
           end
        end
        
        function f = folds(obj)
            % Get the number of folds
            f = obj.partitions{1}.getNumFolds();
        end
        
        function obj = setCurrentPartition(obj,ind)
            obj.currentPartition = ind;
        end
        
        function [trainData, testData, unlabeledData] = getFold(obj, ii)
            if nargin == 1
                obj = obj.setCurrentPartition(ii);
            end
            if (~isempty(obj.shuffles))
                X = obj.X(obj.shuffles{ii},:);
                Y = obj.Y(obj.shuffles{ii},:);
            else
                X = obj.X;
                Y = obj.Y;
            end
            
            partitionStrategy = obj.partitions{obj.currentPartition};
            partitionStrategy = partitionStrategy.setCurrentFold(ii);
            
            trainInd = partitionStrategy.getTrainIndexes();
            testInd = partitionStrategy.getTestIndexes();
            
            trainData = Dataset(obj.name, X(trainInd,:), obj.task, Y(trainInd,:));
            testData = Dataset(obj.name, X(testInd,:), obj.task, Y(testInd,:));
            
            if ~isempty(obj.ss_partitions)
                ssPartitionStrategy = obj.ss_partitions{obj.currentPartition};
                ssPartitionStrategy = ssPartitionStrategy.setCurrentFold(ii);
                
                trainLabeledInd = ssPartitionStrategy.getTrainIndexes();
                trainUnlabeledInd = ssPartitionStrategy.getTestIndexes();
                
                unlabeledData = Dataset(obj.name, trainData.X(trainUnlabeledInd, :), obj.task, trainData.Y(trainUnlabeledInd, :));
                trainData = Dataset(obj.name, trainData.X(trainLabeledInd, :), obj.task, trainData.Y(trainLabeledInd, :));
                
            else
                unlabeledData = [];
            end
        end
        
        function obj = distributeDataset(obj, PartitionStr)
            obj.N_nodes = PartitionStr.getNumFolds;
            for ii = 1:obj.N_nodes
                obj.distrPartition = PartitionStr.partition(obj.Y);
            end
        end
        
        function localDataset = getLocalPart(obj, ii)
            partitionStrategy = obj.distrPartition;
            partitionStrategy = partitionStrategy.setCurrentFold(ii);
            X = obj.X;
            Y = obj.Y;
            
            localInd = partitionStrategy.getTestIndexes();
            localDataset = Dataset(obj.name, X(localInd, :), obj.task, Y(localInd, :));
        end
        
        function obj = translateUnlabeled(obj)
            obj.X = obj.X - repmat(mean(obj.X), size(obj.X,1), 1);
        end
    end
    
end

