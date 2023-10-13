classdef ExactPartition < PartitionStrategy 
      
     properties 
         training_samples;           % Indexes of the training samples 
         test_samples;               % Indexes of the test samples 
         N;                          % Size of the original dataset 
         partition_struct_full;      % cvpartition object to compute the initial split 
         partition_struct_training;  % cvpartition object to compute the training split from the initial split 
     end 
      
     methods 
         function obj = ExactPartition(training_samples, test_samples) 
             % Create an ExactPartition object. The two parameters are the 
             % number of training and test samples to use in the split. 
             obj.training_samples = training_samples; 
             obj.test_samples = test_samples; 
             obj.num_folds = 1; 
         end 
          
        function obj = partition(obj, Y) 
             warning('off', 'stats:cvpartition:HOTrainingZero'); 
             warning('off', 'stats:cvpartition:HOTestZero'); 
              
             % Check that Y has enough samples 
             if(length(Y) < obj.training_samples + obj.test_samples) 
                 error('One of the datasets has not enough samples for the exact partitioning'); 
             end 
              
             % The initial partition is used to obtain (training_samples + 
             % test_samples) elements from the dataset. 
             if(obj.training_samples+obj.test_samples == length(Y)) 
                 obj.partition_struct_full = cvpartition(Y, 'Resubstitution'); 
             else 
             	obj.partition_struct_full = cvpartition(Y, 'holdout', obj.training_samples + obj.test_samples); 
             end 
              
             % From the initial partition, we generate the training 
             % partition. This ensures a correct proportion of samples in 
             % training and test. 
             Ytmp = Y(obj.partition_struct_full.test); 
             if(length(unique(Ytmp)) == 2 && obj.training_samples == 2) 
                     % Horrible hack for binary classification with two 
                     % samples in the training set. Without this, sometimes 
                     % we choose two patterns from the same class. 
                     class1_idx = find(Y == -1); 
                     class2_idx = find(Y == 1); 
                     class1_pattern_idx = randi(length(class1_idx)); 
                     class2_pattern_idx = randi(length(class2_idx)); 
                     obj.partition_struct_training.training = false(length(Ytmp), 1); 
                     obj.partition_struct_training.training(class1_pattern_idx) = true; 
                     obj.partition_struct_training.training(class2_pattern_idx) = true; 
                     obj.partition_struct_training.test = ~obj.partition_struct_training.training; 
             else 
                 obj.partition_struct_training = cvpartition(Ytmp, 'holdout', obj.test_samples); 
             end 
              
             warning('on', 'stats:cvpartition:HOTrainingZero'); 
             warning('on', 'stats:cvpartition:HOTestZero'); 
              
             obj = obj.setCurrentFold(1); 
             obj.N = length(Y); 
         end 
          
         function [logTrainIndexes, numTrainIndexes] = getTrainIndexes(obj) 
             tmp = find(obj.partition_struct_full.test == 1); 
             logTrainIndexes = obj.partition_struct_full.test; 
             logTrainIndexes(tmp(obj.partition_struct_training.test)) = 0;
             numTrainIndexes = find(logTrainIndexes);
         end 
          
         function [logTestIndexes, numTestIndexes] = getTestIndexes(obj) 
             tmp = find(obj.partition_struct_full.test == 1); 
             logTestIndexes = obj.partition_struct_full.test; 
             logTestIndexes(tmp(obj.partition_struct_training.training)) = 0;
             numTestIndexes = find(logTestIndexes);
         end 
         
         function s = getDescription(obj) 
             s = sprintf('Training on %i samples and testing on %i samples\n', sum(obj.getTrainingIndexes()), sum(obj.getTestIndexes())); 
         end 
     end 
      
 end 
