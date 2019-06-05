% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% test on MOT benchmark
function MOT_test

is_train = 0;
opt = globals();

mot2d_train_seqs = {'M0101', 'M0201', 'M0202', 'M0204', 'M0205', 'M0206', 'M0208', 'M0209', 'M0210', 'M0301', 'M0401', 'M0402', 'M0403',  'M0501', 'M0601', 'M0602', 'M0603', 'M0604', 'M0605', 'M0606', 'M0701'};

mot2d_test_seqs = {'M1303', 'M1304', 'M1305', 'M1306', 'M1401'};

% training and testing pairs
%seq_idx_train = {{1, 2}, {3},    {4, 5, 6}, {7, 8}, {9, 10}, {11}};
%seq_idx_test  = {{1},    {2, 6}, {3, 4, 5}, {7, 8}, {9, 10}, {11}};
%seq_idx_train = {{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21}};
seq_idx_train = {{1,3,7,11,17}};
seq_idx_test = {{1}};
seq_set_test = 'test';
N = numel(seq_idx_train);

test_time = 0;
for i = 1:N
    % training
    idx_train = seq_idx_train{i};
    
    if is_train
        % number of training sequences
        num = numel(idx_train);
        tracker = [];
        for j = 1:num
            fprintf('Training on sequence: %s\n', mot2d_train_seqs{idx_train{j}});
            tracker = MDP_train(idx_train{j}, tracker);
            fprintf('%d training examples after training on %s\n', ...
                size(tracker.f_occluded, 1), mot2d_train_seqs{idx_train{j}});
        end
    else
        % load tracker from file
        filename = sprintf('%s/M0401_tracker.mat', opt.results);
        object = load(filename);
        tracker = object.tracker;
        fprintf('load tracker from file %s\n', filename);
    end    
    
    % testing
    idx_test = seq_idx_test{i};
    % number of testing sequences
    num = numel(idx_test);
    for j = 1:num
        fprintf('Testing on sequence: %s\n', mot2d_test_seqs{idx_test{j}});
        tic;
        MDP_test(idx_test{j}, seq_set_test, tracker);
        test_time = test_time + toc;
    end
end

fprintf('Total time for testing: %f\n', test_time);