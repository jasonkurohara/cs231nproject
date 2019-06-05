% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
function opt = globals()

opt.root = pwd;

% path for MOT benchmark
mot_paths = {'MOT_data'};
for i = 1:numel(mot_paths)
    if exist(mot_paths{i}, 'dir')
        disp('yes');
        opt.mot = mot_paths{i};
        break;
    end
end

opt.mot2d = '2DMOT2015';
opt.results = 'results';

opt.mot2d_train_seqs = {'M0101', 'M0201', 'M0202', 'M0204', 'M0205', 'M0206', 'M0208', 'M0209', 'M0210', 'M0301', 'M0401', 'M0402', 'M0403',  'M0501', 'M0601', 'M0602', 'M0603', 'M0604', 'M0605', 'M0606', 'M0701'};
opt.mot2d_train_nums = [407, 1076, 291, 350, 646, 562, 265, 1576, 583, 325, 613, 410, 514, 352, 372, 480, 2035, 1079, 787, 1374, 1308];

opt.mot2d_test_seqs = {'M1303', 'M1304', 'M1305', 'M1306', 'M1401'};
opt.mot2d_test_nums = [445, 1550, 600, 1200, 1050];

addpath(fullfile(opt.mot, 'devkit', 'utils'));
addpath([opt.root '/3rd_party/libsvm-3.20/matlab']);
addpath([opt.root '/3rd_party/Hungarian']);

if exist(opt.results, 'dir') == 0
    mkdir(opt.results);
end

% tracking parameters
opt.num = 10;                 % number of templates in tracker (default 10)
opt.fb_factor = 30;           % normalization factor for forward-backward error in optical flow
opt.threshold_ratio = 0.6;    % aspect ratio threshold in target association
opt.threshold_dis = 3;        % distance threshold in target association, multiple of the width of target
opt.threshold_box = 0.8;      % bounding box overlap threshold in tracked state
opt.std_box = [30 60];        % [width height] of the stanford box in computing flow
opt.margin_box = [5, 2];      % [width height] of the margin in computing flow
opt.enlarge_box = [5, 3];     % enlarge the box before computing flow
opt.level_track = 1;          % LK level in association
opt.level =  1;               % LK level in association
opt.max_ratio = 0.9;          % min allowed ratio in LK
opt.min_vnorm = 0.2;          % min allowed velocity norm in LK
opt.overlap_box = 0.5;        % overlap with detection in LK
opt.patchsize = [24 12];      % patch size for target appearance
opt.weight_tracking = 1;      % weight for tracking box in tracked state
opt.weight_association = 1;   % weight for tracking box in lost state

% parameters for generating training data
opt.overlap_occ = 0.7;
opt.overlap_pos = 0.5;
opt.overlap_neg = 0.2;
opt.overlap_sup = 0.7;      % suppress target used in testing only

% training parameters
opt.max_iter = 10000;     % max iterations in total
opt.max_count = 10;       % max iterations per sequence
opt.max_pass = 2;

% parameters to transite to inactive
opt.max_occlusion = 50;
opt.exit_threshold = 0.95;
opt.tracked = 5;