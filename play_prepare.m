close all
clear
clc

datadirs = {'train' 'test'};
fs = 48000;

for i=1:length(datadirs)
  datadir = datadirs{i};
  printf('\nprepare data from directory: %s\n', datadir);
  corpus_file = sprintf('corpus_%s.mat', datadir);
  
  % Find data files
  files = dir([datadir filesep '*.wav']);
  samples = {files.name};

  %% Load and resample audiodata
  tic;
  audiodata = cell(size(samples));
  for i=1:length(samples)
    [signal_tmp, fs_tmp] = audioread([datadir filesep samples{i}]);
    signal_tmp = sum(signal_tmp,2);
    if fs_tmp ~= fs
      signal_tmp = resample(signal_tmp, fs, fs_tmp);
    end
    signal_tmp = signal_tmp - mean(signal_tmp);
    signal_tmp = signal_tmp./rms(signal_tmp);
    % Add random noise variants to the signal (to possibly improve generalization)
    %signal = 10.^(-rand(1)*15/20).*filter(0.1,[1 -(0.99-0.04*rand(1))], randn(size(signal_tmp)+[fs/2,0]));
    %signal(1+fs/4:size(signal_tmp,1)+fs/4) += signal_tmp;
    audiodata{i} = signal_tmp;
  end
  printf('loaded audio data from %i files in %.1fms\n', numel(files), toc*1000); 
  
  %% Feature extraction
  tic;
  features = cell(size(samples));
  for i=1:length(audiodata)
      features_tmp = feature_extraction(audiodata{i},fs);
      % Mean normalization
      features_tmp -= mean(features_tmp,2);
      %% Variance normalization (only use with MFCC or SGBFB)
      %features_tmp ./= rms(features_tmp,2);
      features{i} = features_tmp;
  end
  printf('performed feature extraction in %.1fms\n', toc*1000);
  
  tic;
  save(corpus_file, '-binary', 'samples', 'audiodata', 'features');
  printf('saved corpus data to %s in %.1fms\n', corpus_file, toc*1000);
end
