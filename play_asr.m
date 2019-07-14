close all
clear
clc

graphics_toolkit qt;

%pkg load parallel;

tic;

%% Corpus

% Filenames of recordings
samples = {'FüNF_14.wav' 'FüNF_15.wav' 'NULL_32.wav' 'NULL_37.wav' 'NULL_9.wav'};
% ALTERNATIVE: Use all *.wav files in directory
%files = dir('*.wav');
%samples = {files.name};

%% Labels

% Strip suffixes from labels
labels = regexprep(samples,'_.*$','');

% Generate labels and add start/stop and silence before and after words
for i=1:length(labels)
  labels_tmp = strsplit(labels{i},'-');
  labels{i} = {'sil' labels_tmp{:} 'sil'};
end

%% Models and states

% Make a list of required models
model_names = sort(unique([labels{:}]));
printf('labels generated: %.1fms\n',toc*1000);

% Function to find model by name
function ids = names2ids(names, model_names)
  ids = zeros(size(names));
  for i=1:length(names)
    ids(i) = find(strcmp(model_names,names{i}),1,'first');
  end
end

% Count needed states
states_per_word = 6;
states_other = 3;
model_states = cell(size(model_names));
num_states = 0;
for i=1:length(model_names)
  switch model_names{i}
    case 'sil'
      model_states{i} = num_states+1:num_states+states_other;
      num_states += states_other;
    otherwise
      model_states{i} = num_states+1:num_states+states_per_word;
      num_states += states_per_word;
  end
end
start_states = cellfun(@(x) x(1), model_states);
end_states = cellfun(@(x) x(end), model_states);

%% Load and resample audiodata
audiodata = cell(size(samples));
fs = 48000;
for i=1:length(samples)
  [signal_tmp, fs_tmp] = audioread(samples{i});
  if fs_tmp ~= fs
    signal = resample(signal_tmp, fs, fs_tmp);
  end
  audiodata{i} = signal_tmp;
end
printf('loaded audio data: %.1fms\n',toc*1000);

%% Feature extraction
features = cell(size(samples));
for i=1:length(audiodata)
  features{i} = feature_extraction(audiodata{i},fs);
end
printf('feature extraction complete: %.1fms\n',toc*1000);

%% Model initialization

% Linear state assignment
data_states = cell(size(labels));
data_frames = cell(size(labels));
for i=1:length(labels)
  features_tmp = features{i};
  labels_tmp = labels{i};
  ids_tmp = names2ids(labels_tmp, model_names);
  states_tmp = [model_states{ids_tmp}];
  index_tmp = round(linspace(1,length(states_tmp),size(features_tmp,2))); 
  data_states{i} = states_tmp(index_tmp);
  data_frames{i} = features_tmp;
end
data_states_joint = [data_states{:}];
data_frames_joint = [data_frames{:}];
num_frames = size(data_frames,2);
printf('initial state assignment complete: %.1fms\n',toc*1000);

% Means and variances/sigmas
gmms = cell(3,num_states);
for i=1:num_states
  frames_tmp = data_frames_joint(:,data_states_joint==i);
  gmms{1,i} = 1; % Weight
  gmms{2,i} = mean(frames_tmp,2); % Mean
  gmms{3,i} = std(frames_tmp,[],2); % Standard deviation
end
printf('initialized GMMs: %.1fms\n',toc*1000);

printf('START TRAINING: %.1fms\n',toc*1000);

%% Viterbi-Training
figure('Position',[0 0 1600 800]);
iteration = 0;
while iteration < 10
  % Visualization of aligned sequences
  for i=1:length(labels)
    features_tmp = features{i};
    subplot(3,length(labels),i);
    imagesc(features_tmp); axis xy;
    subplot(3,length(labels),length(labels)+i);
    hold off;
    plot(data_states{i});
    hold on;
    for j=1:length(start_states)
      plot([1 size(features_tmp,2)],start_states(j).*[1 1],'r');
    end
    axis tight;
    subplot(3,length(labels),2.*length(labels)+i);
    imagesc([gmms{2,:}](:,data_states{i})); axis xy;
  end
  drawnow;
  
  iteration += 1;
  % Expectation step
  printf('start iteration (%i): %.1fms\n',iteration,toc*1000);
  for i=1:length(labels)
    % Sequence of models from labels
    ids_tmp = names2ids(labels{i}, model_names);
    % Sequence of states from labels (via sequence of models)
    states_tmp = [model_states{ids_tmp}];
    % Perform state alginment with viterbi decoder (cf. comments in function)
    data_states{i} = viterbi(states_tmp, gmms, features{i}, [], 1, [], inf);
  end
  %% ALTERNATIVE: parallel recognition with GNU/OCTAVE parallel package
  %data_states = parcellfun(nproc,@(x,y) viterbi([model_states{names2ids(y, model_names)}],gmms,x,[],1,[],inf),features,labels,'UniformOutput',0);
  printf('expectation (%i): %.1fms\n',iteration,toc*1000);

  % Collect new data (most likely state assignment for each data frame)
  data_states_joint = [data_states{:}];

  % Maximization step
  for i=1:num_states
    % Collect all frames assigned to state i
    frames_tmp = data_frames_joint(:,data_states_joint==i);
    if size(frames_tmp,2) > 5
      % Only update GMM parameters if there is some data assigned to the state
      [gmms{1,i}, gmms{2,i}, gmms{3,i}] = em_gmm(gmms{1,i}, gmms{2,i}, gmms{3,i}, frames_tmp, 1);
    else
      warning(sprintf('not suffient data to update state %i',i));
    end
  end
  printf('maximization complete (%i): %.1fms\n',iteration,toc*1000);
end

printf('TRAINING FINISHED: %.1fms\n',toc*1000);

%% Recognition

% Basic left-to-right transition matrix for transitions within the same model
transmat_base = -inf(num_states,num_states);
for i=1:length(model_names)
  states_tmp = model_states{i};
  for j=1:length(states_tmp)
    % Remain transition
    transmat_base(states_tmp(j),states_tmp(j)) = 0;
    % Leave transition
    if j < length(states_tmp)
      transmat_base(states_tmp(j),states_tmp(j+1)) = 0;
    end
  end
end

% Add transition between words (bag of words model)
transmat_bagofwords = transmat_base;
for i=1:length(end_states)
  for j=1:length(start_states)
    % From each end state to each start state
    transmat_bagofwords(end_states(i),start_states(j)) = 0;
  end
end

printf('decoding graph generated: %.1fms\n',toc*1000);

% Use all states for recognition
states_tmp = [model_states{:}];
% Use bag of words transition model for recognition
transmat = transmat_bagofwords;

printf('START RECOGNITION: %.1fms\n',toc*1000);

% Join all feature vectors to "test" the recognizer
features{end+1} = [features{:}];

viterbi_sequences = cell(1,length(features));
for i=1:length(features)
  viterbi_sequences{i} = viterbi(states_tmp, gmms, features{i}, transmat, start_states, nan, inf);
end
%% ALTERNATIVE: parallel recognition with GNU/OCTAVE parallel package
%viterbi_sequences = parcellfun(nproc,@(x) viterbi(states_tmp, gmms, x, transmat, start_states, nan, inf),features,'UniformOutput',0);

% Find first occurences of transitions into new models and generate transcription
transcriptions = cell(size(viterbi_sequences));
for i=1:length(viterbi_sequences)
  state_sequence_tmp = viterbi_sequences{i};
  transcription_tmp = {};
  for j=1:length(state_sequence_tmp)
    if any(state_sequence_tmp(j) == start_states) && (j == 1 || state_sequence_tmp(j) ~= state_sequence_tmp(j-1))
       transcription_tmp{end+1} = model_names{state_sequence_tmp(j) == start_states};
    end
  end
  transcriptions{i} = transcription_tmp;
end

printf('recognition complete (%i): %.1fms\n',iteration,toc*1000);

% Visualization of recognized sequences
figure('Position',[0 0 1600 800]);
for i=1:length(features)
  features_tmp = features{i};
  subplot(3,length(features),i);
  imagesc(features_tmp); axis xy;
  subplot(3,length(features),length(features)+i);
  hold off;
  plot(viterbi_sequences{i});
  hold on;
  for j=1:length(start_states)
    plot([1 size(features_tmp,2)],start_states(j).*[1 1],'r');
  end
  axis tight;
  subplot(3,length(features),2.*length(features)+i);
  imagesc([gmms{2,:}](:,viterbi_sequences{i})); axis xy;
end
drawnow;
