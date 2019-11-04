close all
clear
clc

pkg load parallel;

% Recognition parameters
corpus_file = 'corpus_test.mat';
model_file = 'model.mat';
results_file = 'results.mat';
beamwidth = 1000;
insertion_penalty = 0;

tic;
load(corpus_file);
printf('loaded corpus data from %s in %.1fms\n', corpus_file, toc*1000);

tic;
load(model_file);
printf('loaded model from %s in %.1fms\n', model_file, toc*1000);

%% Preparation
tic;

% Use all states for recognition
states = [model_states{:}];
num_states = numel(unique(states));
start_states = cellfun(@(x) x(1), model_states);
end_states = cellfun(@(x) x(end), model_states);

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

% Add transitions
transmat = transmat_base;

% Put sil between everything
sil_idx = find(ismember(model_names,'sil'));
transmat(end_states(sil_idx),start_states(sil_idx)) = 0;
words_idx = find(~ismember(model_names,'sil'));
for i=1:length(words_idx)
  % From sil to word and back
  transmat(end_states(words_idx(i)),start_states(sil_idx)) = 0;
  transmat(end_states(sil_idx),start_states(words_idx(i))) = -insertion_penalty;
end

printf('recognizer prepared in %.1fms\n',toc*1000);

printf('\nSTART RECOGNITION\n');
tic;
viterbi_sequences = cell(1,length(features));
for i=1:length(features)
  viterbi_sequences{i} = viterbi(states, gmms, features{i}, transmat, sil_idx, nan, beamwidth);
end
% ALTERNATIVE: parallel recognition with GNU/OCTAVE parallel package
% viterbi_sequences = parcellfun(nproc,@(x) viterbi(states, gmms, x, transmat, sil_idx, nan, beamwidth),features,'UniformOutput',0);
printf('recognition completed in %.1fms\n',toc*1000);

tic;
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

save(results_file,'-binary','samples','viterbi_sequences','transcriptions');
printf('saved results to %s in %.1fms\n', results_file, toc*1000);
