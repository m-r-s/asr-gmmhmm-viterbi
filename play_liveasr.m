close all
clear
clc

graphics_toolkit qt;

% Real-time decoding parameters
model_file = 'model.mat';
beamwidth = inf; % Initial beamwidth in log likelihood
rtf_target = 0.8; % Target real time factor
beamwidth_range = [10 10000]; % Adaptive beamwidth range
insertion_penalty = 0; % Word insertion penalty in log likelihood

% Playrec parameters
fs = 48000;
addpath('~/playrec/playrec');
pagesize = 480;
pagebufferlength = 10;

% For simplicity of the real-time implementation of the feature extractiom
% we learn a matrix to extract feature linearly from the last [max_context]
% log mel spectrogram frames.
tic;
signal = randn(fs*60,1);
[b, a] = feature_extraction(signal,fs);
max_context = 10; % 10 for MFCCs and 40 for SGBFBs
c = zeros(max_context.*size(a,1),size(a,2));
for i=0:max_context-1
  c(1+i*size(a,1):(i+1)*size(a,1),:) = circshift(a,+max_context/2-i,2);
end
c = c(:,21:end-20);
b = b(:,21:end-20);
% Train linear neural network ;)
feature_extraction_matrix = b/c;
printf('learnt linear feature extraction model %s in %.1fms\n', model_file, toc*1000);

tic;
load(model_file);
printf('loaded model from %s in %.1fms\n', model_file, toc*1000);

% Use all states for recognition
states = [model_states{:}];
num_states = numel(unique(states));
start_states = cellfun(@(x) x(1), model_states);
end_states = cellfun(@(x) x(end), model_states);
model_state_names = cell(0);
for i=1:length(model_states)
  for j=1:length(model_states{i})
    model_state_names{end+1} = sprintf('%s%i',model_names{i},j);
  end
end

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

% input buffer of 25 ms
signal_frame = zeros(1200,1);

% Functions and values needed for real time calculation of log Mel spectrogram
function f = mel2hz (m)
  % Convert frequency from Mel to Hz
  f = 700.*((10.^(m./2595))-1);
end
function m = hz2mel (f)
  % Convert frequency from Hz to Mel
  m = 2595.*log10(1+f./700);
end
function [transmat, freq_centers_idx] = triafbmat(fs, num_coeff, freq_centers, width)
  width_left = width(1);
  width_right = width(2);
  freq_centers_idx = round(freq_centers./fs .* num_coeff);
  num_bands = length(freq_centers)-(width_left+width_right);
  transmat = zeros(num_bands, num_coeff);
  for i=1:num_bands
    left = freq_centers_idx(i);
    center = freq_centers_idx(i+width_left);
    right = freq_centers_idx(i+width_left+width_right);
    start_raise = 0;
    stop_raise = 1;
    start_fall = 1;
    stop_fall = 0;
    if (left >= 1)
      transmat(i,left:center) = linspace(start_raise, stop_raise, center-left+1);
    end
    if (right <= num_coeff)
      transmat(i,center:right) = linspace(start_fall, stop_fall, right-center+1);
    end
  end
end
band_factor = 1;
window_function = hamming(1200);
window_function = window_function ./ sqrt(mean(window_function.^2));
freq_range = [64 12000];
channel_dist = (hz2mel(4000) - hz2mel(64))./(23+1); % Distance between center frequencies in Mel
num_bands = floor((hz2mel(freq_range(2)) - hz2mel(freq_range(1)))./channel_dist)-1;
freq_range(2) = mel2hz(hz2mel(freq_range(1))+channel_dist.*(num_bands+1));
freq_centers = mel2hz(linspace(hz2mel(freq_range(1)), hz2mel(freq_range(2)), (num_bands+1).*band_factor+1));
mel_matrix = triafbmat(fs, 2048, freq_centers, [1 1].*band_factor);

% variables fo running mean and variance normalization
features_mean = zeros(size(feature_extraction_matrix,1),1);
features_var = ones(size(feature_extraction_matrix,1),1);

% Allocate memory for Trellis diagram of Viterbi decoder
num_frames = 100;
trellis_likelihood = zeros(num_states,num_frames);
trellis_state = ones(size(trellis_likelihood));

% Allocate memory for log ms history
log_mel_spec_hist = ones(num_bands,num_frames);

% Prepare figure
figure('Position',[0 0 800 400]);
subplot(3,1,1);
h0 = imagesc(log_mel_spec_hist);
subplot(3,1,2);
h1 = imagesc(trellis_likelihood);
hold on;
h3 = plot(ones(num_frames,1),'r','linewidth',2);
subplot(3,1,3);
h2 = imagesc(trellis_state,[1 num_states]);
hold on;
h4 = plot(ones(num_frames,1),'r','linewidth',2);

if playrec('isInitialised')
  playrec('reset')
  sleep(0.1);
end

if ~playrec('isInitialised')
  playrec('init', fs, 0, 0, 2, 2);
  pause(0.1);
end

in = zeros(pagesize,2);
out = zeros(pagesize,2);

assert(playrec('isInitialised'));

if playrec('pause')
  playrec('pause', 0);
end

playrec('delPage');

pagelist = zeros(pagebufferlength,1);
for i=1:pagebufferlength
  pagelist(i) = playrec('playrec',out,[1 2],pagesize,[1 2]);
end

printf('\nSTART RECOGNITION\n');

retrace = inf;
profile on;
pagebufferidx = 1;
tic;
while true;
  % Calculate real time factor
  rtf = toc/(pagesize/fs);
  
  % Adapt beamwidth to computational resources to achieve the
  % target real time factor
  if rtf > rtf_target
    beamwidth = max(beamwidth_range(1),beamwidth .* 0.99);
  elseif rtf < rtf_target
    beamwidth = min(beamwidth_range(2),beamwidth .* 1.01);
  end
  
  playrec('block', pagelist(pagebufferidx));
  tic;
  in = playrec('getRec', pagelist(pagebufferidx));
  playrec('delPage', pagelist(pagebufferidx));
  signal_frame = [signal_frame((1+size(in,1)):end);sum(in,2)];
  % Log Mel Spectrum of current frame
  log_mel_spec = 20.*log10(mel_matrix*abs(fft(signal_frame.*window_function,2048)));
  log_mel_spec_hist(:,end) = log_mel_spec;
  
  % Feature extraction by matrix multiplication
  features = feature_extraction_matrix * log_mel_spec_hist(1+end-size(feature_extraction_matrix,2):end).';
  
  % Mean(-and-variance) normalization
  update_coeff = 0.995;
  features_mean = update_coeff .* features_mean + (1-update_coeff) .* features;
  features = features - features_mean;
  % only use variance normalization with mfcc/sgbfb features
  %features_var = update_coeff .* features_var + (1-update_coeff) .* features.^2;
  %features = features./sqrt(features_var);
  
  % Update trellis diagram
  prior = trellis_likelihood(:,end-1) + transmat;
  max_prior = max(prior);
  % Use beamwidth to limit number of evaluations
  eval_id = find(max_prior>(max(max_prior)-beamwidth));
  acoustic_likelihood = -inf(1,num_states);
  for i=1:length(eval_id)
    % Evaluate GMM for k-th state in list
    stateid = states(eval_id(i));
    [weights, means, sigmas] = gmms{:,stateid};
    acoustic_likelihood (eval_id(i)) = ...
      sum(log(sum(normpdf(features ,means, sigmas).*weights,2)));
  end
  % Consider acoustic information
  likelihood = prior + acoustic_likelihood;
  % Only keep the most likely state transition (this is Viterbi)
  [trellis_likelihood(:,end), trellis_state(:,end)] = max(likelihood);
  
  % Re-normalize likelehoods after each frame
  % This is no problem because only the relative likelihoods are relevant
  trellis_likelihood(:,end) = trellis_likelihood(:,end) - max(trellis_likelihood(:,end));

  % Trigger re-trace every 10 updates (100ms)
  if retrace > 10
    % Trace back the most likely path
    viterbi_path = nan(num_frames,1);
    % Start with most likely state
    [~, final_state_tmp] = max(trellis_likelihood(:,end));
    viterbi_path(end) = final_state_tmp;
    % Iterate from last state to the first
    for i=num_frames:-1:2
      viterbi_path(i-1) = trellis_state(viterbi_path(i),i-1);
    end
    % Translate path into state sequence
    state_alignment = states(viterbi_path);
    
    for i=1:min(num_frames,retrace-1)
      printf('% 5s  ',model_state_names{state_alignment(i)});
    end
    printf('   (rtf=%.3f beam=%.3f)\n',rtf,beamwidth);
    
    retrace = 1;
    set(h0,'cdata',log_mel_spec_hist);
    set(h1,'cdata',trellis_likelihood);
    set(h2,'cdata',trellis_state);
    set(h3,'ydata',state_alignment);
    set(h4,'ydata',state_alignment);
    drawnow;
  else
    retrace = retrace + 1;
  end
  
  % Move decision window
  log_mel_spec_hist = [log_mel_spec_hist(:,2:end) ones(num_bands,1)];
  trellis_likelihood = [trellis_likelihood(:,2:end) zeros(num_states,1)];
  trellis_state = [trellis_state(:,2:end) ones(num_states,1)];
  
  % Invent output values for playrec
  out = zeros(size(in));
  pagelist(pagebufferidx) = playrec('playrec',out,[1 2],pagesize,[1 2]);
  pagebufferidx = pagebufferidx + 1;
  if pagebufferidx > pagebufferlength
    pagebufferidx = 1;
  end  
end