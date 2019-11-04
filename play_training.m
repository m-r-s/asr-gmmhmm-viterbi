close all
clear
clc

graphics_toolkit qt;

pkg load parallel;

% Training parameters
corpus_file = 'corpus_train.mat';
model_file = 'model.mat';
states_per_word = 3;
states_other = 3;
add_components = [0];
iterations = [8];
updates = [1];
beamwidth = inf;
min_samples = 10;

% Showcase visualization
show_showcases = 1;

tic;
load(corpus_file);
printf('loaded corpus data from %s in %.1fms\n', corpus_file, toc*1000);

%% Model initialization

% Generate labels and add start/stop and silence before and after words
tic;
labels = regexprep(samples,'_.*$','');
for i=1:length(labels)
  labels_tmp = strsplit(labels{i},'-');
  labels{i} = {'sil' labels_tmp{:} 'sil'};
end

% Make a list of required models
model_names = sort(unique([labels{:}]));

% Function to find model by name
function ids = names2ids(names, model_names)
  ids = zeros(size(names));
  for i=1:length(names)
    ids(i) = find(strcmp(model_names,names{i}),1,'first');
  end
end

% Count needed states
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
samples_idx = cell(size(features));
for i=1:length(features)
  samples_idx{i} = i.*ones(1,size(features{i},2));
end
data_states_joint = [data_states{:}];
data_frames_joint = [data_frames{:}];
samples_idx_joint = [samples_idx{:}];
num_frames_joint = size(data_frames_joint,2);

% Means and variances/sigmas
gmms = cell(3,num_states);
for i=1:num_states
  frames_tmp = data_frames_joint(:,data_states_joint==i);
  gmms{1,i} = 1; % Weight
  gmms{2,i} = mean(frames_tmp,2); % Mean
  gmms{3,i} = std(frames_tmp,[],2); % Standard deviation
end
printf('initialized model in %.1fms\n',toc*1000);

if show_showcases
  figure('Position',[0 0 1600 800]);
end

printf('\nSTART TRAINING\n');
%% Viterbi-Training
for i=1:length(add_components)
  % Split the heaviest component in the dimension with the most variance if requested
  for j=1:num_states
    for k=1:add_components(i)
      [weights_tmp, means_tmp, sigmas_tmp] = gmms{:,j};
      [~, comp] = max(weights_tmp);
      [~, dim] = max(sigmas_tmp(:,comp));
      weights_tmp(end+1) = 0.5.*weights_tmp(comp);
      weights_tmp(comp) = 0.5.*weights_tmp(comp);
      means_tmp(:,end+1) = means_tmp(:,comp);
      means_tmp(dim,comp) = means_tmp(dim,comp) - 0.5.*sigmas_tmp(dim,comp);
      means_tmp(dim,end) = means_tmp(dim,end) + 0.5.*sigmas_tmp(dim,comp);
      sigmas_tmp(:,end+1) = sigmas_tmp(:,comp);
      sigmas_tmp(dim,comp) = 0.5.*sigmas_tmp(dim,comp);
      sigmas_tmp(dim,end) = 0.5.*sigmas_tmp(dim,end);
      gmms{1,j} = weights_tmp;
      gmms{2,j} = means_tmp;
      gmms{3,j} = sigmas_tmp;
    end
  end
  
  for j=1:iterations(i)  
    % Expectation step
    printf('iteration %i\n',j);
    tic;
    
    for k=1:length(labels)
      % Sequence of models from labels
      ids_tmp = names2ids(labels{k}, model_names);
      % Sequence of states from labels (via sequence of models)
      states_tmp = [model_states{ids_tmp}];
      % Perform state alginment with viterbi decoder (cf. comments in function)
      data_states{k} = viterbi(states_tmp, gmms, features{k}, [], 1, [], beamwidth);
    end
    % ALTERNATIVE: parallel recognition with GNU/OCTAVE parallel package
    % data_states = parcellfun(nproc,@(x,y) viterbi([model_states{names2ids(y, model_names)}],gmms,x,[],1,[],beamwidth),features,labels,'UniformOutput',0);
    printf('expectation step completed in %.1fms\n',toc*1000);
    % Collect new data (most likely state assignment for each data frame)
    data_states_joint = [data_states{:}];
    
    tic;
    % Maximization step
    for k=1:num_states
      % Collect all frames assigned to state i
      frames_tmp = data_frames_joint(:,data_states_joint==k);
      if size(frames_tmp,2) >= min_samples
        % Only update GMM parameters if there is some data assigned to the state
        [gmms{1,k}, gmms{2,k}, gmms{3,k}] = em_gmm(gmms{1,k}, gmms{2,k}, gmms{3,k}, frames_tmp, updates(i));
      else
        warning(sprintf('not suffient data to update state %i',i));
      end
    end
    printf('maximization step completed in %.1fms\n',toc*1000);
    
    % Calculate likelihood
    log_likelihood = -inf(1,num_frames_joint);
    for k=1:num_frames_joint
      [weights_tmp, means_tmp, sigmas_tmp] = gmms{:,data_states_joint(k)};
      log_likelihood(k) = sum(log(sum(normpdf(data_frames_joint(:,k),means_tmp,sigmas_tmp).*weights_tmp,2)));
    end
    printf('log likelihood statistis after maximization\n\n');
    for k=1:length(model_names)
      printf('MODEL: %s\n',model_names{k});
      word_numframes_tmp = 0;
      word_likelihood_tmp = 0;
      for l=1:length(model_states{k})
        log_likelihood_tmp = log_likelihood(data_states_joint==model_states{k}(l));
        prctiles_tmp = prctile(log_likelihood_tmp,[0 5 50 95 100]);
        printf('state % 3i (% 6i) % 5.0f   % 5.0f % 5.0f % 5.0f % 5.0f % 5.0f\n',l,numel(log_likelihood_tmp),mean(log_likelihood_tmp),prctiles_tmp(1),prctiles_tmp(2),prctiles_tmp(3),prctiles_tmp(4),prctiles_tmp(5));
        word_numframes_tmp = word_numframes_tmp + numel(log_likelihood_tmp);
        word_likelihood_tmp = word_likelihood_tmp + sum(log_likelihood_tmp);
      end
      printf('average likelihood per frame in word %s: %.3f\n\n',model_names{k},word_likelihood_tmp/word_numframes_tmp);
    end
    % Find worst/best/average fitted example
    [~,sortidx] = sort(log_likelihood);
    printf('new average log likelihood per frame: %.3f (%.3f/%.3f/%.3f best/median/worst)\n', ...
      mean(log_likelihood), ...
      log_likelihood(sortidx(end)), ...
      log_likelihood(sortidx(round(end/2))), ...
      log_likelihood(sortidx(1)));

    if show_showcases    
      showcase = samples_idx_joint(sortidx([end round(end/2) 1]));
      titles = {'best frame','median frame','worst frame'};
      % Visualization of showcase aligned sequences
      for k=1:length(showcase)
        fileidx = showcase(k);
        logms_tmp = log_mel_spectrogram(audiodata{fileidx},48000);
        data_frames_tmp = data_frames{fileidx};
        data_states_tmp = data_states{fileidx};
        subplot(3,length(showcase),k);
        imagesc(logms_tmp); axis xy;
        title(titles{k});
        subplot(3,length(showcase),length(showcase)+k);
        hold off;
        plot(data_states{fileidx});
        hold on;
        for j=1:length(start_states)
          plot([1 size(data_frames_tmp,2)],start_states(j).*[1 1],'r');
        end
        axis tight;
        ylim([1 num_states]);
        subplot(3,length(showcase),2.*length(showcase)+k);
        log_likelihood_tmp = -inf(size(data_frames_tmp));
        for l=1:length(data_states_tmp)
          [weights_tmp, means_tmp, sigmas_tmp] = gmms{:,data_states_tmp(l)};
          log_likelihood_tmp(:,l) = log(sum(normpdf(data_frames_tmp(:,l),means_tmp,sigmas_tmp).*weights_tmp,2));
        end
        plot(sum(log_likelihood_tmp));
        axis tight; 
      end
      drawnow;  
    end
  end
end
printf('\nTRAINING FINISHED\n');

tic;
save(model_file,'-binary','model_names','model_states','gmms');
printf('saved model to %s in %.1fms\n', model_file, toc*1000);
