function state_sequence = viterbi(states, gmms, features, transmat, initial_state, final_state, beamwidth)
num_frames = size(features,2);
num_states = length(states);

% Allocate memory for Trellis diagram
trellis_likelihood = -inf(num_states,num_frames);
trellis_state = nan(size(trellis_likelihood));

% Assume left to right transitions if no transition matrix was provided
if nargin < 4 || isempty(transmat)
  transmat = eye(num_states) + diag(ones(num_states-1,1),1);
  transmat = log(transmat);
end

initial_state_likelihood = -inf(num_states,1);
% Assume start in first state
if nargin < 5 || isempty(initial_state)
  initial_state = 1;
end
initial_state_likelihood(initial_state) = 0;

% Assume end in most likely state
if nargin < 6 || isempty(final_state)
  final_state = num_states;
end

% By default infitite beamwidth
if nargin < 7 || isempty(beamwidth)
  beamwidth = inf;
end

% Iteratively calculate likelihoods from preceeding frames
for j=1:num_frames
  % Calculate prior likelihood
  if j==1
    prior = initial_state_likelihood + transmat;
  else
    prior = trellis_likelihood(:,j-1) + transmat;
  end
  % Only evaluate likely paths further
  max_prior = max(prior);
  % Use beamwidth to limit number of evaluations
  eval_id = find(max_prior>(max(max_prior)-beamwidth));
  acoustic_likelihood = -inf(1,num_states);
  for k=1:length(eval_id)
    % Evaluate GMM for k-th state in list
    stateid = states(eval_id(k));
    [weights, means, sigmas] = gmms{:,stateid};
    acoustic_likelihood (eval_id(k)) = ...
      sum(log(sum(normpdf(features(:,j),means,sigmas).*weights,2)));
  end
  % Consider acoustic information
  likelihood = prior + acoustic_likelihood;
  % Only keep the most likely state transition (Viterbi)
  [trellis_likelihood(:,j), trellis_state(:,j)] = max(likelihood);
end

% Trace back the most likely path
viterbi_path = nan(num_frames,1);
% Start with most likely state if no end state was provided
if isnan(final_state)
  [~, final_state_tmp] = max(trellis_likelihood(:,end));
  viterbi_path(end) = final_state_tmp;
else
  viterbi_path(end) = final_state;
end
% Iterate from last state to the first
for j=num_frames:-1:2
  viterbi_path(j-1) = trellis_state(viterbi_path(j),j-1);
end

% Translate path into state sequence
state_sequence = states(viterbi_path);