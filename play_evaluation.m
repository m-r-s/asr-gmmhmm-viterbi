close all
clear
clc

graphics_toolkit qt;

% Evaluation parameters
results_file = 'results.mat';

tic;
load(results_file);
printf('loaded results from %s in %.1fms\n', results_file, toc*1000);

labels = regexprep(samples,'_.*$','');

% Remove silence from transcriptions
transcriptions = cellfun(@(x) x(~ismember(x,'sil')), transcriptions,'UniformOutput',0);

correct = 0;
insertions = 0;
deletions = 0;
for i=1:length(samples)
  tmp_correct = any(ismember(transcriptions{i},labels{i}));
  tmp_insertion = length(transcriptions{i})-tmp_correct;
  tmp_deletion = length(labels(i))-tmp_correct;
  correct += tmp_correct;
  insertions += tmp_insertion;
  deletions += tmp_deletion;
end

printf('%i/%i (% 6.2f%%) words correctly recognized with %i/%i insertions/deletions\n',correct,length(samples),correct/length(samples)*100,insertions,deletions);