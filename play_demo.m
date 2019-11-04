close all
clear
clc

% Loads training and testing audio data and extracts features
play_prepare

% Generates and trains the GMM/HMM word models with the trainings data
play_training

% Generates a bag of words grammar and recognizes the testing data
play_recognition

% Evaluates the word recognition performance (only works with single word recordings)
play_evaluation

% Star live recognition with the trained model
% Need playrec compiled with support for JACK and JACK running
% play_liveasr