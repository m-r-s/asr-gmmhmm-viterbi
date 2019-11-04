function [features, logms] = feature_extraction(signal, fs)
  % Remove DC component
  signal -= mean(signal);
  % Normalize signal power
  signal ./= rms(signal);
  % Log Mel-spectrogram
  logms = log_mel_spectrogram(signal,fs);
  %features = logms;
  %% Alternative features:
  features = mfcc(logms); % Mel frequency cepstral coefficients (MFCC)
  %features = sgbfb(logms); % Separable Gabor filter bank features (SGBFB)
end

