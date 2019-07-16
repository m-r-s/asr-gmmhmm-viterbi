# Educational GNU/Octave scripts that implement a simple Viterbi-training based automatic word recognizer

## Content
This repository holds a set of scripts which implement the required steps to set up a simple Gaussian Mixture Model (GMM) and Hidden Markov model (HMM) based word recognition system.

The steps include:
1. Label generation
2. Feature extraction
3. Model initialization
4. Viterbi training
5. Transition matrix generation
6. Recognition
7. Transcription

The main script, which executes these steps, is *play_asr.m*

## Purpose
This educational code was partly taken from my lecture with the aim to be reasonably functional, compact _and_ understandable.
The idea is that you look at the code and execute it line by line with GNU/Octave to learn what happens in each step.

## Licence information
Copyright (C) 2019 Marc René Schädler
E-mail marc.r.schaedler@uni-oldenburg.de
Institute Carl-von-Ossietzky University Oldenburg, Germany

All code is licenced under the GPL v3 (cf. LICENCE).

The feature extraction code was taken from another project [1].

The example recordings (*.wav files) were made by Timo Baumann [2] and were published in the VoxForge project [3] under the GPL.

[1] https://github.com/m-r-s/reference-feature-extraction

[2] http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Original/48kHz_16bit/timobaumann-20071121-ziffern2.tgz

[3] http://www.voxforge.org/
