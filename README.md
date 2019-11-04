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

The main script of this repository, which executes all these steps in comparatively few lines of code, is *play_asr.m*
To learn the concepts a single script is suitable.

To perform (even small) experiments it is better to partition the problem.
A possible partition is pesented in the following files:
* *play_prepare.m*
* *play_training.m*
* *play_recognition.m*
* *play_evaluation.m*

The whole chain can be run with the script *play_demo.m*.

In addition, the learned model can be used to illustrate Viterbi-decoding in a live recognition session with the script *play_live.m*.
This script requires a working copy of playrec.

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
