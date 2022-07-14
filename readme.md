# Code demo for Schroeder, Perkins et al. 'Cortical control of virtual self-motion using task-specific subspaces'

This package contains the code to run an offline version of the decode algorithm. A dataset from
one day of training data and online control is included (Monkey G, data from 10-16-2018).

# System Requirements

The code is written for MATLAB (R2018b or later) and tested in MacOSX 10.13+.
Training the decoder requires the Manopt Toolbox for MATLAB: https://github.com/NicolasBoumal/manopt

# Installation

Install the Manopt Toolbox and then run directly in MATLAB.

# Demo + Instructions

Demo code is provided in demo.m. It trains a decoder on a training dataset, runs an offline version of the
decoding algorithm on a test dataset. It produces a trialized table of data and three plots demonstrating 
decoder performance. The table contains the primary decoder output (decoded position) as well as many of the 
intermediate variables decribed in the paper. Expected run time is <10min total. 



