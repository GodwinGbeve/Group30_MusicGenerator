# Group30_MusicGenerator

# Music Generation with Neural Networks

## Overview

This repository contains code for generating music sequences using neural networks, specifically LSTM (Long Short-Term Memory) models. The project involves the preparation of MIDI data, hyperparameter tuning using Keras Tuner, and training a sequence model for music generation.

## Table of Contents

- [Installation](#installation)
- [Setting Up MIDI Data](#setting-up-midi-data)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Training](#model-training)
- [Generating Music](#generating-music)
- [Saving Model and Results](#saving-model-and-results)
- [Website Demonstration](#website-demonstration)
- [Deployment](#deployment)
  
## Installation

To run the code, you need to install the required packages. Execute the following commands:

```bash
sudo apt install -y fluidsynth
pip install --upgrade pyfluidsynth
pip install pretty_midi keras_tuner
```

This installs the necessary libraries for MIDI file processing, audio synthesis, and hyperparameter tuning.

## Setting Up MIDI Data

The code includes functions for processing MIDI files, extracting note information, and visualizing piano rolls. MIDI files should be placed in a specified folder within Google Drive.

## Hyperparameter Tuning

The hyperparameter tuning is performed using Keras Tuner, specifically the Hyperband tuner. The `build_hypermodel` function defines the architecture of the neural network, and the tuner searches for the best combination of LSTM units and learning rates.

## Model Training

The LSTM model is defined using TensorFlow and Keras. It predicts three features: pitch, step, and duration. A custom loss function is used to handle specific requirements for step and duration predictions.

The training process includes an initial evaluation, adjustment of loss weights, and re-evaluation. Model training progress is visualized with a loss plot.

## Generating Music

Once the model is trained, music sequences are generated. The `predict_next_note` function uses the trained model to predict the next pitch, step, and duration. Generated notes are added to a sequence iteratively.

## Saving Model and Results

The final model weights and architecture are saved for future use. The weights are stored in `best_model_weights.h5`, and the architecture is saved as JSON in `best_model_architecture.json`. Additionally, the generated music is saved as a MIDI file named `output.mid`.

Feel free to explore the provided Jupyter notebook for a step-by-step walkthrough of the entire process.

## Website Demonstration

For a detailed demonstration of how the website works, please watch the following video:

(https://youtu.be/-8eyQEBGoHg)

## Deployment

The model and generated music sequences can be deployed on a website. Access the deployed application using the following link:

(http://localhost:8501)

Feel free to explore the provided Jupyter notebook for a step-by-step walkthrough of the entire process.


---

This README provides a clear overview of the project, the steps involved, and instructions for setting up and running the code. Adjust the content as needed based on additional details or specific project requirements.
