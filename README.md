# ThinkInk
## This project aims to develop a groundbreaking system that can decode brain activity, specifically EEG signals, into text. By harnessing the power of machine learning, we strive to bridge the gap between thought and expression, enabling silent communication and empowering individuals with speech impairments.

## Contents
0. [Project Overview](#Project-Overview)
0. [Model](#model)
0. [Baselines and Experiments](#baselines-and-experiments)  
0. [Dataset](#dataset)
0. [Installation](#installation)  
0. [Key Insights](#key-insights)
0. [Results](#results) 
0. [Future Improvements](#Future-Improvements)


## Project Overview

**Electroencephalography-to-Text (EEG-to-Text) generation** is an emerging field at the intersection of neuroscience, artificial intelligence, and human-computer interaction. This groundbreaking technology focuses on transforming brain activity, captured through EEG signals, directly into natural text. It represents a pivotal innovation in Brain-Computer Interfaces **(BCIs)**, opening doors to novel applications that enhance communication, accessibility, and productivity for individuals worldwide.

**EEG-to-Text technology** offers a life-changing solution for individuals who cannot speak or write due to conditions like ALS, paralysis, or severe motor impairments. **By decoding their thoughts into text** , this project provides a pathway for communication and interaction, restoring independence and quality of life.

## Model
The model is designed to transform **word-level EEG features** , these features will be obtained after **Features Extraction Step** which is done by taking raw EEG recording signals then convert it to Preprocessed version, into coherent natural language sentences. It is composed of **three key components**, each playing a crucial role in the processing pipeline:

### Word-Level EEG Feature Construction
  1-This stage concatenates the features of different frequency bands corresponding to a single word to form a unified word-level EEG feature.
  2-Each band’s EEG features are of size 105, and the concatenation creates a comprehensive representation for each word.
  
### Pre-Encoder
The pre-encoder transforms the original EEG feature space into the embedding space required by the pre-trained Seq2Seq model.
 
  1.A non-linear transformation to map the concatenated features to a higher-dimensional space (size 840).
  2.A Transformer encoder, which further processes the features to capture sequential dependencies and richer representations.
  3.The pre-encoder bridges the gap between raw EEG signals and the format needed for language generation.

### Pre-Trained Seq2Seq Model
  1-The pre-trained Seq2Seq component is responsible for generating the output sentence. It consists of:
  2-A pre-trained encoder, which processes the EEG-derived embeddings and encodes the input sequence.
  3-A pre-trained decoder, which generates the natural language sentence from the encoded representation.
  4-Both the encoder and decoder work with a feature size of 1024, ensuring high-quality semantic representation and decoding.
