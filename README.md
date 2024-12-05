# ThinkInk
## This project aims to develop a groundbreaking system that can decode brain activity, specifically EEG signals, into text. By harnessing the power of machine learning, we strive to bridge the gap between thought and expression, enabling silent communication and empowering individuals with speech impairments.

## Contents
0. [Project Overview](#Project-Overview)
0. [Model](#model)
0. [Data Collection](#data-collection)
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
   This stage concatenates the features of different frequency bands corresponding to a single word to form a unified word-level EEG feature.
   
   Each band’s EEG features are of size 105, and the concatenation creates a comprehensive representation for each word.

    
### Pre-Encoder
   The pre-encoder transforms the original EEG feature space into the embedding space required by the pre-trained Seq2Seq model. 

   A non-linear transformation to map the concatenated features to a higher-dimensional space (size 840).
   
   A Transformer encoder, which further processes the features to capture sequential dependencies and richer representations.
   
   The pre-encoder bridges the gap between raw EEG signals and the format needed for language generation.


### Pre-Trained Seq2Seq Model
#### The pre-trained Seq2Seq component is responsible for generating the output sentence. It consists of:
   A pre-trained encoder, which processes the EEG-derived embeddings and encodes the input sequence.
   
   A pre-trained decoder, which generates the natural language sentence from the encoded representation.
    
   Both the encoder and decoder work with a feature size of 1024, ensuring high-quality semantic representation and decoding.




## Data Collection

![image](https://github.com/user-attachments/assets/942f50da-4890-410b-96e2-7f4cfe8c73b1)

**Figure 5**: EEG Brain Signals.  


## Dataset

**Dataset: ZuCo Benchmark**
The dataset used for this project is derived from the **ZuCo Benchmark**, which combines data from two EEG datasets: **ZuCo [Hollenstein et al., 2018]** and **ZuCo 2.0 [Hollenstein et al., 2020]**. This benchmark provides a rich corpus of EEG signals and eye-tracking data collected during natural reading activities, making it highly suitable for EEG-to-Text research.



### **Data Sources**
#### Reading Materials
The EEG signals were recorded while participants read text from two primary sources:
- **Movie Reviews**
- **Wikipedia Articles**

#### EEG Signals
- Each EEG-text pair in the dataset includes a sequence of **word-level EEG features**.
- **Word-Level EEG Features (E)**: For each word, EEG signals consist of 8 frequency bands:
  1. **Theta1 (4–6 Hz)**
  2. **Theta2 (6.5–8 Hz)**
  3. **Alpha1 (8.5–10 Hz)**
  4. **Alpha2 (10.5–13 Hz)**
  5. **Beta1 (13.5–18 Hz)**
  6. **Beta2 (18.5–30 Hz)**
  7. **Gamma1 (30.5–40 Hz)**
  8. **Gamma2 (40–49.5 Hz)**



#### **Feature Construction**
- Each frequency band feature has a **fixed dimension of 105**.
- To construct the final **word-level feature vector**, all 8 frequency band features are concatenated, resulting in a vector of size **840 (E ∈ R⁸⁴⁰)**.
- All features are normalized using **Z-scoring**, as done in prior work (e.g., Willett et al., 2021).



#### **Dataset Splitting**
- The dataset is split into three parts:
  - **Training Set**: 80%
  - **Validation Set**: 10%
  - **Test Set**: 10%

#### Subject Consistency
- Each subset (train, validation, test) is designed to include unique sentences, ensuring **no overlapping sentences** between sets.
- However, the same set of subjects is maintained across all subsets.



#### **Statistics**

|                  | **Training** | **Validation** | **Testing** | 
|------------------|--------------|----------------|-------------|
| **Pairs**        | 14,567       | 1,811          | 1,821       |
| **Unique Sentences** | 1,061       | 173            | 146         |
| **Subjects**     | 30           | 30             | 30          |
| **Avg. Words**   | 19.89        | 18.80          | 19.23       |

 
Statistics for the ZuCo benchmark. **“# pairs”** means the number of EEG-text pairs, **“# unique sent”** represents the number of unique sentences, **“# subject”** denotes the number of subjects and **“avg.words”** means the average number of words of sentences.


#### **References**
- Hollenstein, N., et al. (2018). *ZuCo: Zurich Cognitive Language Processing Corpus.*
- Hollenstein, N., et al. (2020). *ZuCo 2.0: The Zurich Cognitive Corpus - An Updated Dataset for EEG and Eye-Tracking Research.*
- Willett, D., et al. (2021). *EEG Feature Standardization via Z-Scoring.*
- Wang, S., & Ji, H. (2022). *Unified Representation Learning for EEG-to-Text Tasks.*

This dataset serves as the foundation for training and validating the BrainTranslator model, enabling the transformation of EEG signals into meaningful text representations.

