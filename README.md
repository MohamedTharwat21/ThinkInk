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

### **ZuCo 2.0 Corpus: Data Collection and Processing**
This section describes the contents, participants, experimental design, and preprocessing techniques used in the ZuCo 2.0 dataset.


### **1. Participants**


#### Overview
- Total Participants: **19** (data from 1 participant discarded due to technical issues).  
- Final Dataset: **18 participants**.  
- Demographics:  
  - **Mean Age**: 34 years (SD = 8.3).  
  - **Gender**: 10 females.  
  - **Native Language**: English (participants from Australia, Canada, UK, USA, or South Africa).  


### **2. EEG Data Acquisition**

#### Equipment and Setup
- **EEG System**: 128-channel Geodesic Hydrocel system (Electrical Geodesics).  
- **Sampling Rate**: 500 Hz.  
- **Bandpass Filter**: 0.1 to 100 Hz.  
- **Reference Electrode**: Set at electrode **Cz**.  
- **Electrode Setup**:  
  - Head circumference measured to select the appropriate EEG net size.  
  - Impedance of each electrode kept below **40 kOhm** to ensure proper contact.  
  - Impedance levels checked every 30 minutes after every 50 sentences.  

| **Specification**      | **Details**                   |
|-------------------------|-------------------------------|
| **System**             | Geodesic Hydrocel (128-channel) |
| **Sampling Rate**       | 500 Hz                       |
| **Bandpass**            | 0.1–100 Hz                   |
| **Reference Electrode** | Cz                           |
| **Electrode Impedance** | ≤ 40 kOhm                    |



### **3. Data Preprocessing and Extraction**

#### Raw and Preprocessed Data
- **Tools Used**:  
  - Preprocessing performed using **Automagic (version 1.4.6)** for automatic EEG data cleaning and validation.  
  - **MARA (Multiple Artifact Rejection Algorithm)**: A supervised machine learning algorithm used for automatic artifact rejection.  


#### EEG Channel Selection
- **Channels Used**:  
  - **105 EEG channels** from the scalp recordings.  
  - **9 EOG channels** for artifact removal.  
  - **14 channels** (mainly on the neck and face) were discarded.  

| **Channels**             | **Count**           | **Purpose**              |
|--------------------------|---------------------|--------------------------|
| **EEG Channels**         | 105                 | Scalp recordings         |
| **EOG Channels**         | 9                   | Artifact removal         |
| **Discarded Channels**   | 14                  | Neck and face channels   |

#### Artifact Rejection
- **Artifacts Addressed**: Eye movements and muscle noise.  
- **Artifact Rejection Criterion**: Trials with transient noise above **90 µV** were excluded.


### **4. Synchronization and Feature Extraction**

#### Synchronization
- EEG signals synchronized with eye-tracking data to enable analyses time-locked to fixation onsets.  

#### Oscillatory Power Measures
- **Frequency Bands**:  
  - Theta1 (4–6 Hz), Theta2 (6.5–8 Hz).  
  - Alpha1 (8.5–10 Hz), Alpha2 (10.5–13 Hz).  
  - Beta1 (13.5–18 Hz), Beta2 (18.5–30 Hz).  
  - Gamma1 (30.5–40 Hz), Gamma2 (40–49.5 Hz).  
- **Processing Steps**:  
  - Bandpass filtering for each frequency band.  
  - **Hilbert Transformation** applied to preserve temporal amplitude information.  

| **Frequency Band** | **Range (Hz)** |
|--------------------|----------------|
| **Theta1**         | 4–6            |
| **Theta2**         | 6.5–8          |
| **Alpha1**         | 8.5–10         |
| **Alpha2**         | 10.5–13        |
| **Beta1**          | 13.5–18        |
| **Beta2**          | 18.5–30        |
| **Gamma1**         | 30.5–40        |
| **Gamma2**         | 40–49.5        |

#### Sentence-Level EEG Features
- **Power Calculations**:  
  - Power computed for each frequency band.  
  - Difference in power spectra between frontal left and right homologue electrodes.  
- **Eye-Tracking Features**: Corresponding EEG features computed for each fixation.  

| **Feature**                 | **Description**                                |
|-----------------------------|-----------------------------------------------|
| **Frequency Band Power**    | Power calculated for each band.               |
| **Electrode Differences**   | Power difference between left/right pairs.    |
| **Artifact Rejection**      | Channels excluded for noise > 90 µV.          |


### **5. Summary**
- The data collection process ensures high-quality EEG recordings synchronized with eye-tracking data.  
- Advanced preprocessing techniques, such as MARA and Hilbert transformations, enhance the usability of the dataset for downstream analyses.  
- The dataset captures meaningful EEG signals corresponding to natural reading tasks, enabling in-depth exploration of brain activity and behavior.


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
| **# Pairs**        | 14,567       | 1,811          | 1,821       |
| **# Unique Sentences** | 1,061       | 173            | 146         |
| **# Subjects**     | 30           | 30             | 30          |
| **# Avg. Words**   | 19.89        | 18.80          | 19.23       |

 
Statistics for the ZuCo benchmark. **“# pairs”** means the number of EEG-text pairs, **“# unique sent”** represents the number of unique sentences, **“# subject”** denotes the number of subjects and **“avg.words”** means the average number of words of sentences.


#### **References**
- Hollenstein, N., et al. (2018). *ZuCo: Zurich Cognitive Language Processing Corpus.*
- Hollenstein, N., et al. (2020). *ZuCo 2.0: The Zurich Cognitive Corpus - An Updated Dataset for EEG and Eye-Tracking Research.*
- Willett, D., et al. (2021). *EEG Feature Standardization via Z-Scoring.*
- Wang, S., & Ji, H. (2022). *Unified Representation Learning for EEG-to-Text Tasks.*

This dataset serves as the foundation for training and validating the BrainTranslator model, enabling the transformation of EEG signals into meaningful text representations.

