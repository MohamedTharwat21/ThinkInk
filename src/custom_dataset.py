"""ML-ready ZuCo dataset for EEG-to-Text decoding.
"""

import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import BartTokenizer
from torch.utils.data import Dataset, DataLoader

from data.get_input_sample import get_input_sample
from utils.HashTensor import HashTensor


class ZuCo(Dataset):
    """A custom dataset class for the ZuCo dataset.

    Split for a given task(s), subject(s) and unique subject/sentence setting.

    Constructor Arguments:
        - input_dataset_dicts (list or dict): The pickle data for each task.
        - phase (str): The dataset split. One of 'train', 'dev', or 'test'.
        - tokenizer (object): The tokenizer used to convert text to tokens
        - subject (str, optional): The subject(s) to use. Default is 'ALL'.
        - eeg_type (str, optional): The type of eye-tracking features.
        - bands (str or list, optional): The frequency bands. Default is 'ALL'.
        - setting (str, optional): 'unique_sent' or 'unique_subj'. Default is 'unique_sent'.

    Note:
    ----
        The dataset is split into three parts: 80% for training, 10% for
        development (dev), and 10% for testing based on the 'phase' argument.

        The 'unique_sent' setting creates the dataset by grouping sentences
        based on their uniqueness, while the 'unique_subj' setting groups the
        dataset based on unique subjects.

        WARNING!!! The 'unique_subj' setting is specific to the SR v1 dataset.

        For the 'unique_subj' setting, the following subjects are used:
        - ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for training
        - ['ZMG'] for dev
        - ['ZPH'] for test

    The getter method returns a tuple of:
        - Word-level EEG embeddings of the sentence.
        - Number of non-padding tokens in the sentence.
        - Attention mask for input embeddings (for huggingface, 1 is not masked, 0 is masked)
        - Inverted attention mask (for PyTorch, 1 is masked, 0 is not masked)
        - Tokenized target sentence.
        - Attention mask for target sentence.
        - The subject.
        - The target sentence.

    """


    def __init__(self,
                 input_dataset_dicts,
                 phase,
                 tokenizer,
                 subject='ALL',
                 eeg_type='GD',
                 bands='ALL',
                 setting='unique_sent'):

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        self.inputs = []
        self.tokenizer = tokenizer
        self.subject = subject  #ALL
        self.setting = setting  #unique_sent
        self.eeg_type = eeg_type  #GD
        self.train = 0.8
        self.dev = 0.1
        self.bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'] \
            if bands == 'ALL' else bands


        # go through all task datasets (3 iterations)
        for input_dataset_dict in input_dataset_dicts:

            # get the subject(s) key/name for this task
            subjects = list(input_dataset_dict.keys()) \
                if subject == 'ALL' else [subject]

            # number of sentences per subject in this task
            total_num_sentence = len(input_dataset_dict[subjects[0]])




            # create dataset grouped by unique sentence or subject
            if setting == 'unique_sent':
                self.unique_sent(
                    phase, subjects, input_dataset_dict, total_num_sentence
                    )
            elif setting == 'unique_subj':
                self.unique_subj(phase, input_dataset_dict, total_num_sentence)




    def __getitem__(self, idx):
        # ((self.inputs)) now is a list of 
        # input samples of len ((3 * (12 / 18) * 240)--if unique set.)
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['subject'],
            input_sample['sentence']
        )

    def __len__(self):
        return len(self.inputs)


    """
    info!!
    unique sent : sort the sentences for all the subj into   train/dev/test splits
    unique subjects : sort the subjetcs into                 train/dev/test splits
    """
    def unique_sent(self, 
                    phase, 
                    subjects, 
                    input_dataset_dict, 
                    total_num_sentence):
        
        # indices separating the sentences into train/dev/test splits
        train_divider = int(self.train * total_num_sentence)
        dev_divider = train_divider + int(self.dev * total_num_sentence)

        if phase == 'train':
            range_iter = range(train_divider)
        elif phase == 'dev':
            range_iter = range(train_divider, dev_divider)
        elif phase == 'test':
            range_iter = range(dev_divider, total_num_sentence)

        for key in subjects: #12 subjects 
            for i in range_iter: #train --> (0.8 * 300)=240
                self.append_input_sample(input_dataset_dict, key, i)


    def unique_subj(
            self, phase, input_dataset_dict, total_num_sentence
            ):
        # sort the subjetcs into train/dev/test splits
        if phase == 'train':
            subj_iter = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
        elif phase == 'dev':
            subj_iter = ['ZMG']
        elif phase == 'test':
            subj_iter = ['ZPH']

        for i in range(total_num_sentence):
            for key in subj_iter:
                self.append_input_sample(input_dataset_dict, key, i)



    """key is for subject k and i is for sentence i"""
    """input_dataset_dict --> which dataset dictionary (SR1,NR2,NR2.0)"""
    def append_input_sample(self, input_dataset_dict, key, i):
        input_sample = get_input_sample(
            input_dataset_dict[key][i],
            self.tokenizer,    #tokenizer
            self.eeg_type,     #GD
            self.bands         #ALL
            )
        
        
        # (((adding new keys))) on the dictionary
        if input_sample is not None:
            input_sample['input_embeddings'] = input_sample['input_embeddings'].to(torch.float)
            input_sample['subject'] = key
            input_dataset_dict[key][i]['word_tokens_all'] #initalization
            # take the tokens make them a sentence 
            input_sample['sentence'] = " ".join(
                input_dataset_dict[key][i]['word_tokens_all']
                )
            
            self.inputs.append(input_sample)



"""this function is just testing and i will not use it again 
   i will call zuco dataset class directly in training ."""
def main():
    """ML-ready ZuCo dataset sanity check."""
    # load the pickle files for all tasks
    whole_dataset_dicts = []

    dataset_path_task1 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
        )
    dataset_path_task2 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task2-NR', 'pickle', 'task2-NR-dataset.pickle'
        )
    dataset_path_task2_v2 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task2-NR-2.0', 'pickle', 'task2-NR-2.0-dataset.pickle'
        )

    whole_dataset_dicts = []
    for t in [dataset_path_task1, dataset_path_task2, dataset_path_task2_v2]:
        with open(t, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

  
    # check the number of subjects and unique sentences in each task
    for idx, dataset_dict in enumerate(whole_dataset_dicts):
        if idx == 0:
            num_sent = 400
            num_subj = 12
        elif idx == 1:
            num_sent = 300
            num_subj = 12
        else:
            num_sent = 349
            num_subj = 18

        assert len(dataset_dict) == num_subj
        
        # tharwat : loop over each subject 
        for key in dataset_dict:
            assert len(dataset_dict[key]) == num_sent


    # data config
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    subject_choice = 'ALL'
    eeg_type_choice = 'GD' #????
    bands_choice = 'ALL'
    dataset_setting = 'unique_sent'


    # check split size
    for split in tqdm(['train', 'dev', 'test']):
        dataset = ZuCo(
            whole_dataset_dicts,
            split,
            tokenizer,
            subject=subject_choice,
            eeg_type=eeg_type_choice,
            bands=bands_choice,
            setting=dataset_setting
            )
        
        print(f' {split}set size:', len(dataset))




if __name__ == '__main__':
    main()
