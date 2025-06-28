"""This Script Generates a tokenized sentence (not word embeddings) 
    and EEG features for decoding.
 """
 

#  """
#  input_sample['target_ids']
#  input_sample['input_embeddings']
#  input_sample['input_attn_mask']
#  input_sample['input_attn_mask_invert']
#  input_sample['target_mask']
#  input_sample['seq_len']
#  """
 
import numpy as np
import torch
 
"""
     `0` first training example in `subject_name`
     print ( dataset_dict_task2_NR[subject_name][0].keys() )
 
     dict_keys(['content', 'sentence_level_EEG', 'word', 
     'word_tokens_has_fixation', 'word_tokens_with_mask',
     'word_tokens_all'])
 
     like : input_dataset_dict[key][i] 
     here `i` training example in subject `key`
        
"""
 
""" input_sample = get_input_sample(
             input_dataset_dict[key][i], -->key is subject and i is sentence 
             self.tokenizer,
             self.eeg_type,
             self.bands
             )     
"""


def get_input_sample(sent_obj, tokenizer, eeg_type, bands, max_len=56):
    """Get a sample for a given sentence and subject EEG data.

    Args
    -------
        sent_obj (dict): A sentence object with EEG data.
        tokenizer: An instance of the tokenizer used to convert text to tokens.
        eeg_type (str): The type of eye-tracking features.
        bands (list): The EEG frequency bands to use.
        max_len (int, optional): Maximum length of the input. Defaults to 56.

    Returns
    -------
        input_sample (dict or None):
            - 'target_ids': Tokenized and encoded target sentence.
            - 'input_embeddings': Word-level EEG embeddings of the sentence.
            - 'input_attn_mask': Attention mask for input embeddings.
            - 'input_attn_mask_invert': Inverted attention mask.
            - 'target_mask': Attention mask for target sentence.
            - 'seq_len': Number of non-padding tokens in the sentence.

            Returns None if the input sentence is invalid or contains NaNs.
    """



    def normalize_1d(input_tensor):
        mean = torch.mean(input_tensor)
        std = torch.std(input_tensor)
        input_tensor = (input_tensor - mean)/std
        return input_tensor
       


    # 840
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands: #8
            frequency_features.append(     #GD          #GD_t1
                word_obj['word_level_EEG'][eeg_type][eeg_type+band] #105
                )
            

        word_eeg_embedding = np.concatenate(frequency_features) #(8*105)-->(840)
        if len(word_eeg_embedding) != 105*len(bands):
            # print(f'expect word eeg embedding dim to be {105*len(bands)},
            # but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)




    # 840 from the sentence
    def get_sent_eeg(sent_obj, bands): #sent_obj --> sent_obj[key][i]
        sent_eeg_features = []
        for band in bands: #loop 8 times each time to get 105 features. 
            key = 'mean'+band  #mean_t1 , mean_t2 , mean_a1 , mean_a2 , ..
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key]) 

        """stack 8 elements to get 8 * 105 """
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)    #840
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    


    # i passed to him sent_obj[key][i] --> subject name is key and number of sent is i

    """this is for the sentence.content"""
    if sent_obj is None:
        # print(f'  - skip bad sentence')
        return None
    

    # Start from here 
    input_sample = {}
    # get target label
    target_string = sent_obj['content'] #text 
    
    target_tokenized = tokenizer(
                target_string, 
                padding='max_length',
                max_length=max_len,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
                )   
    
    
    # ['input_ids'][0] gives the first (and in this case, only) tokenized sequence in the batch.
    # here keep just the ids , not the word vectors
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    

    """useless"""
    """sentence level eeg features"""
    # get sentence level EEG features (840)
    # sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands) #sent_obj[key][i] , ALL
    # if torch.isnan(sent_level_eeg_tensor).any():
    #     # print('[NaN sent level eeg]: ', target_string)
    #     return None





    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')



    # get input embeddings 
    word_embeddings = []  #num of words * 840  

    # all the words in the current sentence. with it's word level EEG features
    for word in sent_obj['word']: #looping over words 
        # add each word's EEG embedding as Tensors   (len: 840)
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        
        if word_level_eeg_tensor is None:   # check none, for v2 dataset
            return None
        
        if torch.isnan(word_level_eeg_tensor).any():
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            return None
        
        """if sentence is 9 words ,so u will have list of len 9 each item is
        tensor of shape 840"""
        word_embeddings.append(word_level_eeg_tensor)




    # collate_fn like style
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))




    """now if u have sent of 9 words (list) each word is a tensor of 840
    and u make padding to the max_len = 56 , this means u will add to ur list
    47 tensor of zeros ,which u must ignore them later in the training process
    so we invented the `masking` .....
    for u , MAsking --> it means ignoring 
    if mask is the `1` so this is which is padded ."""


    # input_sample['input_embeddings'].shape = max_len * (105*num_bands)
    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    len_sent_word = len(sent_obj['word'])  # len_sent_word <= max_len
    



    """u did padding , so the actual tokens set it with 1 and the others set it with 0"""
    # mask out padding tokens, 0 is masked out, 1 is not masked
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 56 [000000..]
    input_sample['input_attn_mask'][:len_sent_word] = torch.ones(len_sent_word) #[1110000]

    # mask out padding tokens reverted: handle different use case: this is for
    # pytorch transformers. 1 is masked out, 0 is not masked
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    input_sample['input_attn_mask_invert'][:len_sent_word] = torch.zeros(len_sent_word)




    """this is the masking that for Sentence(Target) 
       which is output of Bart Tokenizer by default .
    -- to also be ignored during word embedding and training and reduce the overhead."""
    
    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0] #output of Bart
    input_sample['seq_len'] = len(sent_obj['word'])



    # clean 0 length data
    if input_sample['seq_len'] == 0:
        # print('discard length zero instance: ', target_string)
        return None

    return input_sample
