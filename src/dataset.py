import copy
import pickle
import time
import torch
import torch.optim as optim
# import wandb
from torch.utils.data import DataLoader
# from transformers import BartTokenizer, BartForConditionalGeneration
# for sentence tokenization 
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from utils import set_seed 
from configuration import cfg 
from dataset import ZuCo




def get_train_dev_loader():

    """ ###  input_sample['input_embeddings'],
        ###  input_sample['seq_len'],
        ###  input_sample['input_attn_mask'],
        ###  input_sample['input_attn_mask_invert'],
        ###  input_sample['target_ids'],
        ###  input_sample['target_mask'],
        ###  input_sample['subject'],
        ###  input_sample['sentence']
    """

    dataset_path_task1 = r'/kaggle/input/eeg-dataset-task1-sr-task2-nr/task1-SR-dataset.pickle'
    whole_dataset_dicts = []
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))


    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small' , legacy=False)

    # tokens = t5_tokenizer(raw_sentence, 
    #                       max_length=max_length, 
    #                       padding='max_length', 
    #                       truncation=True, 
    #                       return_tensors='pt')['input_ids']


    train_set = ZuCo(
            whole_dataset_dicts,
            'train',
            tokenizer=t5_tokenizer,
            subject=cfg['subject_choice'],   #ALL
            eeg_type=cfg['eeg_type_choice'], #GD
            bands=cfg['bands_choice'],       #ALL
            setting=cfg['dataset_setting']   #unique_sent
        )

    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['batch_size'], 
        shuffle=cfg['shuffle']
    )



    dev_set = ZuCo(
        whole_dataset_dicts,
        'dev',
        tokenizer=t5_tokenizer,
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
    )

    dev_loader = DataLoader(
        dev_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
    )

    # dataloaders = {'train': train_loader, 'dev': dev_loader}
     
    return train_loader , dev_loader




def sainty_check():
    # batch_size = 1
    for batch, (EEG, _, mask, _, tokens, _, _, _) in enumerate(train_loader):
        EEG = EEG.to(cfg['device'])
        mask_pre_encoder = mask.to(cfg['device'])
        mask_seq2seq = (~mask.bool()).float().to(cfg['device'])  # Invert mask for seq2seq (0=masked, 1=not)
        labels = tokens.to(cfg['device'])


        print(f'EEG : {EEG.shape }')
        print(f'mask_pre_encoder : {mask_pre_encoder.shape }')
        print(f'mask_seq2seq : {mask_seq2seq.shape }')

        print(f'labels : {labels}')
        print(f'labels : {labels.shape}')
        print(mask)

        break


if __name__ == "__main__":
    # Data and dataloaders
    # dataset_path_task1 = os.path.join(
    #     '../', 'dataset', 'ZuCo',
    #     'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
    # )

    train_loader , dev_loader = get_train_dev_loader()
