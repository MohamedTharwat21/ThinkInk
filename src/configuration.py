import torch

cfg = {
        'seed': 312,
        'subject_choice': 'ALL',
        'eeg_type_choice': 'GD',
        'bands_choice': 'ALL',
        'dataset_setting': 'unique_sent',
        'batch_size': 16,  #what ?? take care
        'shuffle': False,
        'input_dim': 840,
        'num_layers': 1,  # 6
        'nhead': 1,  # 8
        'dim_pre_encoder': 2048,
        'dim_s2s': 1024,
        'dropout': 0,
        'lr': 1e-6,
        'epochs': 5,
        'wandb': True,
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
