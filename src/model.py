import torch
import torch.nn as nn
import torch.optim as optim


# Custom Transformer Encoder for EEG features (batch-first)
class EEGTransformerEncoder(nn.Module):
    """
    Input: EEG features (batch_size, 56, 840) and EEG mask (batch_size, 56).

    Architecture: A ((Transformer encoder)) with ((multi-head self-attention)), 
    designed to handle EEG features 
    [[ by projecting the 840-dimensional input to a hidden size compatible with T5’s decoder. ]]

    Output: Hidden states (batch_size, 56, d_model) to be used as 
    encoder_hidden_states for the T5 decoder.

    """
   
    def __init__(self, 
                 input_dim,  # input_dim = 840  # EEG feature dimension
                 d_model,    # d_model = 512    # Matches t5-small
                 nhead, 
                 num_layers, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        
        super(EEGTransformerEncoder, self).__init__()
        # convert (bs,56,840) to (bs,56,512) to match T5-samll LLM

        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(    #Transformer Encoder Layer
            d_model=d_model,  
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        #how many layers need to be stacked 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model


    def forward(self, x, mask):
        # x: (batch_size, seq_length, input_dim) = (N, 56, 840)
        # mask: (batch_size, seq_length) = (N, 56)
        x = self.dropout(self.input_projection(x))  # (batch_size, seq_length, d_model) (bs,56,512) 
        

        """Important Note : Torch masking is different from Hugging face masking"""
        # Transformer expects src_key_padding_mask: True for padded positions (this is the inversion)
        # Invert mask for Hugging Face: True for padded (0s), False for valid (1s)
        # or You can use the inverted from the dataset .
        key_padding_mask = (mask == 0)  # (batch_size, seq_length)

        # EEG + Masking
        output = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        return output  # (batch_size, seq_length, d_model) (bs,56,512)
    




# Hybrid Seq2Seq Model (Custom EEG Encoder + T5 Decoder)
class Seq2Seq(nn.Module):

    """
    
    Combine the (custom Transformer encoder) and (T5 decoder) into a (hybrid Seq2Seq model).
    The encoder processes EEG features, and the decoder generates text (autoregressively) ,
    using teacher forcing during training.
    
     
    decoder_input_ids are shifted tokens (excluding the last token), 
    and labels are shifted forward (excluding the first token) for teacher forcing.
    """

    def __init__(self, encoder, t5_model):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.t5_model = t5_model

    def forward(self, 
                eeg, 
                eeg_mask, 
                decoder_input_ids, 
                labels=None):
        """
        The forward method passes EEG features and mask to 
        the encoder, then feeds the output to T5 with 
        decoder_input_ids and labels.

        """
        
        # eeg: (batch_size, seq_length, input_dim) = (N, 56, 840)
        # eeg_mask: (batch_size, seq_length) = (N, 56)
        # decoder_input_ids: (batch_size, seq_len)
        # labels: (batch_size, seq_len) or None
        
        # encoder_hidden_state (Memory)
        encoder_outputs = self.encoder(eeg, eeg_mask)  # (batch_size, seq_length, d_model)
        
        # Pass to T5 decoder
        outputs = self.t5_model(
            decoder_input_ids = decoder_input_ids, #(batch_size, seq_len)
            encoder_outputs = (encoder_outputs,),
            labels=labels,
            return_dict=True
        )
        return outputs
