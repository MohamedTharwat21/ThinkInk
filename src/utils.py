import os
import torch
import random
import numpy as np
from transformers.modeling_outputs import BaseModelOutput


 
# Translate EEG to sentence
def translate_sentence(model, eeg, eeg_mask, tokenizer, device, max_length=50, num_beams=4):
    """
    Translates an EEG input tensor into a natural language sentence using a seq2seq model with beam search.

    Args:
        model: The trained EEG-to-text model with encoder and T5 decoder.
        eeg (torch.Tensor): EEG input tensor of shape (1, seq_len, hidden_dim).
        eeg_mask (torch.Tensor): Attention mask for EEG input.
        tokenizer: Tokenizer used for encoding/decoding.
        device: CUDA or CPU device.
        max_length (int): Maximum length of the generated sentence.
        num_beams (int): Number of beams for beam search decoding.

    Returns:
        str: The translated natural language sentence.
    """
    model.eval()
    with torch.no_grad():
        # Forward EEG input through encoder
        encoder_hidden_states = model.encoder(eeg.to(device), eeg_mask.to(device))
        
        # Wrap encoder output in BaseModelOutput format
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Prepare decoder input (start token)
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

        # Generate output with beam search
        generated_ids = model.t5_model.generate(
            input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode generated IDs to string
        translated_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return translated_sentence



# BLEU score calculation
def bleu(data_loader, model, tokenizer, device, max_samples=100):
    references = []
    hypotheses = []
    for i, (eeg, _, eeg_mask, _, _, _, _, raw_sentence) in enumerate(data_loader):
        if i >= max_samples:
            break
        eeg = eeg.to(device)  # (batch_size, seq_length, input_dim)
        eeg_mask = eeg_mask.to(device)  # (batch_size, seq_length)
        pred = translate_sentence(model, eeg, eeg_mask, tokenizer, device)
        references.append([raw_sentence[0].split()])  # Assuming batch_size=1 for simplicity
        hypotheses.append(pred.split())
    return corpus_bleu(references, hypotheses)


# Save checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)


# Load check point
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
