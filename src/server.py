from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from http import HTTPStatus
# from configuration import cfg

app = Flask(__name__)

# Custom Transformer Encoder for EEG features
class EEGTransformerEncoder(nn.Module):
    def __init__(self,
                   input_dim=840, 
                   d_model=512, 
                   nhead=8, 
                   num_layers=6,
                   dim_feedforward=2048, 
                   dropout=0.1):
        
        super(EEGTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.dropout(self.input_projection(x))
        key_padding_mask = (mask == 0)  # 1s for valid, 0s for padded -> True for padded, False for valid
        output = self.transformer_encoder(x, 
                                          src_key_padding_mask=key_padding_mask)
        return output



# Hybrid Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 t5_model):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.t5_model = t5_model

    def forward(self, 
                eeg, 
                eeg_mask, 
                decoder_input_ids, 
                labels=None):
        
        encoder_outputs = self.encoder(eeg, 
                                       eeg_mask)
        outputs = self.t5_model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_outputs,),
            labels=labels,
            return_dict=True
        )
        return outputs


# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
encoder = EEGTransformerEncoder().to(device)
model = Seq2Seq(encoder, t5_model).to(device)



# try:
#     checkpoint = torch.load("my_checkpoint.pth.tar", map_location=device, weights_only=True)
#     model.load_state_dict(checkpoint["state_dict"])
# except FileNotFoundError:
#     raise Exception("Checkpoint file 'my_checkpoint.pth.tar' not found")
# except KeyError:
#     raise Exception("Checkpoint missing 'state_dict' key")



model.eval()

# ignore it
# Optional: Enable FP16 for lower VRAM usage
# model = model.half()

def translate_sentence(eeg, 
                       eeg_mask, 
                       max_length=56):
    with torch.no_grad():
        encoder_outputs = model.encoder(eeg, eeg_mask)  # (batch_size, 56, 512)
        outputs = model.t5_model.generate(
            encoder_outputs=(encoder_outputs,),
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )
        translated = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return translated



@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        if not data or 'eeg' not in data or 'eeg_mask' not in data:
            return jsonify({'error': 'Missing eeg or eeg_mask'}), HTTPStatus.BAD_REQUEST

        eeg = np.array(data['eeg'], dtype=np.float32)
        eeg_mask = np.array(data['eeg_mask'], dtype=np.float32)

        # Validate input shapes
        eeg_tensor = torch.tensor(eeg).to(device)
        eeg_mask_tensor = torch.tensor(eeg_mask).to(device)
        if eeg_tensor.ndim == 2:
            eeg_tensor = eeg_tensor.unsqueeze(0)
            eeg_mask_tensor = eeg_mask_tensor.unsqueeze(0)
        if eeg_tensor.shape[1:] != (56, 840) or eeg_mask_tensor.shape[1:] != (56,):
            return jsonify({'error': 'Invalid input shapes'}), HTTPStatus.BAD_REQUEST

        translated = translate_sentence(eeg_tensor, eeg_mask_tensor)
        return jsonify({'translated': translated})
    except Exception as e:
        return jsonify({'error': str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
