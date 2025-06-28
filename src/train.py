from utils import set_seed
from configuration import config as cfg
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model import EEGTransformerEncoder , Seq2Seq
from utils import load_checkpoint , \
    translate_sentence , save_checkpoint , bleu
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
from dataset import get_train_dev_loader


def train_eeg_to_text_model():
    example_eeg , example_eeg_mask = inference_with_one_eeg()
    scaler = GradScaler()
    step = 0
    clip_grad_norm = 1.0
    accumulation_steps = 1  # Increase if you want gradient accumulation

    # Optional: add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}]")

        # Save checkpoint
        checkpoint = {"state_dict": model.state_dict(), 
                    "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
          

        # Evaluate on one example
        model.eval()
        with torch.no_grad():
            translated_sentence = translate_sentence(model, 
                                                     example_eeg, 
                                                     example_eeg_mask, 
                                                     t5_tokenizer, 
                                                     device, 
                                                     max_length=max_length)
            print(f"Translated example sentence: \n{translated_sentence}")

        model.train()
        running_loss = 0.0

        for batch, (EEG, _, eeg_mask, _, tokens, _, _, _) in tqdm(enumerate(train_loader)):
            EEG = EEG.to(device)
            eeg_mask = eeg_mask.to(device)
            decoder_input_ids = tokens[:, :-1].to(device)
            labels = tokens[:, 1:].to(device)

            with autocast():  # Mixed-precision forward
                outputs = model(EEG, eeg_mask, decoder_input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            if batch % 100 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch} | Loss: {running_loss / (batch + 1):.4f}")

            writer.add_scalar("Training loss", loss.item() * accumulation_steps, global_step=step)
            step += 1

        scheduler.step()



def inference_with_one_eeg():
    from dataset import whole_dataset_dicts
    from dataset_handler import get_input_sample 
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
    eeg_input = get_input_sample(whole_dataset_dicts[0]['ZAB'][55],
                    t5_tokenizer ,
                    eeg_type='GD' , 
                    bands = bands,
                    max_len= 56)

    # print(eeg_input['input_embeddings'].shape)
    # print(eeg_input['input_attn_mask'].shape)

    example_eeg = eeg_input['input_embeddings'].unsqueeze(0).float().to(device)  # (batch_size, seq_length, input_dim)
    example_eeg_mask = eeg_input['input_attn_mask'].unsqueeze(0).to(device)  # (batch_size, seq_length)

    # print(example_eeg.shape)
    # print(example_eeg_mask.shape)

    # Create dummy EEG input (mimicking training example)
    # example_eeg = torch.randn(1, 56, 840).to(device)  # (batch_size, seq_length, input_dim)
    # example_eeg_mask = torch.ones(1, 56).to(device)  # (batch_size, seq_length)
    # example_eeg_mask[:, 17:] = 0  # Mimic variable-length EEG sequence (17 valid positions)

    text = whole_dataset_dicts[0]['ZAB'][55]['content']
    print(f'target sent.: {text}')

    # Test translation
    translated_sentence = translate_sentence(model, 
                                            example_eeg,
                                            example_eeg_mask, 
                                            t5_tokenizer, 
                                            device, 
                                            max_length=max_length)

    print(f"Translated sentence: {translated_sentence}")

    """    
    target sent.: A strong script, powerful direction and splendid production design allows us to be transported into the life of Wladyslaw Szpilman, who is not only a pianist, but a good human being.
    Translated sentence: sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun sun
    """

    return example_eeg , example_eeg_mask



if __name__ == "__main__":
    set_seed(cfg['seed']) #seeding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available")



    # Training setup
    # num_epochs = 100
    # learning_rate = 0.001
    # batch_size = 64
    load_model = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 840  # EEG feature dimension
    d_model = 512  # Matches t5-small
    # nhead = 8
    # num_encoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    max_length = 56


    # Load T5 model and tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')


    # Initialize model
    encoder = EEGTransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers'],
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)


    model = Seq2Seq(encoder, t5_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=cfg['lr'],
                                weight_decay=1e-4)


    # Optionally load checkpoint
    if load_model:
        checkpoint = torch.load("my_checkpoint.pth.tar",
                                map_location=device)
        
        load_checkpoint(checkpoint, model, optimizer)

    # Tensorboard
    writer = SummaryWriter(f"runs/loss_plot")
    step = 0


    

    train_loader , dev_loader = get_train_dev_loader()

    # training
    train_eeg_to_text_model()


    # Evaluate BLEU score
    score = bleu(dev_loader, 
                model, 
                t5_tokenizer,
                device)

    print(f"BLEU score {score*100:.2f}")
