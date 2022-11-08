"""

python main.py --mode=train --workdir=autoenc_end2end --config=configs.py:encoder


"""

import os
import time
import datetime
import torch
import torch.nn as nn
from models import *
from datasets import *

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
torch.autograd.set_detect_anomaly(True)

config_flags.DEFINE_config_file(
  "config", None, "System Configuration.", lock_config=True)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.mark_flags_as_required(["workdir", "mode"])

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(_):
    os.makedirs(FLAGS.workdir,exist_ok=True)
    if FLAGS.mode == "train":
        import pickle 
        print(FLAGS.config)
        with open(f"{FLAGS.workdir}/config.pkl","wb") as f:
            pickle.dump(FLAGS.config,f)
        
        config = FLAGS.config 
        trainer = Trainer(FLAGS.workdir, config)
        trainer.train()
    elif FLAGS.mode == "eval":
        import pickle
        config = pickle.load(open(f"{FLAGS.workdir}/config.pkl","rb"))
    
    pass

def lr_scheduler(ckpt, config):
    return 1e-3
    # Triangle schedule
    # if ckpt["step"] < config.train.warmup:
    #     return ckpt["step"] * (config.train.lr / config.train.warmup)
    # return -(ckpt["step"] - config.train.iterations) * (config.train.lr / (config.train.iterations - config.train.warmup))

    # Attention is all you need (args.lr is not used)
    return config.transformer.embed_dim**-0.5 * min(ckpt["step"]**-0.5, ckpt["step"]*config.train.warmup**-1.5)

def time_since(since):
    return str(datetime.timedelta(seconds=round(time.time()-since)))

def print_progress(batch, time, loss, valid_loss):
    print("B%6d %s \t train_loss: %.4f val_loss %.4f"%(batch, time, loss, valid_loss), flush=True)

def get_model(workdir, config):
    if config.type == "autoencoder":
        encoder = BidirectionalEncoder(config)
        if config.pt_enc_path != "None":
            pt_enc_ckpt = torch.load(config.pt_enc_path,map_location="cpu")
            encoder.load_state_dict(pt_enc_ckpt["model_state_dict"])
        decoder = Decoder(config, encoder=encoder)
        decoder_ckpt_path = f"{workdir}/{config.dec_path}"
        if os.path.exists(decoder_ckpt_path):
            decoder_ckpt = torch.load(decoder_ckpt_path,map_location="cpu")
            decoder.load_state_dict(decoder_ckpt["model_state_dict"])
        else:
            decoder_ckpt = None 
        return {
            "model": decoder,
            "ckpt": decoder_ckpt,
            "ckpt_path": decoder_ckpt_path
        }
    elif config.type == "encoder":
        encoder = BidirectionalEncoder(config)
        encoder_ckpt_path = f"{workdir}/{config.enc_path}"
        if os.path.exists(encoder_ckpt_path):
            encoder_ckpt = torch.load(encoder_ckpt_path,map_location="cpu")
            encoder.load_state_dict(encoder_ckpt["model_state_dict"])
        else:
            encoder_ckpt = None 
        return {
            "model": encoder,
            "ckpt": encoder_ckpt,
            "ckpt_path": encoder_ckpt_path
        }


class Trainer:
    def __init__(self, workdir, config, device="cuda:0"):
        self.config = config

        model_info = get_model(workdir, config)
        self.device = device
        self.model = model_info["model"].to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), weight_decay=1e-5)
        self.ckpt = model_info["ckpt"]
        self.ckpt_path = model_info["ckpt_path"]

        if self.ckpt is None:
            self.ckpt = {
                'step': 0,
                'best_loss': 1000.0,
                'train_losses': [],
                'valid_losses': [],
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'time': time.time(),
            }
        else:
            self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])

    def train(self):
        config = self.config 

        print(f"Training {config.type}")
        dataset_path=config.dataset.path
        mode=config.dataset.mode

        train_data = MidiMelodyDataset(f"{dataset_path}/train_data.pickle", seq_len=config.transformer.seq_len)
        valid_data = MidiMelodyDataset(f"{dataset_path}/test_data.pickle", seq_len=config.transformer.seq_len)
        
        model = self.model 
        optimizer = self.optimizer
        checkpoint = self.ckpt
        criterion = nn.CrossEntropyLoss()

        best_loss = checkpoint['best_loss']
        all_train_losses = checkpoint['train_losses']
        all_valid_losses = checkpoint['valid_losses']

        print("Num trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
        
        # Training pass
        while True:
            model.train()
            batch = train_data.get_batch(config.train.batch_size)
            tgt = batch["tgt"].to(self.device)
            
            pad_mask = (tgt == PAD_IDX).to(self.device)
            inp = batch["inp"].to(self.device)
            optimizer.zero_grad()
            
            if config.type == "encoder":
                out=model(inp)
                out_mask=tgt==-1
                # convert to self.device
                out_mask=out_mask.to(self.device)
                loss=criterion(out[~out_mask],tgt[~out_mask])
                
            elif config.type == "autoencoder":
                out = model(inp,tgt)
                loss = criterion(out[~pad_mask], tgt[~pad_mask])
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise Exception("Nans!")
            self.ckpt["step"] += 1
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr_scheduler(self.ckpt, config)
            optimizer.step()
            # Print progress
            if self.ckpt["step"] % config.train.print_every == 0:
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for val_idx in range(config.train.print_every):
                        vbatch = valid_data.get_batch(config.train.batch_size)
                        tgt = vbatch["tgt"].to(self.device)
                        
                        pad_mask = (tgt == PAD_IDX).to(self.device)
                        inp = vbatch["inp"].to(self.device)
                        out = model(inp,tgt)
                        if config.type == "encoder":
                            pass
                        elif config.type == "autoencoder":
                            valid_loss += criterion(out[~pad_mask], tgt[~pad_mask])
                valid_loss = valid_loss / (config.train.print_every)
                all_train_losses.append(loss.detach())
                all_valid_losses.append((valid_loss.detach()))
                print_progress(self.ckpt["step"], time_since(self.ckpt["time"]), loss.detach(), valid_loss.detach())

                if config.eval.generate:
                    for fname in config.eval.inp_files:
                        model.generate_sequence(fname, os.path.splitext(fname)[0] + "%d.mid"%(self.ckpt["step"]), device=self.device)

                # Checkpoint
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    temp_path = os.path.splitext(self.ckpt_path)[0] + "_temp.pt"
                    
                    self.ckpt["best_loss"] = best_loss
                    self.ckpt["model_state_dict"] = model.state_dict()
                    self.ckpt["optimizer_state_dict"] = optimizer.state_dict()

                    torch.save(self.ckpt, temp_path)
                    os.replace(temp_path, self.ckpt_path)
                    print('Checkpoint.', flush=True)

if __name__ == '__main__':
    app.run(main)
