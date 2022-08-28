import math
import torch
import torch.nn as nn
from datasets import MASK_IDX, PAD_IDX, key_mapping, tok_mapping
from midi_processing import mid2dat_anna, dat2mid_anna


class BidirectionalEncoder(nn.Module):
    def __init__(self,
                 config):
        super(BidirectionalEncoder, self).__init__()
        
        
        self.embed_dim   = config.transformer.embed_dim
        self.config      = config
        
        self.embed       = nn.Embedding(config.transformer.num_tokens, config.transformer.embed_dim, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(config.transformer.embed_dim, config.transformer.dropout, max_len=config.transformer.seq_len)
        enc_layer        = nn.TransformerEncoderLayer(config.transformer.embed_dim, config.transformer.nhead, config.transformer.feedforward, config.transformer.dropout, batch_first=True, activation="gelu")
        self.enc         = nn.TransformerEncoder(enc_layer, config.transformer.layers, norm=nn.LayerNorm(config.transformer.embed_dim))
        self.fc          = nn.Linear(config.transformer.embed_dim, config.transformer.num_tokens)

    def forward(self, inp, repr=False, pad_mask=None):
        # (batch_sz, seq_len)
        if self.training and self.config.causal:
            inp = torch.cat([torch.ones_like(inp[:,0:1])*MASK_IDX, inp[:,:-1]], dim=1)
        enc_out = self.embed(inp) * math.sqrt(self.embed_dim)
        # (batch_sz, seq_len, embed_dim)
        if pad_mask is None:
            pad_mask = (inp == PAD_IDX).to(inp).bool()
        enc_out = self.pos_encoder(enc_out, pad_mask)
        if self.config.causal:
            mask = generate_square_subsequent_mask(enc_out.shape[1])
        else:
            mask = None # bidirectional
        enc_out = self.enc(enc_out, src_key_padding_mask=pad_mask, mask=mask)
        if repr:
            return enc_out
        enc_out = self.fc(enc_out)
        return enc_out

class Decoder(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        
        self.encoder     = encoder
        self.config      = config 
        self.train_enc   = config.train_enc
        self.embed_dim   = config.transformer.embed_dim

        self.embed       = nn.Embedding(config.transformer.num_tokens, config.transformer.embed_dim, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(config.transformer.embed_dim, config.transformer.dropout, max_len=config.transformer.seq_len)
        dec_layer        = nn.TransformerDecoderLayer(config.transformer.embed_dim, config.transformer.nhead, config.transformer.feedforward, config.transformer.dropout, batch_first=True, activation="gelu")
        self.dec         = nn.TransformerDecoder(dec_layer, config.transformer.layers, norm=nn.LayerNorm(config.transformer.embed_dim))
        self.fc          = nn.Linear(config.transformer.embed_dim, config.transformer.num_tokens)

    def forward(self, inp, dec_out=None,repr=None,shift=True):
        if shift:
            dec_in  = torch.cat([torch.ones_like(dec_out[:,0:1])*MASK_IDX, dec_out[:,:-1]], dim=1)
        
        else:
            dec_in=dec_out
        enc_pad=(inp == PAD_IDX).to(inp).bool()
        dec_pad=(dec_in == PAD_IDX).to(inp).bool()
        
        if repr is None:
            repr = self.encoder(inp, repr=True, pad_mask=enc_pad)

        dec_out = self.embed(dec_in) * math.sqrt(self.embed_dim)
        # (batch_sz, seq_len, embed_dim)
        dec_out = self.pos_encoder(dec_out, pad_mask=dec_pad)
        dec_out = self.dec(dec_out,
                            repr,
                            tgt_mask=generate_square_subsequent_mask(dec_out.size(1)).to(dec_out.device),
                            memory_key_padding_mask=enc_pad,
                            tgt_key_padding_mask=dec_pad
        )
        dec_out = self.fc(dec_out)
        return dec_out
    
    def generate_sequence(self, inp_path, out_path, device='cpu'):
        enc_inp = mid2dat_anna(inp_path)[:self.config.transformer.seq_len]
        enc_inp = torch.tensor(list(map(key_mapping, enc_inp))).unsqueeze(0).to(device)
        repr = self.encoder(enc_inp, repr=True)
        mem  = self.mem_fn(repr)
        dec_inp = torch.tensor([[MASK_IDX]]).to(device)
        for i in range(self.config.transformer.seq_len):
            out_logits = self.forward(enc_inp, dec_inp, mem)[0,-1,:]
            new_tok = torch.multinomial(torch.nn.functional.softmax(out_logits, dim=0), num_samples=1)
            if new_tok[0] == 389:
                break
            dec_inp = torch.cat((dec_inp, new_tok.unsqueeze(0)), dim=-1)
        dat2mid_anna(list(map(tok_mapping, dec_inp[0].cpu())), fname=out_path)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term) # All even indices of the embedding dim.
        pe[:, 1::2] = torch.cos(position * div_term) # All odd indices of the embedding dim.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, pad_mask=None):
        x = x + self.pe[:, :x.size(1)]
        if pad_mask is not None:
            x[pad_mask] = 0
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


    