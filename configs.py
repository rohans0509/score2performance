import ml_collections

def backbone_config():
    config = ml_collections.ConfigDict()

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 16
    train.lr = 1e-4
    train.warmup = 4000
    train.iterations = 100000
    train.print_every = 1

    config.eval = eval = ml_collections.ConfigDict()
    eval.generate = False
    eval.inp_files = ("inp.mid",)

    config.transformer = transformer = ml_collections.ConfigDict()
    transformer.num_tokens = 392
    transformer.embed_dim = 512
    transformer.nhead = 8
    transformer.feedforward = 2048
    transformer.layers = 6 
    transformer.seq_len = 512
    transformer.dropout = 0.1

    return config 

def autoencoder_config():
    config = backbone_config()
    config.type = "autoencoder"

    config.bottleneck = bottleneck = ml_collections.ConfigDict()
    bottleneck.pooling_method = "moments"
    bottleneck.pooling_args = {"num_moments":2, "train_power":False}
    
    config.pt_enc_path = "transformer_enc.pt"
    config.dec_path = "transformer_dec.pt"
    config.train_enc = True 
    config.mask_probs = (0., 0., 0.)
    config.causal = False
    config.drop_moment = False 
    
    return config

def encoder_config():
    config = backbone_config()
    config.type = "encoder"

    config.enc_path = "transformer_enc.pt"
    config.mask_probs = (0.12, 0.015, 0.015)
    config.causal = False

    return config

def get_config(type):
    if type=="autoencoder":
        return autoencoder_config()
    elif type=="encoder":
        return encoder_config()