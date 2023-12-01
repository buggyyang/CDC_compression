# training config
n_step = 1000000
scheduler_checkpoint_step = 100000
log_checkpoint_step = 5000
gradient_accumulate_every = 1
lr = 4e-5
decay = 0.9
minf = 0.5
optimizer = "adam"  # adamw or adam
n_workers = 4

# load
load_model = True
load_step = True

# diffusion config
pred_mode = 'noise'
loss_type = "l1"
iteration_step = 20000
sample_steps = 500
embed_dim = 64
dim_mults = (1, 2, 3, 4, 5, 6)
hyper_dim_mults = (4, 4, 4)
context_channels = 3
clip_noise = "none"
val_num_of_batch = 1
additional_note = ""
vbr = False
context_dim_mults = (1, 2, 3, 4)
sample_mode = "ddim"
var_schedule = "linear"
aux_loss_type = "lpips"
compressor = "big"

# data config
data_config = {
    "dataset_name": "vimeo",
    "data_path": "*",
    "sequence_length": 1,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}

batch_size = 4

result_root = "*"
tensorboard_root = "*"
