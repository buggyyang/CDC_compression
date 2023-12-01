# training config
n_step = 1000000
scheduler_checkpoint_step = 100000
log_checkpoint_step = 5000
lr = 4e-5
decay = 0.9
minf = 0.5
optimizer = "adam"  # adamw or adam
n_workers = 4

# load
load_model = True
load_step = True

# data config
data_config = {
    "dataset_name": "vimeo",
    "data_path": "/extra/ucibdl0/shared/data",
    "sequence_length": 1,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}

batch_size = 4
result_root = "/extra/ucibdl0/ruihan/params_compress_v7"
tensorboard_root = "/extra/ucibdl0/ruihan/tblogs_compress_v7"
