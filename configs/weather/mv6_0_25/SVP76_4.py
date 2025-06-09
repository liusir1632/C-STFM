method = 'SVP76'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 264
N_T = 8
N_S = 3

# training
lr = 5e-3
batch_size = 32 # bs16 = 4gpus x bs4
drop_path = 0.4
sched = 'cosine'
warmup_epoch = 0