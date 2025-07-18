import torch

epochs = 400
batch_size = 64
learning_rate = 1e-4
T_timestep = 1000

save_every = 200
sample_every = 100

image_size = 96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")