from adabelief_pytorch import AdaBelief
import random 
import torch
from scipy.io import wavfile
import numpy as np
import os
import librosa

def get_optim(config, model):
    optim_INR = AdaBelief(model.net.parameters(), lr=config.lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = 0, weight_decay = config.weight_decay, rectify = False)
    if hasattr(model, 'mapping_net'):
        optim_mapping = AdaBelief(model.mapping_net.parameters(), lr=config.lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = 0, weight_decay = 0, rectify = False)
    else:
        optim_mapping = None
    return optim_INR, optim_mapping

def saveAudioBatch(data, path, basename, sr=16000, overwrite=False):
    from librosa.util.utils import ParameterError
    os.makedirs(path, exist_ok=True)
    if len(data.shape) != 3:
        data.unsqueeze(0)
    try:
        for i, audio in enumerate(data):

            if type(audio) != np.ndarray:
                audio = np.array(audio.cpu(), float)

            out_path = os.path.join(path, f'{basename}_{i}.wav')
            
            if not os.path.exists(out_path) or overwrite:
                wavfile.write(out_path, sr, audio)
            else:
                print(f"saveAudioBatch: File {out_path} exists. Skipping...")
                continue
    except ParameterError as pe:
        print(pe)

def calc_rfft_mag(samples, samplerate=16000):
    samples = samples.squeeze(-1) 
    N = samples.shape[0]

    rfft = torch.fft.rfft(samples)[:samplerate//2]
    freqs = pytorch_fftfreq(N, 1/samplerate)[:samplerate//2]
    return rfft, freqs

def sample_coords(batch_size, device,
                num_samples,
                full_coord=False, 
                sample_even=True, 
                samplerate=16000,
                ratio=1):

        coords = torch.arange(-1,1, (2/samplerate)/ratio, requires_grad=False).to(device)

        if full_coord:
            num_samples = samplerate*ratio
            indices = torch.arange(0,num_samples).to(device).repeat(batch_size, 1)
            sampled_coords = coords.repeat(batch_size, 1)
        else:
            if sample_even:
                step_size = samplerate // num_samples

                indices_list = []
                for i in range(batch_size): 
                    phase = int(torch.randint(step_size, (1,)))
                    indices = torch.arange(phase,samplerate, step_size).to(device)
                    indices_list.append(indices)
                
                indices = torch.stack(indices_list, dim=0)
                sampled_coords = coords[indices]

            else:
                indices = torch.multinomial(torch.ones(batch_size, samplerate).to(device), num_samples, replacement=False)
                sampled_coords = coords[indices]

        return sampled_coords.unsqueeze(-1), indices

def init_latent(dataset_size, latent_size, device, std=0):
    all_z = []
    for i in range(dataset_size):
        all_z.append(torch.randn(latent_size, requires_grad=False).unsqueeze(0)*std)
    all_z = torch.stack(all_z, dim=0).to(device)
    return all_z

def init_seed(rand_seed=False):
    if rand_seed:
        seed = random.randint(0, 9999)
    else:
        seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("Random Seed: ", seed)
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self