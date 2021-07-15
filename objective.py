import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio 

class objective(nn.Module):
    def __init__(self, device, derivative=False, cdpam=False, double=False):
        super(objective, self).__init__()
        if derivative:
            self.finite_diff_derivative = torch.range(-1,1,2).unsqueeze(0).unsqueeze(0).to(device)
            if double:
                self.finite_diff_derivative = self.finite_diff_derivative.double()
        if cdpam:
            import cdpam
            self.CDPAM = cdpam.CDPAM()
        self.device=device
    
    def get_multiscale_stft(self, signal, scales=[4096, 2048, 1024, 512, 256, 128], overlap=.75):
        stfts = []
        for s in scales:
            S = torch.stft(
                signal,
                s,
                int(s * (1 - overlap)),
                s,
                torch.hann_window(s).to(signal),
                True,
                normalized=True,
                return_complex=True
            )
            stfts.append(S)
        return stfts

    def forward(self, recon_x, x,
                per_sample=0, 
                deriv_per_sample=0,
                cdpam = 0,
                multiscale_STFT=0,
                ):

        num_coords = recon_x.shape[1]

        recon_loss = 0
        if per_sample:
            recon_loss += ((recon_x.squeeze(-1) - x.squeeze(-1))**2).sum(1).mean() * per_sample
        if deriv_per_sample:
            recon_x_deriv = F.conv1d(recon_x.view(-1, 1, num_coords), self.finite_diff_derivative)
            x_deriv = F.conv1d(x.view(-1, 1, num_coords), self.finite_diff_derivative)
            recon_loss += ((recon_x_deriv.squeeze(-1) - x_deriv.squeeze(-1))**2).sum(1).mean() * deriv_per_sample
        if cdpam:
            resampler = torchaudio.transforms.Resample(orig_freq=num_coords, new_freq=22050)
            x_resample = resampler(x).squeeze(1)
            x_recon_resample = resampler(recon_x).squeeze(1)
            dist = self.CDPAM.forward(wav_in=x_resample, wav_out=x_recon_resample).sum()
            recon_loss += dist * cdpam
        if multiscale_STFT:
            orig_stft = self.get_multiscale_stft(x.squeeze(0))
            recon_stft = self.get_multiscale_stft(recon_x.squeeze(0))
            multi_stft_loss = 0
            for s_x, s_y in zip(orig_stft, recon_stft):
                lin_loss = (s_x.abs() - s_y.abs()).abs().mean()
                multi_stft_loss += lin_loss 
            recon_loss += multi_stft_loss * multiscale_STFT

        return recon_loss

