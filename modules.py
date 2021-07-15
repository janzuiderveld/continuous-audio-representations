import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from INR_collection import *

#=== Decoders =================================================================
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
            reflection_padding = kernel_size // 2
            self.reflection_pad = nn.ConstantPad1d(reflection_padding, value = 0)
            self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv1d(out)
        return out

class WaveGANGenerator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, latent_dim=256, audio_length=16000,
                 post_proc_filt_len=512, verbose=False, upsample=True):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size # d
        self.num_channels = num_channels # c
        self.latent_dim = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.audio_length=16000
        self.verbose = verbose

        self.fc1 = nn.DataParallel(nn.Linear(latent_dim, 256 * model_size))
        
        self.tconv1 = None
        self.tconv2 = None
        self.tconv3 = None
        self.tconv4 = None
        self.tconv5 = None
           
        self.upSampConv1 = None
        self.upSampConv2 = None
        self.upSampConv3 = None
        self.upSampConv4 = None
        self.upSampConv5 = None
        
        self.upsample = upsample
    
        if self.upsample:
            self.upSampConv1 = nn.DataParallel(
                UpsampleConvLayer(16 * model_size, 8 * model_size, 25, stride=1, upsample=4))
            self.upSampConv2 = nn.DataParallel(
                UpsampleConvLayer(8 * model_size, 4 * model_size, 25, stride=1, upsample=4))
            self.upSampConv3 = nn.DataParallel(
                UpsampleConvLayer(4 * model_size, 2 * model_size, 25, stride=1, upsample=4))
            self.upSampConv4 = nn.DataParallel(
                UpsampleConvLayer(2 * model_size, model_size, 25, stride=1, upsample=4))
            self.upSampConv5 = nn.DataParallel(
                UpsampleConvLayer(model_size, num_channels, 25, stride=1, upsample=4))
            
        else:
            self.tconv1 = nn.DataParallel(
                nn.ConvTranspose1d(16 * model_size, 8 * model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv2 = nn.DataParallel(
                nn.ConvTranspose1d(8 * model_size, 4 * model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv3 = nn.DataParallel(
                nn.ConvTranspose1d(4 * model_size, 2 * model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv4 = nn.DataParallel(
                nn.ConvTranspose1d(2 * model_size, model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv5 = nn.DataParallel(
                nn.ConvTranspose1d(model_size, num_channels, 25, stride=4, padding=11,
                                   output_padding=1))

        if post_proc_filt_len:
            self.ppfilter1 = nn.DataParallel(nn.Conv1d(num_channels, num_channels, post_proc_filt_len))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, coordinates, z):
        assert coordinates.shape[1] == self.audio_length, \
                f"received coordinate amount of {coordinates.shape[1]} in WaveGAN generator, need {self.audio_length}"
        z = z[:,0,:]
        x = self.fc1(z).view(-1, 16 * self.model_size, 16)
        x = F.relu(x)
        output = None
        if self.verbose:
            print(x.shape)

        if self.upsample:
            x = F.relu(self.upSampConv1(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv2(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv3(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv4(x))
            if self.verbose:
                print(x.shape)

            output = F.tanh(self.upSampConv5(x))
        else:
            x = F.relu(self.tconv1(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.tconv2(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.tconv3(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.tconv4(x))
            if self.verbose:
                print(x.shape)

            output = F.tanh(self.tconv5(x))
            
        if self.verbose:
            print(output.shape)

        if self.post_proc_filt_len:
            # Pad for "same" filtering
            if (self.post_proc_filt_len % 2) == 0:
                pad_left = self.post_proc_filt_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filt_len - 1) // 2
                pad_right = pad_left
            output = self.ppfilter1(F.pad(output, (pad_left, pad_right)))
            if self.verbose:
                print(output.shape)

        output = output[:, :, :self.audio_length].transpose(1,2)
        return output

#=== Encoders =================================================================
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, first_filter_size=320):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, first_filter_size, 4)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(14)
        self.fc_mu = nn.Linear(512, latent_dim)

        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) 
        x_mu = self.fc_mu(x)
        return x_mu

#=== Latent inference =================================================================
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, audio_length=16000, device='cuda'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.audio_length = audio_length
        self.device = device
    
    def forward(self, x, coordinates):
        z = self.encoder(x)
        x_recon = self.decoder(coordinates, z)
        return x_recon, z
        
def get_decoder(config):
    if config.architecture == "wavegan":
        return WaveGANGenerator(model_size=8, ngpus=4, num_channels=1, 
                                latent_dim=config.num_latent, audio_length=16000,
                                post_proc_filt_len=512)
    
    if config.architecture == "im-net":
        return IMNET(config.input_dim, config.output_dim, bias=True, coord_multi=config.coord_multi,
                num_layers=5, initial_hidden=1024, z_size=config.num_latent, 
                init_method={"weights": 'basic', "bias": "zero"})

    if config.architecture.startswith("pi-gan"):
        arch = config.architecture
        if arch == "pi-gan":
            return piGAN(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0)
        if arch == "pi-gan_prog":
            return piGAN_prog(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        total_epochs=config.num_epochs, num_groups=config.num_groups)
        if arch == "pi-gan_sine_first":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "relu", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_sine_last":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["relu", "relu", "sine"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_relu":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["relu", "relu", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_concat_middle":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "concat", # "concat", "film", "both"
                        conditioning_location = "middle", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_concat_all":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "concat", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
                        
        if arch == "pi-gan_min_mapping":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=0, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_five_mapping":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=5, num_INR_layers=8, num_hidden_mapping=256,
                        num_hidden_INR=256, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_shrinking":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=4, num_hidden_mapping=256,
                        num_hidden_INR=1024, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "shrinking", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_deep":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=12, num_hidden_mapping=256,
                        num_hidden_INR=210, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )
        if arch == "pi-gan_wide":
            return piGAN_custom(config.input_dim, config.output_dim, bias=True,
                        num_mapping_layers=3, num_INR_layers=4, num_hidden_mapping=256,
                        num_hidden_INR=365, z_size=config.num_latent, 
                        first_omega_0=config.first_omega_0, hidden_omega_0=config.hidden_omega_0,
                        activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                        conditioning_method = "film", # "concat", "film", "both"
                        conditioning_location = "all", # "all, middle"
                        network_shape= "consistent", # "consistent", "shrinking"
                        )

