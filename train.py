import os
import sys
import argparse
import random 
import glob
import pickle 
import time

import matplotlib.pyplot as plt
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

import torchaudio
import torch.fft
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import get_dataloader
import utils
import modules
from objective import objective

try:
    import wandb
except:
    pass

def train(model, optim_INR, optim_mapping, scheduler, train_loader, config):
    utils.init_seed(rand_seed=False)
    loss_f = objective(config.device, derivative=config.deriv_per_sample, cdpam=config.cdpam, double=config.double)
    model.train()
    train_loss_avg = []
    print("Seeing ", torch.cuda.device_count(), "GPUs")
    if torch.cuda.device_count() > 1 and config.use_multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs!")
        if not config.architecture == "wavegan":
            model = nn.DataParallel(model)
    if config["meta_architecture"] == "autodecoder":
        all_z = utils.init_latent(config["dataset_size"], 
                                    config["num_latent"],
                                    config["device"],
                                    std=config["latent_init_std"])
        if config.double:
            all_z = all_z.double()
    try:
        print(f"Starting run for {config.num_epochs} epochs..")
        for epoch in range(config.num_epochs):
            if config.prog_weight_decay_every:
                if not epoch % config.prog_weight_decay_every:
                    weight_decay_override = max(0, config.weight_decay / (config.prog_weight_decay_factor**(epoch / config.prog_weight_decay_every)))
                    optim_INR.weight_decay = config.weight_decay / (config.prog_weight_decay_factor**(epoch / config.prog_weight_decay_every))
                    optim_INR.param_groups[0]['weight_decay'] = weight_decay_override
                    print(f"decreased weight decay to {weight_decay_override}")
            if config.architecture == "pi-gan_prog":
                model.step()

            train_loss_avg.append(0)
            processed_batches = 0

            for x, idx in train_loader:
                starttime = time.time()
                x = x.to(config["device"]).unsqueeze(-1)
                if config.double:
                    x = x.double()
                sampled_coords, indices = utils.sample_coords(x.shape[0], config["device"],
                                                            num_samples=config.samples_per_datapoint,
                                                            full_coord=False, 
                                                            sample_even=True, 
                                                            ratio=1)    
                if config.double:
                    sampled_coords = sampled_coords.double()
                
                if config["meta_architecture"] == "autodecoder":
                    z = all_z[idx, :, :].clone().requires_grad_()
                    for grad_step in range(config["latent_descent_steps"]):
                        # Get reconstructions              
                        g = model(sampled_coords, z=z)
                        L = loss_f( g.squeeze(-1),
                                    x.squeeze(-1).gather(1, indices), 
                                    per_sample=config.per_sample, 
                                    deriv_per_sample=config.deriv_per_sample,
                                    cdpam = config.cdpam,
                                    multiscale_STFT=config.multiscale_STFT)
                        # Update latents
                        z = z-torch.autograd.grad(L, [z], create_graph=True, retain_graph=True)[0] * (config.latent_lr/(grad_step+1))

                if config["meta_architecture"] == "autoencoder":
                    g, z = model(x.transpose(1,2), sampled_coords)
                    L = loss_f( g.squeeze(-1),
                                x.squeeze(-1).gather(1, indices), 
                                per_sample=config.per_sample, 
                                deriv_per_sample=config.deriv_per_sample,
                                cdpam = config.cdpam,
                                multiscale_sfft=config.multiscale_STFT)

                scheduler.step(L)
                optim_INR.zero_grad()
                if optim_mapping:
                    optim_mapping.zero_grad()
                L.backward() 
                optim_INR.step()
                if optim_mapping:
                    optim_mapping.step()

                if L.isnan():
                    print("loss is nan, exiting")
                    return 1
                if L.isinf():
                    print("loss is inf, exiting")
                    return 1

                # Track latent
                if config["meta_architecture"] == "autodecoder":
                    for batch_i, all_i in enumerate(idx):
                        all_z[all_i, :] = z[batch_i, :].detach().clone()
                
                train_loss_avg[-1] += L.item()
                processed_batches += 1

            train_loss_avg[-1] /= processed_batches
            print(f"Epoch {epoch}  Loss: {train_loss_avg[-1]:.3f}, {len(indices.view(-1))/(time.time()-starttime)} samples/sec")

            # Evaluation ========================================== 
            if not epoch%config.eval_every:
                with torch.no_grad():
                    # gather full resolution reconstructions if needed
                    if config.save_audio_plots or (config.save_audio and epoch >= (config.num_epochs-config.eval_every)):
                        g = [] 
                        sampled_coords, indices = utils.sample_coords(config.batch_size, config["device"],
                                                            num_samples=config.audio_length,
                                                            full_coord=True,
                                                            sample_even=True, 
                                                            ratio=config.eval_upscale_ratio)  
                        splits = config.dataset_size // config.max_high_res_batch_size
                        for i in range(splits):
                            if config["meta_architecture"] == "autodecoder":
                                g_temp = model(sampled_coords[config.max_high_res_batch_size*i:config.max_high_res_batch_size*(1+i), :, :],
                                z=all_z[config.max_high_res_batch_size*i:config.max_high_res_batch_size*(1+i), :, :])
                            if config["meta_architecture"] == "autoencoder":
                                g_temp, z_temp = model(x.transpose(1,2)[config.max_high_res_batch_size*i:config.max_high_res_batch_size*(1+i), :, :],
                                                        sampled_coords[config.max_high_res_batch_size*i:config.max_high_res_batch_size*(1+i), :, :])
                            g.extend(g_temp)
                            if i*config.max_high_res_batch_size >= config.eval_samples:
                                break
                        g = torch.stack(g, dim=0)

                    if config.save_audio_plots:
                        for i, wave in enumerate(g):
                            wave_orig_npy = x[i].squeeze(0).squeeze(-1).cpu().numpy()
                            wave_gen_npy = g[i].squeeze(0).squeeze(0).cpu().detach().numpy()  
                            fig, axs = plt.subplots(1)
                            axs.plot(wave_orig_npy[1000:1200])
                            axs.plot(wave_gen_npy[1000:1200])
                            if config.wandb:
                                wandb.log({f"Audio {i} And reconstructions epoch {epoch}": wandb.Image(plt)})
                            else:
                                os.makedirs(f"{config.save_path}/plots", exist_ok=True)
                                plt.savefig(f"{config.save_path}/plots/reconstruction_{i}_epoch_{epoch}.png")
                        plt.close(fig="all")

                    if config.save_latents:
                        os.makedirs(f"{config.save_path}/latents", exist_ok=True)
                        if config.meta_architecture == "autoencoder":
                            z = z.detach().cpu().numpy()
                            np.save(f"{config.save_path}/latents/z_batch_{epoch}.npy", z)
                        else:
                            z = z.detach().cpu().numpy()
                            all_z_np = all_z.detach().cpu().numpy()
                            np.save(f"{config.save_path}/latents/z_batch_{epoch}.npy", z)
                            np.save(f"{config.save_path}/latents/z_all_{epoch}.npy", all_z_np)
                            
                    # save generated audio if in final evaluation before end of training 
                    if config.save_audio and epoch >= (config.num_epochs-config.eval_every):
                        os.makedirs(f"{config.save_path}/audio", exist_ok=True)
                        utils.saveAudioBatch(g, f"{config.save_path}/audio", 
                            f"reconstruction_epoch_{epoch}",
                            sr=16000*config.eval_upscale_ratio, overwrite=True)
                        
                        os.makedirs(f"{config.save_path}/audio", exist_ok=True)
                        utils.saveAudioBatch(x, f"{config.save_path}/audio", 
                            "original",
                            sr=16000*config.eval_upscale_ratio, overwrite=False)
                            
                    # save model if in final evaluation before end of training 
                    if config["save_model"] and epoch >= (config.num_epochs-config.eval_every):
                        os.makedirs(f"{config.save_path}/checkpoint", exist_ok=True)
                        torch.save(model, f"{config.save_path}/checkpoint/model_{epoch}.pt")
                    
                    torch.cuda.empty_cache()
            
                if config.wandb:
                    wandb.log({"loss": train_loss_avg[-1]})

    except KeyboardInterrupt:
        torch.save(model, f"{args.save_path}/model_final.pth")
        if config["track_latent"]:
            with open(f"{args.save_path}/all_z.pth", "wb") as fh:
                pickle.dump(all_z, fh)
        if config.wandb:
            run.finish()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--wandb", type=int, default = 0, help="Enable wandb logging")
    parser.add_argument("--wandb_project_name", type=str, default = "default", help="Name of wandb project")

    # === TRAINING ============================================
    parser.add_argument('--dataset_name', type=str, default = 'SPEECHCOMMANDS', help="Which dataset to train on.", choices=["NSYNTH.diverse_baseline", "NSYNTH.keyboard_baseline", "SPEECHCOMMANDS", "custom"]) 
    parser.add_argument('--dataset_size', type=int, default = 128, help="Number of samples to train on. Maximum is 1024 in given datasets") 
    parser.add_argument('--audio_length', type=int, default = 16000, help="Audio length")
    parser.add_argument('--autoconfig', type=int, default = 0, help="Enable autoconfig. Overrides omega_0 values depending on dataset and architecture for tested setups.")
    
    parser.add_argument('--lr', type=float, default = 1e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default = 32, help="Batch_size")
    parser.add_argument('--num_epochs', type=int, default = 5001, help="Number of epochs")
    parser.add_argument('--use_gpu', type=int, default = 1, help="Enable GPU")
    parser.add_argument('--use_multi_gpu', type=int, default = 0, help="Enable multiple GPUs")

    # === MODEL Decoder invariant params ============================================
    parser.add_argument('--architecture', type=str, default = "pi-gan", help="What architecture to use as the decoder.", choices=["wavegan", "im-net", "pi-gan", "pi-gan_prog", "pi-gan_sine_first", "pi-gan_sine_last", "pi-gan_relu", "pi-gan_concat_middle", "pi-gan_concat_all", "pi-gan_min_mapping", "pi-gan_five_mapping", "pi-gan_shrinking", "pi-gan_deep", "pi-gan_wide"]) 
    parser.add_argument('--meta_architecture', type=str, default = "autodecoder", help="What latent embedding inference method to use.", choices=["autoencoder", "autodecoder"]) 
    parser.add_argument('--num_latent', type=int, default = 256, help="Number of latent dimensions")
    parser.add_argument('--double', type=int, default = 0, help="Enable double precision throughout training")

    # === MODEL implicit Decoder params ============================================
    parser.add_argument('--weight_norm', type=int, default = 0, help="Enable weight normalization")
    parser.add_argument('--first_omega_0', type=int, default = 3000, help="First layer input scaling for sinusoidal architectures")
    parser.add_argument('--hidden_omega_0', type=int, default = 30, help="Hidden layer input scaling for sinusoidal architectures")
    parser.add_argument('--coord_multi', type=int, default = 1, help="Input scaling for any architecture")
    
    # === Autodecoder ============================================
    parser.add_argument('--latent_init_std', type=float, default = 0.001, help="Latent embedding initialization std")
    parser.add_argument('--latent_descent_steps', type=int, default = 1, help="Number of gradient descent steps per iteration for latent embedding optimization")
    parser.add_argument('--latent_lr', type=int, default = 0.3, help="Learning rate for latent optimization.")

    # === SAMPLING ============================================
    parser.add_argument('--samples_per_datapoint', type=int, default = 8000, help="Number of samples per wave") 
    parser.add_argument('--sample_even', type=int, default = 1, help="Sample coordinates with equal spacing.")              
    
    # === LOSS ============================================
    parser.add_argument('--per_sample', type=int, default=1, help="MSE per sample multiplier for objective function.")    
    parser.add_argument('--deriv_per_sample', type=int, default = 0, help="MSE per sample of derivative of functions multiplier for objective function.")
    parser.add_argument('--cdpam', type=int, default = 0, help="CDPAM multiplier for objective function")
    parser.add_argument('--multiscale_STFT', type=int, default = 0, help="Multi STFT multiplier for objective function")
    parser.add_argument('--weight_decay', type=float, default = 0, help="L2 weight decay amount.")
    
    # === Evaluation ============================================
    parser.add_argument('--eval_every', type=int, default = 500, help="Evaluate every n iterations")
    parser.add_argument('--save_audio_plots', type=int, default = 1, help="Save audio plots at every evaluation")
    parser.add_argument('--save_latents', type=int, default = 1, help="Save latent embeddings")
    parser.add_argument('--save_audio', type=int, default = 1, help="Save generated audio at end of training.")
    parser.add_argument('--save_model', type=int, default = 1, help="Save model at end of training.")
    parser.add_argument('--eval_samples', type=int, default = 1, help="Number of samples to evaluate on.")
    parser.add_argument('--eval_upscale_ratio', type=int, default = 1, help="Upscale ratio for evaluation of generations.")
    parser.add_argument('--save_path', type=str, default = "auto", help="Path to save output. 'auto' creates directories based on setup.") 
    parser.add_argument('--max_high_res_batch_size', type=int, default = 16, help="Maximum batch size for high resolution evaluations.")

    parser.add_argument('--note_general', type=str, default = "default", help="Note to add to general output directory name")
    parser.add_argument('--note', type=str, default = "default", help="Note to filter wandb results.")

    # === pi-gan PROG ============================================
    parser.add_argument('--num_groups', type=int, default = 0, help="Number of groups to use for progressive activation scaling in pi-gan_prog")
    parser.add_argument('--prog_weight_decay_factor', type=float, default = 0, help="Weight decay reduction factor for progressive weight decay.")
    parser.add_argument('--prog_weight_decay_every', type=int, default = 0, help="Number of iterations after which to reduce weight decay.")

    args = parser.parse_args()
    
    if args.autoconfig:
        print("#"*10, "Warning: args autoconfig is modifying args!!!", "#"*10)
        if args.architecture == "wavegan":
            args.lr = 1e-4
            args.num_latent = 256
        if args.architecture.startswith("pi-gan"):
            args.lr = 1e-5
            if args.dataset_name == "SPEECHCOMMANDS":
                args.first_omega_0 = 615
                args.hidden_omega_0 = 200
            else:
                args.first_omega_0 = 1760
                args.hidden_omega_0 = 245
            if args.architecture == "pi-gan_prog":
                args.first_omega_0 = 10000
                args.hidden_omega_0 = 300

    if args.architecture == "wavegan":
        args.samples_per_datapoint = 16000
    else:
        args.input_dim = 1
        args.output_dim = 1

    train_loader = get_dataloader(args.dataset_name, args.dataset_size, args.batch_size)
    
    if args.save_path == "auto":
        args.save_path = f"{args.note}/{args.dataset_name}/{args.architecture}/{args.meta_architecture}"
    args.save_path = f"results/{args.save_path}"
    
    config_dict = utils.AttrDict()
    config_dict.update(vars(args))
    pp.pprint(config_dict)

    if args.wandb:
        experiment_name = wandb.util.generate_id()
        run = wandb.init(project=args.wandb_project_name, 
                group=experiment_name)
        config = wandb.config
        config.update(args) # adds all of the arguments as config variables
    else:
        config = config_dict

    config.device = torch.device('cuda' if (torch.cuda.is_available() and config.use_gpu) else 'cpu') 
    decoder = modules.get_decoder(config).to(config.device)

    if config.meta_architecture == "autoencoder":
        encoder = modules.ConvEncoder(config.num_latent)
        model = modules.Autoencoder(encoder, decoder, 
                                            audio_length=config.audio_length, device=config.device).to(config.device)
    else:
        model = decoder

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)
    optim_INR, optim_mapping = utils.get_optim(config, model)
    batch_per_epoch = args.dataset_size // args.batch_size
    epoch_patience = 100
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_INR, 'min', patience=epoch_patience * batch_per_epoch,
                                factor=0.95, verbose=True, threshold=0.0001, 
                                threshold_mode='rel', cooldown=1)
    
    if config.wandb:
        wandb.watch(model)

    if config.double:
        model = model.double()

    train(model, optim_INR, optim_mapping, scheduler, train_loader, config)
    print("finished training")
