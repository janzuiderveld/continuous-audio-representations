# Continuous audio representations
PyTorch implementation for <a href="https://www.linkedin.com/in/jan-zuiderveld/detail/overlay-view/urn:li:fsd_profileTreasuryMedia:(ACoAACpnYc4Beb_aC_BHJocToa2u1xbuJjtxXvc,1635463815861)/"> learning to represent audio in a distribution of continuous functions </a> using implicit neural representations a.k.a. INRs, see <a href="https://github.com/janzuiderveld/INR-collection">INR-collection</a> for more details on the used implicit implementations. 

Supports training on subsets of the Speech Commands and NSYNTH datasets out of the box. Other datasets need customizing. Several implicit decoder architectures, ablations and latent embedding inference methods are implemented.

# Quickstart

```
git clone https://github.com/janzuiderveld/continuous-audio-representations
cd continuous-audio-representations

pip3 install -r requirements.txt
```

Running `Train.py` automatically downloads datasets and set them in place.

# Usage:

The Following architectures are readily available (these can be trained by supplying these tags as `--architecture` argument for `Train.py`): 
- wavegan
- im-net
- pi-gan
  - pi-gan_prog
  - pi-gan_sine_first
  - pi-gan_sine_last
  - pi-gan_relu
  - pi-gan_concat_middle
  - pi-gan_concat_all
  - pi-gan_min_mapping
  - pi-gan_five_mapping
  - pi-gan_shrinking
  - pi-gan_deep
  - pi-gan_wide

Latent embedding inference methdods (`--meta_architecture`):
- autoencoder
- autodecoder

For datasets, the following are automatically downloaded when supplied as `--dataset_name`:
- SPEECHCOMMANDS
- NSYNTH.diverse_baseline
- NSYNTH.keyboard_baseline

Extensive list of parameters:

|long|default|help|
| :--- | :--- | :--- |
|`--help`||show this help message and exit|
|`--wandb`|`0`|Enable wandb logging|
|`--wandb_project_name`|`default`|Name of wandb project|
|`--dataset_name`|`SPEECHCOMMANDS`|Which dataset to train on.|
|`--dataset_size`|`128`|Number of samples to train on. Maximum is 1024 in given datasets|
|`--audio_length`|`16000`|Audio length|
|`--autoconfig`|`0`|Enable autoconfig. Overrides omega_0 values depending on dataset and architecture for tested setups.|
|`--lr`|`1e-05`|Learning rate|
|`--batch_size`|`32`|Batch_size|
|`--num_epochs`|`5001`|Number of epochs|
|`--use_gpu`|`1`|Enable GPU|
|`--use_multi_gpu`|`0`|Enable multiple GPUs|
|`--architecture`|`pi-gan`|What architecture to use as the decoder.|
|`--meta_architecture`|`autodecoder`|What latent embedding inference method to use.|
|`--num_latent`|`256`|Number of latent dimensions|
|`--double`|`0`|Enable double precision throughout training|
|`--weight_norm`|`0`|Enable weight norm|
|`--first_omega_0`|`155`|First layer input scaling for sinusoidal architectures|
|`--hidden_omega_0`|`390`|Hidden layer input scaling for sinusoidal architectures|
|`--coord_multi`|`1`|Input scaling for any architecture|
|`--latent_init_std`|`0.001`|Latent embedding initialization std|
|`--latent_descent_steps`|`1`|Number of gradient descent steps per iteration for latent embedding optimization|
|`--latent_lr`|`0.3`|Learning rate for latent optimization.|
|`--samples_per_datapoint`|`8000`|Number of samples per wave|
|`--sample_even`|`1`|Sample coordinates with equal spacing.|
|`--per_sample`|`1`|MSE per sample multiplier for objective function.|
|`--deriv_per_sample`|`0`|MSE per sample of derivative of functions multiplier for objective function.|
|`--cdpam`|`1`|CDPAM multiplier for objective function|
|`--multiscale_STFT`|`1`|Multi STFT multiplier for objective function|
|`--weight_decay`|`1`|L2 weight decay amount.|
|`--eval_every`|`500`|Evaluate every n iterations|
|`--save_audio_plots`|`1`|Save audio plots at every evaluation|
|`--save_latents`|`1`|Save latent embeddings|
|`--save_audio`|`1`|Save generated audio at end of training.|
|`--save_model`|`1`|Save model at end of training.|
|`--eval_samples`|`1`|Number of samples to evaluate on.|
|`--eval_upscale_ratio`|`1`|Upscale ratio for evaluation of generations.|
|`--save_path`|`auto`|Path to save output. 'auto' creates directories based on setup.|
|`--max_high_res_batch_size`|`16`|Maximum batch size for high resolution evaluations.|
|`--note_general`|`default`|Note to add to general output directory name|
|`--note`|`default`|Note to filter wandb results.|
|`--num_groups`|`0`|Number of groups to use for progressive activation scaling in pi-gan_prog|
|`--prog_weight_decay_factor`|`0`|Weight decay reduction factor for progressive weight decay.|
|`--prog_weight_decay_every`|`0`|Number of iterations after which to reduce weight decay.|
