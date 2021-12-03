import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from scipy.io import wavfile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchaudio

import requests
from tqdm import tqdm
from pathlib import Path
import json
import glob
import os
import shutil
import numpy as np
import tarfile

import requests
import shutil

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename

class BatchItem:
    """
    Reprensents a single batch item. A :class:`Batch` can be built
    from multiple :class:`BatchItem` using :func:`collate`.
    Attributes:
        metadata (dict[str, object]): Contains all the metadata
            about the batch item. Those elements will not be
            collated together
            when building a batch
        tensors (dict[str, tensor]): Contlsains all the tensors
            for the batch item. Those elements will be collated
            together when building a batch using :func:`default_collate`.
    """

    def __init__(self, metadata=None, tensors=None):
        self.metadata = dict(metadata) if metadata else {}
        self.tensors = dict(tensors) if tensors else {}


def collate(items):
    """
    Collate together all the items into a :class:`Batch`.
    The metadata dictionaries will be added to a list
    and the tensors will be collated using
    :func:`torch.utils.data.dataloader.default_collate`.
    Args:
        items (list[BatchItem]): list of the items in the batch
    Returns:
        Batch: a batch made from `items`.
    """
    metadata = [item.metadata for item in items]
    tensors = default_collate([item.tensors for item in items])
    return Batch(metadata=metadata, tensors=tensors)


class Batch:
    """
    Represents a batch. Supports iteration
    (yields individual :class:`BatchItem`) and indexing. Slice
    indexing will return another :class:`Batch`.
    Attributes:
        metadata (list[dict[str, object]]): a list of dictionaries
            for each element in the batch.
            Each dictionary contains information
            about the corresponding item.
        tensors (dict[str, tensor]): a dictionary of collated tensors.
            The first dimension of each tensor will always be `B`,
             the batch size.
    """

    def __init__(self, metadata, tensors):
        self.metadata = metadata
        self.tensors = tensors

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None:
                raise IndexError("Does not support slice with step")
            metadata = self.metadata[index]
            tensors = {
                name: tensor[index]
                for name, tensor in self.tensors.items()
            }
            return Batch(metadata=metadata, tensors=tensors)
        else:
            return BatchItem(
                metadata=self.metadata[index],
                tensors={
                    name: tensor[index]
                    for name, tensor in self.tensors.items()
                })

    def apply(self, function):
        """
        Apply a function to all tensors.
        Arguments:
            function: callable to be applied to all tensors.
        Returns:
            Batch: A new batch
        """
        tensors = {
            name: function(tensor)
            for name, tensor in self.tensors.items()
        }
        return Batch(metadata=self.metadata, tensors=tensors)

    def apply_(self, function):
        """
        Inplace variance of :meth:`apply`.
        """
        other = self.apply(function)
        self.tensors = other.tensors
        return self

    def cuda(self, *args, **kwargs):
        """
        Returns a new batch on GPU.
        """
        return self.apply(lambda x: x.cuda())

    def cuda_(self, *args, **kwargs):
        """
        Move the batch inplace to GPU.
        """
        return self.apply_(lambda x: x.cuda())

    def cpu(self, *args, **kwargs):
        """
        Returns a new batch on CPU.
        """
        return self.apply(lambda x: x.cpu())

    def cpu_(self, *args, **kwargs):
        """
        Move the batch inplace to CPU.
        """
        return self.apply_(lambda x: x.cpu_())

class NSynthMetadata:
    """
    NSynth metadata without the wavforms.
    Arguments:
        path (Path): path to the NSynth dataset.
            This path should contain a `examples.json` file.
    An item of the nsynth metadata dataset will contain the follow tensors:
        - instrument (LongTensor)
        - pitch (LongTensor)
        - velocity (LongTensor)
        - instrument_family (LongTensor)
        - index (LongTensor)
    Attributes:
        cardinalities (dict[str, int]): cardinality of
            instrument, instrument_family, pitch and velocity
        instruments (dict[str, int]): mapping from instrument
            name to instrument index
    """
    _json_cache = {}

    _FEATURES = ['instrument', 'instrument_family', 'pitch', 'velocity']

    def _map_velocity(self, metadata):
        velocity_mapping = {
            25: 0,
            50: 1,
            75: 2,
            100: 4,
            127: 5,
        }
        for meta in self._metadata.values():
            meta["velocity"] = velocity_mapping[meta['velocity']]

    def __init__(self, path="", batch=True):
        self.path = Path(path)
        
        # Cache the json to avoid reparsing it everytime
        if self.path in self._json_cache:
            self._metadata = self._json_cache[self.path]
        else:
            if self.path.suffix == ".gz":
                file = gzip.open(self.path)
            else:
                file = open(self.path, "rb")
            self._metadata = json.load(file)
            self._map_velocity(self._metadata)
            self._json_cache[self.path] = self._metadata

        self.names = sorted(self._metadata.keys())

        # Compute the mapping instrument_name -> instrument id
        self.instruments = {}
        for meta in self._metadata.values():
            self.instruments[meta["instrument_str"]] = meta["instrument"]

        # Compute the cardinality for the features velocity, instrument,
        # pitch and instrument_family
        self.cardinalities = {}
        for feature in self._FEATURES:
            self.cardinalities[feature] = 1 + max(
                i[feature] for i in self._metadata.values())

        self.batch = batch

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if hasattr(index, "item"):
            index = index.item()
        name = self.names[index]
        metadata = self._metadata[name]
        tensors = {}

        metadata['name'] = name
        metadata['index'] = index
        for feature in self._FEATURES:
            tensors[feature] = torch.LongTensor([metadata[feature]])

        if self.batch:
            return BatchItem(metadata=metadata, tensors=tensors)
        else:
            return metadata, tensors

class NSynthDataset:
    """
    NSynth dataset.
    Arguments:
        path (Path): path to the NSynth dataset.
            This path should contain a `examples.json` file
            and an `audio` folder containing the wav files.
        pad (int): amount of padding to add to the waveforms.
    Items from this dataset will contain all the information
    coming from :class:`NSynthMetadata` as well as a `'wav'`
    tensor containing the waveform.
    Attributes:
        metadata (NSynthMetadata): metadata only dataset
    """
    def __init__(self, path="data", pad=0,
                dataset_size=0, full_meta=False, instrument=""):
        if instrument=="diverse_baseline":
            self.metadata = NSynthMetadata(path=Path(path) / f"diverse_1024_baseline.json")
        elif instrument=="keyboard_baseline":
            self.metadata = NSynthMetadata(path=Path(path) / f"keyboard_1024_baseline.json")

        try:
            path = self.metadata.path.parent / "audio" / "{}.wav".format(
                        item.metadata['name'])      
            _, wav = wavfile.read(str(path), mmap=True)  
        except:
            print("!!! NSYNTH dataset not found... !!! \n")
            inp = input("Is your OS able to adress ~20GB for the NSYNTH download? FAT32 filesystems generally are not. y/n \n")
            if inp == "y":
                print("starting download...")
                NSYNTH_train_url = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
                fname = f"{path}/nsynth_train.tar.gz"
                r = requests.get(NSYNTH_train_url, stream=True)
                with open(fname, 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    print("Downloading... This might take a while...")
                    for chunk in tqdm(r.iter_content(chunk_size=1024)): 
                        if chunk:
                            f.write(chunk)
                            f.flush()
                print("Finished downloading, unzipping file..")
                tar = tarfile.open(fname, "r:gz")
                tar.extractall()
                tar.close()
            else:
                print("Please download the dataset manually from: http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz \n Put the 'audio' folder containing wavefiles in the data folder: data/audio/*.wav \n")
                print("Exiting..")
                exit()
        self.pad = pad  
        self.dataset_size = dataset_size

        self.full_meta = full_meta

    def __len__(self):
        return self.dataset_size or len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]

        path = self.metadata.path.parent / "audio" / "{}.wav".format(
            item.metadata['name'])
        item.metadata['path'] = path

        _, wav = wavfile.read(str(path), mmap=True)
        wav = torch.as_tensor(wav, dtype=torch.float)
        wav /= 2**15 - 1
        item.tensors['wav'] = F.pad(wav, (self.pad, self.pad))

        if self.full_meta:
            return item, index

        return item.tensors['wav'][:16000], index

def audio_pad_fn(batch):
    tensors = [b.t() for b, idx in batch]
    idx = [idx for b, idx in batch]
    tensors.append(torch.zeros(16000,1))
    tensors = pad_sequence(tensors, batch_first=True)
    tensors = tensors[:-1,:,:]
    tensors = tensors.transpose(1, -1)
    return tensors.squeeze(1), idx

def filter_speech_commands(root, dataset_size=1024):
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]

    items_per_word = dataset_size // len(words)
    rest = dataset_size % len(words)

    to_delete = [item for item in os.listdir(f"{root}/SpeechCommands/speech_commands_v0.02/") if item not in words]
    print(to_delete)
    for item in to_delete:
        if os.path.isdir(f"{root}/SpeechCommands/speech_commands_v0.02/{item}"):
            shutil.rmtree(f"{root}/SpeechCommands/speech_commands_v0.02/{item}")

    for word in words:
        og_items = glob.glob(f"SPEECHCOMMANDS/SpeechCommands/speech_commands_v0.02/{word}/*")
        items_subset = og_items[items_per_word:]
        if word == words[-1]:
            items_subset = og_items[(items_per_word+rest):]
        for item in items_subset:
            os.remove(item)

class SPEECHCOMMANDSidx(Dataset):
    def __init__(self, root='./data/SCNUMBERS1024'):
        os.makedirs(root, exist_ok=True)
        self.SPEECHCOMMANDS = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True)
        if len(self.SPEECHCOMMANDS) != 1024:
            filter_speech_commands(root)

    def __getitem__(self, index):
        data, sr, word, hash, smth = self.SPEECHCOMMANDS[index]
        return data, index

    def __len__(self):
        return len(self.SPEECHCOMMANDS)

class CustomDataset(Dataset):

    def __init__(self, audio_dir, dataset_size):
        self.audio_dir = audio_dir
        self.samples = glob.glob(f"{self.audio_dir}/*.wav")[:dataset_size]
        self.indices = np.arange(len(self.samples))
        # self.indices = [i for i in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_sample_path = self.samples[index]
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, index

def get_dataloader(dataset, dataset_size, batch_size):
    if "." in dataset:
        dataset_name, instrument = dataset.split(".")
    else:
        dataset_name = dataset

    if "/" in dataset:
        dataset = CustomDataset(dataset, dataset_size)
        sampler = torch.utils.data.SubsetRandomSampler(dataset.indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=audio_pad_fn, sampler=sampler)

    if dataset_name == "NSYNTH":
        dataset = NSynthDataset(dataset_size=dataset_size, instrument=instrument)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=0, shuffle=True)
    elif dataset_name == "SPEECHCOMMANDS":
        dataset = SPEECHCOMMANDSidx()
        indices = np.arange(dataset_size)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=0, collate_fn=audio_pad_fn, sampler=sampler)
    
    return dataloader

if __name__ == "__main__":
    # Test dataloading speed
    import time

    dataset_name = "SPEECHCOMMANDS"
    dataset_size = 1024
    batch_size = 8

    train_loader = get_dataloader(dataset_name, dataset_size, batch_size)

    timestart = time.time()
    for batch, index in train_loader:
        timeend= time.time()
        print(f"loading batch took {(timeend - timestart)} seconds")
        print(index)
        timestart = time.time()