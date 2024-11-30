import requests
import torch
from datasets import load_dataset
import tiktoken
import os
import subprocess
import zipfile

def load_data(args):
    """
    Loads and prepares text data based on the specified dataset in args.dataset.
    """
    if args.dataset == "tinyshakespeare":
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(data_url)
        data = response.text
        
        chars = sorted(set(data))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        
        
        encoded_data = [stoi[c] for c in data]
        n = int(0.9 * len(encoded_data))
        
        train_data = torch.tensor(encoded_data[:n], dtype=torch.long)
        val_data = torch.tensor(encoded_data[n:], dtype=torch.long)
        additional_data = itos

    elif args.dataset == "tinystories":
        tinystories = load_dataset("roneneldan/TinyStories")
        text = '\n'.join(tinystories['train']['text'][:])
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        data = torch.tensor(enc.encode(text), dtype=torch.long)
        n = int(0.9 * len(data)) 
        
        train_data = data[:n]
        val_data = data[n:]
        additional_data = enc
        
    elif args.dataset == "wikitext":

        data_cache_dir = './data_raw/'
        checkpoint_path = os.path.join(data_cache_dir, 'wikitext-103.pt')
        tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = tokenizer.n_vocab
        
        if os.path.exists(checkpoint_path):
            print("Loading tokenized data from checkpoint.")
            data = torch.load(checkpoint_path)
        else:
            print("Downloading and preparing Wikitext data")
            raw_data_source = 'https://wikitext.smerity.com/wikitext-103-raw-v1.zip'
            raw_data_cache = './data_raw/'

            if not os.path.exists(raw_data_cache):
                os.makedirs(raw_data_cache, exist_ok=True)
                subprocess.run(["wget", raw_data_source, "-O", raw_data_cache + "data.zip"], stdout=subprocess.PIPE)

                with zipfile.ZipFile(raw_data_cache + 'data.zip', 'r') as zip_ref:
                    zip_ref.extractall(raw_data_cache)

            # Load and tokenize data
            with open(raw_data_cache + 'wikitext-103-raw/wiki.train.raw') as data_file:
                raw_train_data = data_file.read()

            with open(raw_data_cache + 'wikitext-103-raw/wiki.valid.raw') as data_file:
                raw_eval_data = data_file.read()

            raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
            raw_tokenized_eval  = tokenizer.encode_ordinary(raw_eval_data)

            train_tokenized = torch.tensor(raw_tokenized_train, device=f"cuda:{args.device}", dtype=torch.long) 
            eval_tokenized  = torch.tensor(raw_tokenized_eval,  device=f"cuda:{args.device}", dtype=torch.long)

            data = {
                'train': train_tokenized,
                'eval': eval_tokenized
            }

            torch.save(data, checkpoint_path)
            print("Tokenized data saved to checkpoint.")

        train_data = data['train']
        val_data = data['eval']
        additional_data = tokenizer

        print("Completed the tokenization process!")

    else:
        raise NotImplementedError(f"Dataset {args.data} not supported.")

    print(f"vocab size: {vocab_size}")
    print(f"train data shape: {train_data.shape}, validation data shape: {val_data.shape}")

    return train_data, val_data, vocab_size, additional_data


