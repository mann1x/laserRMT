import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from lib.utils import gptq_data_utils
from tqdm import tqdm
import random
import numpy as np
import argparse
import os, sys
from pathlib import Path, PurePath
import signal
import time

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    

def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, -1)
        
class GracefulExiter():

    def __init__(self):
        self.state = False
        self.graceful = False
        signal.signal(signal.SIGINT, self.change_state)
        signal.signal(signal.SIGTERM, self.change_state)

    def change_state(self, signum, frame):
        if self.graceful:
            self.state = True
        else:
            sys.exit(1)

    def exit(self):
        return self.state

    def setgraceful(self):
        self.graceful = True

    def setimmediate(self):
        self.graceful = False

output_dir = Path.resolve(Path.cwd())
model_name = ""

resumefile = Path.resolve(output_dir).joinpath('.resume')
resumefileow = Path.resolve(output_dir).joinpath('.resumeow')
resumefileml = Path.resolve(output_dir).joinpath('.resumeml')
resumefilefa = Path.resolve(output_dir).joinpath('.resumefa')
resumefileip = Path.resolve(output_dir).joinpath('.resumeip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name', type=str, help='model name')

    args = parser.parse_args()
    if len(args.model_name) < 1:
        print("Please specify a model name")
        sys.exit(1)
    else:
        model_name = args.model_name
        print(f"Output dir: {output_dir}")

class ModelModifier:
    def __init__(self, model_name, original_weights, modified_layers, failed_attempts):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"":0})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.original_weights = original_weights
        self.modified_layers = modified_layers
        self.failed_attempts = failed_attempts
        self.flag = GracefulExiter()

    def exitresume(self):
        print("Saving models & co for resume...")
        self.save_model(output_dir)
        save_object(self.original_weights, resumefileow)
        save_object(self.modified_layers, resumefileml)
        save_object(self.failed_attempts, resumefilefa)
        resumefile.touch()
        print("Models and runtime data saved, exiting")
        sys.exit(0)

    def update_model_reduce_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        if layer_id in self.modified_layers:
            print(f"Layer {layer_id} has already been modified. Skipping.")
            return False

        for name, module in self.model.named_modules():
            if self.flag.exit():
                self.exitresume()
            if layer_type in name and str(layer_number) in name:
                print(f"Reconstructing layer: {name}")
                original_dtype = module.weight.dtype
                self.original_weights[name] = module.weight.detach().clone()
                weights = module.weight.double()
                U, S, V = torch.linalg.svd(weights, full_matrices=False)

                # Estimate sigma using the full IQR method
                sigma_estimated_full_iqr = self.estimate_sigma_with_full_iqr(S)

                # Calculate Marchenko-Pastur threshold
                n, m = weights.shape
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(sigma_estimated_full_iqr, n, m)

                # Retain only the singular values above the MP threshold
                S_reduced = torch.zeros_like(S)
                k = (S > mp_threshold_full_iqr).sum().item()
                S_reduced[:k] = S[:k]
                print(f"Reduced from {S.shape} to {k}")

                # Reconstruct the matrix using the thresholded singular values
                reconstructed_weights = U @ torch.diag(S_reduced) @ V
                reconstructed_weights = reconstructed_weights.to(original_dtype)
                module.weight = torch.nn.Parameter(reconstructed_weights)
                self.modified_layers.add(layer_id)
                return True

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold
    
    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349 ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
        return sigma_estimated


    def restore_model_original_layer(self, layer_type, layer_number):
        if self.flag.exit():
            self.exitresume()
        layer_id = f"{layer_type}_{layer_number}"
        for name, module in self.model.named_modules():
            if layer_type in name and layer_number in name:
                if name in self.original_weights:
                    module.weight = torch.nn.Parameter(self.original_weights[name])
                    print(f"Restored original weights for layer: {name}")
                    if layer_id in self.modified_layers:
                        self.modified_layers.remove(layer_id)
                else:
                    print(f"No original weights saved for layer: {name}")

    def calculate_model_perplexity(self, datasets=['wikitext2', 'c4', 'ptb'], seqlen=384, use_cuda_graph=False, use_flash_attn=False):
        model = self.model
        model_str = self.model_name
        acc_loss = 0.0
        total_samples = 0

        for dataset in datasets:
            input_tok = gptq_data_utils.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=model_str)
            nsamples = input_tok.numel() // seqlen
            input_tok = input_tok[0, :(seqlen * nsamples)].view(nsamples, seqlen)
            total_samples += nsamples

            #if not use_cuda_graph:
            #    model.reset()

            loss_fct = torch.nn.CrossEntropyLoss().cuda()
            progress = tqdm(range(nsamples))
            for ii in progress:
                input = input_tok[ii, :].cuda().view(1, -1)
                output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = input[:, 1:]
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                acc_loss += loss.item()
                progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / total_samples
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        return ppl
    
    ### Implement a Backward Search
    # Search for the optimal lower ranking approximations from the top layers downwards
    # Also, we try doing a greedy approach, in order to maximize the rank reduction.
    # We tune the compression rate based on Marchenko-Pastur Random Matrix Theory
    ######################################################################################

    def search_optimal_layer_modification(self, layer_types, layer_numbers, max_mod=5):
        # Calculate initial perplexity with original model weights
        if resumefileip.exists():
            with open(resumefileip, 'rb') as inp:
                initial_perplexity = pickle.load(inp)
        else:
            initial_perplexity = self.calculate_model_perplexity()
            save_object(initial_perplexity, resumefileip)
        print("="*70)
        print(f"The initial perplexity of the model is {initial_perplexity}")
        print("="*70)
        min_loss = initial_perplexity
        optimal_params = (None, None)
        mods = 0

        self.flag.setgraceful()
        print("Process can be resumed from now on")

        for layer_number in layer_numbers:
            if self.flag.exit():
                self.exitresume()
            for layer_type in layer_types:
                if mods >= max_mod and max_mod != -1:
                    return optimal_params, min_loss
                attempt = (layer_type, layer_number)
                if attempt in self.failed_attempts:
                    continue  # Skip this attempt if it has failed before

                try_update = self.update_model_reduce_layer(layer_type, layer_number)

                if not try_update:
                    continue  # Skip this attempt if it has already been modified before

                try:
                    loss = self.calculate_model_perplexity()
                    if loss < min_loss:
                        min_loss = loss
                        optimal_params = (layer_type, layer_number)
                        mods = mods + 1
                        # Break out of the loop as soon as a better configuration is found
                        print("*"*50)
                        print(f"Improved perplexity found: {min_loss} for layer {layer_type} {layer_number}. Total modifications is {mods}")
                        print("*"*50)
                    else:
                        self.restore_model_original_layer(layer_type, layer_number)
                        self.failed_attempts.add(attempt)  # Record the failed attempt

                except NotImplementedError:
                    print("Perplexity calculation method is not implemented yet.")
                    return False, min_loss

        return optimal_params, min_loss

    def save_model(self, save_dir):

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

# Usage

_original_weights = {}
_modified_layers = set()
_failed_attempts = set()
if resumefile.exists():
        model_name = output_dir
        _original_weights = {}
        _modified_layers = set()
        _failed_attempts = set()
        with open(resumefileow, 'rb') as inp:
            _original_weights = pickle.load(inp)
        with open(resumefileml, 'rb') as inp:
            _modified_layers = pickle.load(inp)
        with open(resumefilefa, 'rb') as inp:
            _failed_attempts = pickle.load(inp)

modifier = ModelModifier(model_name, _original_weights, _modified_layers, _failed_attempts)

# %%
## Sequence of tests

modifier.update_model_reduce_layer('mlp.down_proj', '25')
modifier.update_model_reduce_layer('mlp.down_proj', '25')
modifier.restore_model_original_layer('mlp.down_proj', '25')
modifier.update_model_reduce_layer('mlp.down_proj', '25')
modifier.restore_model_original_layer('mlp.down_proj', '25')


# %%
layers = list(range(31, -1, -1))
layers = [f".{l}." for l in layers]
print('Layers:')
print(layers)

# %%
## Search and modify all layers

loop_check, min_loss = modifier.search_optimal_layer_modification(layer_types=['mlp.down_proj', 'mlp.up_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
                                    layer_numbers=layers)

# %%
modifier.save_model(output_dir)
resumefile.unlink()
resumefileow.unlink()
resumefileml.unlink()
resumefilefa.unlink()
resumefileip.unlink()
