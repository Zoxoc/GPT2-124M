"""This file is where the code is written from scratch to train the gpt2 model"""

# importing libraries needed
# It’s a decorator that makes it easy to create classes that just store data (no need to write boilerplate __init__, __repr__, etc.)
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import math
import time
import inspect
import os
import numpy as np
from hellaswag import render_example, iterate_examples

# -------------------------------------------------------------------------------------------------


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50,000 BPE Tokens + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # did the transpose so that PyTorch treats B, n_head as batches and trains them parallely
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP or Feed-Forward Network"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #smoother version of ReLU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Defining what a transformer block is"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # using ModuleDict as lets us index through this like a dictionary
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #initialising parameters
        self.apply(self._init_weights)

    def _init_weights(self, module): #module = layer
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                # we have 2 residual blocks calculating in each layer
                std *= (2 * self.config.n_layer) ** -0.5
            #The underscore (_) means in-place operation. (modifies the tensor directly)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #used to set an existing tensor to zero

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        #idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of lengths {T}, block size is only {self.config.block_size}"

        #forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape (T)
        pos_emb = self.transformer.wpe(pos) #positional embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        #forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)

        #forward the final layer_norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        #calculating the loss
        loss = None
        if targets is not None:
            #flattening out the logits into 2D as it doesn't take multidimensional 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257

        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        # This creates a fresh model that matches GPT-2’s size — but it’s randomly initialized.
        model = GPT(config)

        sd = model.state_dict()  # sd --> state_dict()
        sd_keys = sd.keys()

        # Some keys are buffers, like attention masks — we don’t need them when copying weights.
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # initialise a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)  # hf --> hugging face
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight','mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape

                #Tells PyTorch "don't track gradients" while doing this copy - we’re not training here, just copying.
                with torch.no_grad(): 
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        #start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        #create optim groups. Any parameters that is 2D will be weight decayed. otherwise no
        #i.e. all weight tensors in matmuls + embedding decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        #Fuse several GPU operations (like momentum, weight decay, etc.) | Run them more efficiently in one CUDA kernel, reducing overhead
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters 
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")

        #β1 = 0.9, β2 = 0.95 & ε = 10^-8 from GPT-3 paper
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


#-------------------------------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        """
        Streaming loader for FineWeb10B via HuggingFace, with inline 90/10 train/val split.
        It is taking directly from the web instead of you saving it locally.
        """
        assert split in {"train", "val"}
        self.B = B
        self.T = T
        self.split = split

        # store these so reset() can re-shard
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # load & immediately shard the single 'train' split
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        ds = ds.shard(num_shards=self.num_processes, index=self.process_rank)
        self.iterator = iter(ds)
        self.buffer = []
        self.example_counter = 0

    def reset(self):
        """Rewind the stream & clear buffer."""
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        ds = ds.shard(num_shards=self.num_processes, index=self.process_rank)
        self.iterator = iter(ds)
        self.buffer = []
        self.example_counter = 0

    def _fill_buffer(self, min_tokens=1_000_000):
        """Accumulate at least min_tokens tokens, applying 90/10 split logic."""
        while len(self.buffer) < min_tokens:
            try:
                ex = next(self.iterator)
            except StopIteration:
                break
            self.example_counter += 1
            is_val = (self.example_counter % 10 == 0)
            if self.split == "train" and is_val:
                continue
            if self.split == "val" and not is_val:
                continue
            self.buffer.extend(self.tokenizer.encode(ex["text"]))

    def next_batch(self):
        """Return x,y of shape (B,T)."""
        self._fill_buffer(self.B * self.T + 1)
        needed = self.B * self.T + 1
        if len(self.buffer) < needed:
            raise RuntimeError("Ran out of data; call reset() to start over")
        chunk, self.buffer = self.buffer[:needed], self.buffer[needed:]
        x = torch.tensor(chunk[:-1], dtype=torch.long).view(self.B, self.T)
        y = torch.tensor(chunk[1:],  dtype=torch.long).view(self.B, self.T)
        return x, y

#-------------------------------------------------------------------------------------------------

# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

#-------------------------------------------------------------------------------------------------
#launch ddp: torchrun --standalone --nproc_per_node=1 train.py

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP at the moment demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


#for reproducability
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

#tokenizer
enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2^19, ~0.5M, in no.of tokens as said in GPT-2 paper
B = 32 #micro batch size
T = 1024 #sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"--> calculated gradient accumulation steps: {grad_accum_steps}")



train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high') #so it takes Tensor32 internally

#torch randomly initialises the weights
model = GPT(GPTConfig(vocab_size=50304)) #as 50,257 is not a power of 2
model.to(device)

"""torch.compile() traces your model and builds an optimized graph of operations. 
This graph can be run more efficiently — reducing Python overhead, 
fusing kernels, and minimizing trips between memory and GPU — which improves
performance."""
use_compile = False #torch.compile interferes with Hellaswag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the unwrapped model

#learning_rate with cosine decay
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

def get_lr(iteration):
    # 1. Linear warmup for warmup_iters steps
    if iteration < warmup_steps:
        return max_lr * (iteration+1) / warmup_steps
    # 2. If epoch > lr_decay_iters, return min learning rate
    if iteration > max_steps:
        return min_lr
    # 3. In between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to 0

    return min_lr + coeff * (max_lr - min_lr)

#optimize
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

#create a log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: #open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            max_len = raw_model.config.block_size
            tokens = tokens[:, :max_len]
            mask = mask[:, :max_len]

            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
                
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    #do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        """we have to scale the loss to account for gradient accumulation,
            because the gradients just add on each successive backward().
            So we need to average it out as the reduction 
            for cross_entropy_loss is mean."""
        loss /= grad_accum_steps
        #for printing the entire loss
        loss_accum += loss.detach() #for just the numeric value
        # we want the gradients to syncronise after DDP is done on all the process and at the last step, we average it out
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) 
        loss.backward()
    # averages out the losses across all the GPUs 
    if ddp:
        dist.all_reduce(loss_accum, op=list.ReduceOp.AVG)

    #It clips (limits) the gradients of your model’s parameters during backpropagation to prevent them from getting too large, which can destabilize training.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    #determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() #wait for the gpu to finish the scheduled work
    t1 = time.time()
    dt = (t1 - t0) #time difference in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
    if master_process:
        print(f"step{step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()