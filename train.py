import os
import time
import json
import torch
import random
import numpy as np
from copy import deepcopy
from utils import *
from config import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# -------------------------
# Distributed / Device Setup
# -------------------------
world_size = int(os.environ.get('WORLD_SIZE', 1))
global_rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))

if world_size > 1:
    os.environ['USE_LIBUV'] = '0'  # Disable libuv on Windows
    dist.init_process_group(backend='gloo', init_method='env://')  # Use gloo for CPU
    print(f"[Rank {global_rank}] Distributed training enabled (world size={world_size})")
else:
    print("Single-CPU mode")

device = torch.device("cpu")  # Explicitly set to CPU

# -------------------------
# Reproducibility
# -------------------------
seed = 42 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# Model, Patchilizer, Optimizer
# -------------------------
batch_size = BATCH_SIZE
patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                          max_length=PATCH_LENGTH,
                          max_position_embeddings=PATCH_LENGTH,
                          vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                         max_length=PATCH_SIZE,
                         max_position_embeddings=PATCH_SIZE,
                         vocab_size=128)

model = MelodyT5(patch_config, char_config).to(device)
print("Parameter Number:", sum(p.numel() for p in model.parameters() if p.requires_grad))

if world_size > 1:
    model = DDP(model, find_unused_parameters=True)  # No device_ids for CPU

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# -------------------------
# Dataset / DataLoader - FIXED
# -------------------------
def collate_batch(batch):
    input_patches, input_masks, output_patches, output_masks = [], [], [], []

    for input_patch, output_patch in batch:
        input_patches.append(input_patch)
        input_masks.append(torch.tensor([1] * input_patch.shape[0]))
        output_patches.append(output_patch)
        output_masks.append(torch.tensor([1] * output_patch.shape[0]))

    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)
    output_patches = torch.nn.utils.rnn.pad_sequence(output_patches, batch_first=True, padding_value=0)
    output_masks = torch.nn.utils.rnn.pad_sequence(output_masks, batch_first=True, padding_value=0)

    return input_patches.to(device), input_masks.to(device), output_patches.to(device), output_masks.to(device)


class MelodyHubDataset(Dataset):
    def __init__(self, items):
        self.inputs = []
        self.outputs = []
        for item in tqdm(items):
            input_patch = torch.tensor(patchilizer.encode(item['input'], add_special_patches=True))
            output_patch = torch.tensor(patchilizer.encode(item['output'], add_special_patches=True))
            if torch.sum(output_patch) != 0:
                self.inputs.append(input_patch)
                self.outputs.append(output_patch)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# -------------------------
# Training / Evaluation - FIXED
# -------------------------
def process_one_batch(batch):
    # Debug: print batch structure to understand what's being passed
    if isinstance(batch, (list, tuple)):
        print(f"Batch is a {type(batch)} with {len(batch)} elements")
        for i, item in enumerate(batch):
            if hasattr(item, 'shape'):
                print(f"  Element {i}: shape {item.shape}")
            else:
                print(f"  Element {i}: type {type(item)}")
    else:
        print(f"Batch is type {type(batch)}")
        if hasattr(batch, 'shape'):
            print(f"Batch shape: {batch.shape}")

    # Try to unpack the batch
    try:
        input_patches, input_masks, output_patches, output_masks = batch
        print("Successfully unpacked batch into 4 tensors")
    except ValueError as e:
        print(f"Error unpacking batch: {e}")
        # If unpacking fails, investigate what we actually have
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            input_patches, input_masks, output_patches, output_masks = batch[0], batch[1], batch[2], batch[3]
        else:
            raise e

    loss = model(input_patches, input_masks, output_patches, output_masks)
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)
    return loss


def train_epoch():
    tqdm_train_set = tqdm(train_loader)  # Use train_loader instead of train_set
    total_train_loss, iter_idx = 0, 1
    model.train()
    for batch in tqdm_train_set:
        loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({str(global_rank) + '_train_loss': total_train_loss / iter_idx})
        iter_idx += 1
    return total_train_loss / (iter_idx - 1)


def eval_epoch():
    tqdm_eval_set = tqdm(eval_loader)  # Use eval_loader instead of eval_set
    total_eval_loss, iter_idx = 0, 1
    model.eval()
    for batch in tqdm_eval_set:
        with torch.no_grad():
            loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank) + '_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx - 1)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Load data
    train_items, eval_items = [], []
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f: train_items.append(json.loads(line.strip()))
    with open(VALIDATION_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f: eval_items.append(json.loads(line.strip()))

    # Trim to batch multiples
    train_items = train_items[:(len(train_items) // batch_size) * batch_size]
    eval_items = eval_items[:(len(eval_items) // batch_size) * batch_size]

    train_dataset = MelodyHubDataset(train_items)
    eval_dataset = MelodyHubDataset(eval_items)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=local_rank) if world_size > 1 else None
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size,
                                      rank=local_rank) if world_size > 1 else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler,
                              shuffle=(train_sampler is None))
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler,
                             shuffle=(eval_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)

    # Load pretrained / checkpoint if needed
    pre_epoch, best_epoch, min_eval_loss = 0, 0, float('inf')
    if LOAD_FROM_PRETRAINED and os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        if world_size > 1 and hasattr(model, 'module'):
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        print(f"Loaded pretrained checkpoint at epoch {checkpoint['epoch']}")

    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        if world_size > 1 and hasattr(model, 'module'):
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print(f"Loaded checkpoint from epoch {pre_epoch}")

    # Training loop
    for epoch in range(pre_epoch + 1, NUM_EPOCHS + 1):
        if train_sampler: train_sampler.set_epoch(epoch)
        if eval_sampler: eval_sampler.set_epoch(epoch)
        print('-' * 21 + f" Epoch {epoch} " + '-' * 21)

        train_loss = train_epoch()
        eval_loss = eval_epoch()

        if global_rank == 0:
            with open(LOGS_PATH, 'a') as f:
                f.write(f"Epoch {epoch}\ntrain_loss: {train_loss}\neval_loss: {eval_loss}\ntime: {time.asctime()}\n\n")
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = {
                    'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'min_eval_loss': min_eval_loss
                }
                torch.save(checkpoint, WEIGHTS_PATH)

        if world_size > 1: dist.barrier()

    if global_rank == 0:
        print("Best Eval Epoch:", best_epoch)
        print("Min Eval Loss:", min_eval_loss)