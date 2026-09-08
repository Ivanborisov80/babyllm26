# train_mask_basic.py
import argparse
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from transformers import AutoConfig, AutoModelForMaskedLM, DebertaV2Tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset

from preprocessing import tokenize, padding_collate_fn, group_texts

from bitsandbytes.optim import LAMB

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="")
parser.add_argument("--valid_data", type=str, default="data/even.dev")
parser.add_argument("--max_seq_len", type=str, default="0:64,5:256", help="Either num or e.g. 0:32,5:64")  
parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=256, help="Number of examples per forward pass. Not affected by grad_acc.")
parser.add_argument("--grad_acc", type=int, default=1, help="Split the batch size into N mini-batches.")
parser.add_argument("--lr", type=float, default=0.007)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--cpus", type=int, default=10)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--all_checkpoints", action="store_true", help="Save and evaluate model every 1/10/100M tokens, \
                    per challenge stipulations. Overrides eval_steps and save_steps.")
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--intermediate_size", type=int, default=3072)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--mlm_prob", type=float, default=0.15)
parser.add_argument("--mask_replace_prob", type=float, default=0.8)
parser.add_argument("--random_replace_prob", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pretrained", action="store_true", help="Load pretrained model")
parser.add_argument("--eval_only", action="store_true", help="Evaluate only")
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
parser.add_argument("--wandb", action="store_true", help="Report to wandb")
parser.add_argument("--lamb", action="store_true", help="LAMB optimization")
parser.add_argument("--lower", action="store_true", help="Lowercase")
parser.add_argument("--flops", action="store_true", help="Compute FLOPs")
parser.add_argument("--log_gpu_mem", action="store_true", help="Log detailed GPU memory usage")

def evaluate(model, tokenizer, dataloader, args):
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if len(batch["input_ids"]) == 0:
                continue # NOTE: not sure why this happens in 100M case...

            masked_batch = mask_batch(batch, tokenizer, mlm_prob=0.15, mask_replace_prob=0.8, random_replace_prob=0.1)
            # split batch into grad_acc chunks
            batches = split_batch(masked_batch, args)
            for minibatch in batches:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda:0"):
                    outputs = model(**move_dict_to_cuda(minibatch))
            
                avg_loss += outputs.loss.item()
                logits = outputs.logits
                preds = logits.argmax(dim=-1)

                labels = minibatch["labels"].to(device=logits.device, dtype=logits.dtype)
                label_mask = labels != -100
                correct += (preds[label_mask] == labels[label_mask]).sum().item()
                total += preds[label_mask].numel()

    model.train()
    return {'acc': 100 * correct / total, 'loss': avg_loss / (len(dataloader) * args.grad_acc)}


def regroup_texts(args, max_seq_len):
    cur_max_seq_len = args.cur_max_seq_len
    print("CUR MAX SEQ LEN:", cur_max_seq_len)

    grouped_dataset = args.dataset.map(group_texts,
        batched = True,
        fn_kwargs = {'max_len': max_seq_len},
        # remove_columns = dataset["train"].column_names,
        num_proc=args.cpus,
        desc = "Grouping",
        # load_from_cache_file=False,
        )
    change_ratio = max_seq_len / cur_max_seq_len
    args.batch_size = max(1, int(args.batch_size / change_ratio))
    
    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['train'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=True, 
        collate_fn=padding_collate_fn
        )
    
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['validation'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=False, 
        collate_fn=padding_collate_fn
        )

    args.cur_max_seq_len = max_seq_len

    return train_dataloader, eval_dataloader

def mask_batch(batch, tokenizer, mlm_prob=0.15, mask_replace_prob=0.8, random_replace_prob=0.1):
    masked_batch = batch.copy()
    for i in range(len(batch["input_ids"])):
        enc = batch["input_ids"][i]
        pad_mask = enc == tokenizer.pad_token_id
        enc_mask_mask = (torch.rand(len(enc)) < mlm_prob) & ~pad_mask

        rand = torch.rand(len(enc))
        to_mask = (rand < mask_replace_prob) & enc_mask_mask
        to_replace = (rand >= mask_replace_prob) & (rand < mask_replace_prob + random_replace_prob) & enc_mask_mask

        randoms = torch.randint(0, tokenizer.vocab_size, (len(enc),))
        enc[to_mask] = tokenizer.mask_token_id
        enc[to_replace] = randoms[to_replace]

        masked_batch["input_ids"][i] = enc

        labels = batch["labels"][i]
        labels[~enc_mask_mask] = -100
        masked_batch["labels"][i] = labels

    return masked_batch

def split_batch(batch, args):
    minibatch_size = args.batch_size // args.grad_acc
    if len(batch["input_ids"]) == minibatch_size:
        return [batch]

    batches = []
    for i in range(0, len(batch["input_ids"]), minibatch_size):
        minibatch = {}
        for key in batch.keys():
            minibatch[key] = batch[key][i:i+minibatch_size]
        batches.append(minibatch)
    return batches

def move_dict_to_cuda(d):
    return {key: value.to(device="cuda:0") for key, value in d.items()}

def calculate_num_words(examples):
    return {'num_words': [sum(len(examples["text"][i]) for i in range(len(examples["text"])))]}

def calculate_num_tokens(examples):
    return {'num_tokens': [sum(len(examples["input_ids"][i]) for i in range(len(examples["input_ids"])))]}

def calculate_total_steps(args):
    def calc_exs_per_epoch(tokens_per_1000, max_seq_len):
        return sum([t // max_seq_len for t in tokens_per_1000])

    
    exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, args.init_max_seq_len)
    batches_per_epoch = math.ceil(exs_per_epoch / args.batch_size)
    cur_epoch = 0
    if len(args.max_seq_len) == 0:
        return batches_per_epoch * args.epochs
    else:
        total_steps = 0
        batch_size = args.batch_size
        prev_seq_len = args.init_max_seq_len
        for epoch_num, seq_len in args.max_seq_len:
            total_steps = total_steps + batches_per_epoch * (epoch_num - cur_epoch)

            len_ratio = prev_seq_len / seq_len
            batch_size = int(batch_size * len_ratio)

            exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, seq_len)
            batches_per_epoch = math.ceil(exs_per_epoch / batch_size)
            cur_epoch = epoch_num
            prev_seq_len = seq_len
    
        total_steps = total_steps + batches_per_epoch * (args.epochs - cur_epoch)
        return total_steps

def is_step(step_type: str, global_step: int, args):
    # step_arg = args.logging_steps, args.save_steps, or args.eval_steps
    step_arg = getattr(args, f'{step_type}_steps')

    if args.all_checkpoints:
        if global_step in args.checkpoints:
            return True
    else:
        if global_step % step_arg == 0 and global_step != 0:
            return True
        
    return False



def train(args, model, tokenizer, train_dataloader, eval_dataloader):
    if args.flops:
        from fvcore.nn import FlopCountAnalysis


    if args.lamb:
        optimizer = LAMB(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.total_steps//100, num_training_steps=args.total_steps)

    model.train()
    model = model.to(dtype=torch.bfloat16, device="cuda:0")

    if args.log_gpu_mem:
        # Print memory usage
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    global_step = 0
    print(f"Total steps: {args.total_steps}")
    print(f"Max seq len: {args.init_max_seq_len}")
    print(f"Next seq len: {args.max_seq_len}")
    with tqdm(total=args.total_steps) as pbar:
        for epoch in range(args.epochs):
            if len(args.max_seq_len) > 0:
                if epoch >= args.max_seq_len[0][0]:
                    train_dataloader, eval_dataloader = regroup_texts(args, args.max_seq_len[0][1])
                    args.max_seq_len = args.max_seq_len[1:]

            for step, batch in enumerate(train_dataloader):
                masked_batch = mask_batch(
                    batch,
                    tokenizer,
                    mlm_prob=args.mlm_prob,
                    mask_replace_prob=args.mask_replace_prob,
                    random_replace_prob=args.random_replace_prob,
                )

                # split batch into grad_acc chunks
                batches = split_batch(masked_batch, args)
                
                for minibatch in batches:

                    with torch.autocast(dtype=torch.bfloat16, device_type="cuda:0"):
                        if args.flops:
                            model.eval()
                            flops = FlopCountAnalysis(model, tuple(move_dict_to_cuda(minibatch).values()))
                            flops = flops.by_operator()
                            total = 0
                            for key in flops:
                                total += flops[key]
                            print(f"Estimated Total FLOPs: {total*3*args.total_steps}")
                            exit()


                        outputs = model(**move_dict_to_cuda(minibatch))
                        loss = outputs.loss
                        
                        # Add z-loss
                        logits = outputs.logits  # [B,T,V]
                        labels = minibatch["labels"].to(device=logits.device)
                        valid = labels.ne(-100)
                        z = torch.logsumexp(logits, dim=-1)  # [B,T]
                        z = z.masked_select(valid)
                        z_loss = (z**2).mean() if z.numel() else torch.tensor(0., device=logits.device)
                        lam = 0.0001  # z-loss coefficient
                        loss = loss + lam * z_loss

                    loss = loss / args.grad_acc # To ensure consistent gradient magnitude
                    loss.backward()
                    
                # Gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # ----- LOGGING -----
                if is_step("logging", global_step, args):
                    epoch_float = global_step * args.epochs / args.total_steps
                    print(f"Epoch {epoch_float:.2f}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
                    
                    if args.log_gpu_mem:
                        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                    
                    if args.wandb:
                        log_dict = {
                            "epoch": epoch_float,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0]
                        }
                        if args.log_gpu_mem:
                            log_dict.update({
                                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3
                            })
                        wandb.log(log_dict)

                # ----- EVALUATION -----
                if is_step("eval", global_step, args):
                    metrics = evaluate(model, tokenizer, eval_dataloader, args)
                    print(f"----- Eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f} -----", flush=True)

                    if args.wandb:
                        wandb.log({
                            "eval_acc": metrics["acc"],
                            "eval_loss": metrics["loss"]
                        })

                # ----- SAVING -----
                if is_step("save", global_step, args):
                    save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"----- Saved checkpoint to: {save_path} -----", flush=True)

                pbar.update(1)
                global_step += 1

    metrics = evaluate(model, tokenizer, eval_dataloader, args)
    print(f"Final eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f}", flush=True)

    save_path = os.path.join(args.output_path, f"checkpoint-{args.total_steps}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if args.wandb:
        wandb.finish()

def parse_max_seq_len(max_seq_len):
    if "," in max_seq_len:
        max_seq_len = max_seq_len.split(",")
        assert ":" in max_seq_len[0]
        return [(int(val.split(":")[0]), int(val.split(":")[1])) for val in max_seq_len]
    elif ":" in max_seq_len:
        max_seq_len = max_seq_len.split(":")[1]
    return [(0, int(max_seq_len))]

def main():
    args = parser.parse_args()
    args.max_seq_len = parse_max_seq_len(args.max_seq_len)
    set_seed(args.seed)

    if args.wandb:
        assert wandb_available
        output_dir = os.path.basename(os.path.normpath(args.output_path))
        wandb.init(
            project='babylm25',
            name=output_dir,
            config=vars(args),   
        )

    tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer, do_lower_case=args.lower)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config.vocab_size = tokenizer.vocab_size
    config.max_position_embeddings = 1024
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.sep_token_id = tokenizer.sep_token_id


    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.dropout = args.dropout
    config.hidden_dropout_prob = args.dropout


    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    # print model num parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")

    # for p in model.named_parameters():
    #     print(p[0], p[1].shape)

    dataset = load_dataset('text', data_files={'train': args.train_data, 'validation': args.valid_data})

    if args.debug:
        dataset['train'] = dataset['train'].select(range(100))
        dataset['validation'] = dataset['validation'].select(range(100))

    dataset = dataset.map(tokenize, 
        batched = True, 
        fn_kwargs = {'tokenizer': tokenizer, 'input_field': 'text'}, 
        remove_columns = dataset["train"].column_names, 
        num_proc=args.cpus,
        desc = "Tokenizing",
        # load_from_cache_file=False,
        )
        
    args.dataset = dataset

    print(dataset)

    max_seq_len = args.max_seq_len.pop(0)[1]
    args.init_max_seq_len = max_seq_len
    args.cur_max_seq_len = max_seq_len
    args.dataset_len_lines = len(dataset['train'])
    args.tokens_per_1000 = dataset['train'].map(calculate_num_tokens, 
                                                       batched = True,
                                                       num_proc=args.cpus, 
                                                       remove_columns=dataset["train"].column_names)['num_tokens']

    args.dataset_len_tokens = sum(args.tokens_per_1000)
    args.total_steps = calculate_total_steps(args)

    args.is_strict_small = (args.dataset_len_tokens // 10e6) < 10 # assuming there are fewer than 10 tokens per word

    if args.is_strict_small:
        steps_1m = np.round(np.linspace(args.total_steps//100, args.total_steps//10, 10)).astype(int)
        steps_10m = np.round(np.linspace(args.total_steps//10, args.total_steps, 10)).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:]
    else: # strict
        steps_1m = np.linspace(args.total_steps//1000, args.total_steps//100, 10).astype(int)
        steps_10m = np.linspace(args.total_steps//100, args.total_steps//10, 10).astype(int)
        steps_100m = np.linspace(args.total_steps//10, args.total_steps, 10).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:] + list(steps_100m)[1:]


    print(f"Dataset length: {args.dataset_len_lines} lines, {args.dataset_len_tokens} tokens", flush=True)
    print(f"Total steps: {args.total_steps}", flush=True)
    print(f"Save checkpoints: {args.checkpoints}", flush=True)


    grouped_dataset = dataset.map(group_texts,
        batched = True,
        fn_kwargs = {'max_len': max_seq_len},
        # remove_columns = dataset["train"].column_names,
        num_proc=args.cpus,
        desc = "Grouping",
        # load_from_cache_file=False,
        )

    print("Dataset length after grouping:", len(grouped_dataset['train']))

    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['train'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=True, 
        collate_fn=padding_collate_fn,
        )
    
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['validation'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=False, 
        collate_fn=padding_collate_fn,
        )

    train(args, model, tokenizer, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()
