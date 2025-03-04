import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import torch.nn.functional as F
from torchinfo import summary
from model import DeepSeekTransfomerModel
from utils import get_device
from config import Config
import os
import time
from torch_lr_finder import LRFinder, TrainDataLoaderIter

class CustomDataLoader(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch):
        return batch[:, :-1].contiguous(),  batch[:, 1:].contiguous().view(-1)

class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, block_size=512):
        # self.dataset = load_dataset("smollm-ai/smollm-corpus", streaming=True)["train"]
        self.dataset =  load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True)["train"]
        
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        
        for item in iterator:
            tokens = self.tokenizer.encode(item['text'])
            buffer.extend(tokens)
            
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[:self.block_size])
                buffer = buffer[self.block_size:]



def get_pretrained_tokenizer_n_model():
    checkpoint = "deepseek-ai/DeepSeek-V3"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer

def get_custom_tokenizer_n_model(config):
    model = DeepSeekTransfomerModel(config=config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)  # You can use a different tokenizer
    return model, tokenizer


    # return mask.to(device)

def printModelSummary(model, config):
    print(f"Model: {model}")
    summary(model, 
            input_size=(config.micro_batch_size, config.nn_train_tok_seq),
            dtypes=[torch.long],
            col_names=["input_size", "output_size", "num_params", "mult_adds", "params_percent"])

def compareModels(device, config):
    model1, tokenizer1 = get_pretrained_tokenizer_n_model()
    model1.to(device)
    model2, tokenizer2 = get_custom_tokenizer_n_model(Config())
    model2.to(device)

    print("Model 1 - HuggingFaceTB/SmolLM2-135M:")
    printModelSummary(model1, config)
    print("Model 2 - Custom SmolLM2-135M Model :")
    printModelSummary(model2, config)

def test(model, tokenizer, device, config):
    inputs = tokenizer.encode("What is Gravity?", return_tensors="pt").to(device)
    B, T = inputs.size()

    outputs = model.generate(inputs, max_new_tokens=30, temperature=config.nn_temperature, top_k=config.nn_top_k)
    print(tokenizer.decode(outputs[0]))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    # Use weights_only=True for safer loading
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    steps = checkpoint['steps']
    return start_epoch, steps, checkpoint['loss']

def save_checkpoint(model, optimizer, scheduler, epoch, steps, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'steps': steps,
        'loss': loss
    }, checkpoint_path)

def check_n_update_router_bias(model, inputs):
    # TODO as per the class notes, the router bias is updated for only 0th layer?
    # Also, we are passing the direct inputs to the router for the expert_load calculation. Wouldn't it make more sense to calculate the expert_load within the model itself by passing a boolean flag?

    for (layerIdx, layer) in enumerate(model.model.layers):
        layerFfn = model.model.layers[layerIdx].ffn
        #  expert_load matrix representing the load on each expert
        expert_load = torch.zeros(layerFfn.num_routed_experts, device=model.device) # [num_routed_experts]
        for k in range(layerFfn.top_k):
            # Calculate the routing logits and probabilities for the given input
            # TODO why the top_k_indices needs to be calculated within the layerFfn.top_k. Shouldn't it be outside the first loop?
            routing_logits = layerIdx.routing(inputs) + layerIdx.routing_bias
            routing_probs = torch.sigmoid(routing_logits)
            _, top_k_indices = torch.topk(routing_probs, layerFfn.top_k, dim=-1) # top_k_indices: [B, seq_len, top_k]
            
            # Calculate the expert load for every router expert for the calculated top_k_indices for this input
            for i in range(layerFfn.num_routed_experts):
                expert_load[i] += top_k_indices[..., k].eq(i).sum().item()
            expert_load = expert_load / (inputs.size(0) * inputs.size(1) * layerFfn.top_k) # Normalize the expert load by the number of tokens
            
        layerFfn.update_bias_terms(expert_load)
        

def get_lr_scheduler(optimizer, max_steps):
    # Warmup for 10% of total steps or minimum 100 steps, whichever is larger
    warmup_steps = max(100, int(0.1 * max_steps))
    
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def find_optimal_lr(model, train_loader, device, criterion=nn.functional.cross_entropy, gradient_accumalate_steps=1):
    criterion = torch.nn.functional.cross_entropy
    """
    Find the optimal learning rate using the LR Finder.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        device: Device to run the model on
        criterion: Loss function (if None, assumes model returns loss)
    
    Returns:
        suggestion: The suggested learning rate
    """
    # Initialize optimizer with a very low learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

    # Following is the code from the class notes
    # lr_finder = LRFinder(model, optimizer, criterion, device=device)
    # lr_finder.range_test(dataloader, end_lr=10, num_iter=100, step_mode='linear', accumulation_steps=gradient_accumalate_steps)
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()
    
     # Initialize the LR Finder 
    lr_finder = LRFinder(
        model, 
        optimizer, 
        criterion, 
        device=device,
    )
    
    # Run the range test
    lr_finder.range_test(
        train_loader,
        end_lr=10,
        num_iter=100,
        step_mode="exp",  # exponential increase in learning rate
        accumulation_steps=gradient_accumalate_steps,
        smooth_f=0.05,
        diverge_th=5,
    )
    
    # Plot the results
    lr_finder.plot()
    
    # Get the suggestion for the learning rate
    suggestion = lr_finder.suggestion()
    
    # Reset the model and optimizer to their initial states
    lr_finder.reset()
    
    return suggestion

def setup_one_cycle(model, train_loader, lr, max_lr, max_steps):
    """
    Set up the optimizer and OneCycleLR scheduler.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        max_lr: Maximum learning rate (from LR Finder)
        epochs: Number of training epochs
    
    Returns:
        optimizer: AdamW optimizer
        scheduler: OneCycleLR scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        # epochs=epochs,
        # steps_per_epoch=len(train_loader),
        total_steps=max_steps,
        pct_start=0.3,  # Spend 30% of time increasing LR
        div_factor=25,  # Initial LR will be max_lr/25
        final_div_factor=1e4,  # Final LR will be max_lr/10000
        anneal_strategy='cos',
        three_phase=False  # Use two-phase training
    )
    
    return optimizer, scheduler


def train_model():
    config = Config()
    SEED = config.seed
    device = get_device(seed=SEED)

    # Compare Actual HuggingFaceTB/SmolLM2-135M with my model for parameters and layers
    # compareModels(device)

    # Speed up
    torch.set_float32_matmul_precision('high')

    # Initialize model
    # model, tokenizer = get_pretrained_tokenizer_n_model()
    model, tokenizer = get_custom_tokenizer_n_model(config)
    printModelSummary(model, config)
    
    print(f"Device: {device}")
    model, tokenizer = get_custom_tokenizer_n_model(config)
    model.to(device)

    torch.compile(model) # As per the class, torch.compile doesn't work for Windows or Mac, but it appears to be working for Mac M4Pro
    vocab_size = tokenizer.vocab_size
    
    gradient_accumalate_steps = config.intended_batch_size // config.micro_batch_size
    print(f"gradient_accumalate_steps: {gradient_accumalate_steps}")
     # Initialize dataset and dataloader
    dataset = StreamingDataset(tokenizer, block_size=config.nn_train_tok_seq + 1)  # + 1 to get an extra token for token we use [0..n] for input and [1..n+1] for target
    dataloader = DataLoader(dataset, batch_size=config.micro_batch_size)

    # Find the learning rate
    find_lr = True
    lr = 1.59E-03
    max_lr = 1E-02
    # if (find_lr):
    #     max_lr = find_optimal_lr(model, CustomDataLoader(dataloader), device, gradient_accumalate_steps=gradient_accumalate_steps)
    #     # Suggested LR: 1.59E-03
    #     print(f"max_lr: {max_lr}")

    checkpoint_interval =  1000
    max_steps =  10000 
    
    if (find_lr):
        optimizer, scheduler = setup_one_cycle(model, dataloader, lr, max_lr, max_steps)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config.optimizer_learning_rate_scheduler_learning_rate,
                                 weight_decay=config.optimizer_weight_decay,
                                #  eps=config.optimizer_factory_adam_eps,
                                 betas=(config.optimizer_factory_adam_beta1, config.optimizer_factory_adam_beta2))
        scheduler = get_lr_scheduler(optimizer, max_steps)
    
    # Try to load checkpoint if it exists
    start_epoch = 0
    start_step = 0
    checkpoint_path = config.checkpoints_path + '/checkpoint_step_final-1.pt'  # or specify a specific checkpoint like 'checkpoint_step_500.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, start_step, loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        print(f"Last Saved epoch {start_epoch} and step {start_step} with loss {loss}")
        start_step = start_step + 1
        print(f"Resuming from epoch {start_epoch} and next step {start_step} with loss {loss}")

    if start_step > 0:
        max_steps =  11000  # id already trained for maxSteps, then add 1000 more as per the assignment 
    # print("Testing the model before training...")
    # test(model, tokenizer, device, config)

    # Training loop
    model.train()
    # for epoch in range(start_epoch, 10):  # For this assignment we are not going to go over 1 epoch
    epoch = start_epoch
    expert_load_update_interval = config.expert_load_update_interval

    if start_step >= max_steps:
        print(f"Already Trained for max steps {max_steps}. Only testing and existing")
        test(model, tokenizer, device, config)
        return

    optimizer.zero_grad() # I am not sure if I need to call this one time in beginning when using the accumalating gradient
    for step, batch in enumerate(dataloader, start=start_step):
        batch = batch.to(device)
        start_time = time.time()
        
        # Create targets (shifted by 1 position)
        inputs = batch[:, :-1].contiguous().to(device)
        targets = batch[:, 1:].contiguous().to(device)

        if (step == 0):
            print(f"inputs: {inputs.shape}, targets: {targets.shape}")
            print(f"input.device: {inputs.device}, targets.device: {targets.device}")

        if step % expert_load_update_interval == 0:
            update_mlp_bias = True
            print(f"Updating MLP bias")
        else:
            update_mlp_bias = False

        # Forward pass
        # Speed up - Auto Cast (Forward pass)
         # Modified autocast section to handle MPS properly
        if device.type == 'mps':
            # MPS only supports float16 (didn't see any improvement with it , disabling it)
            #  with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs, loss = model(inputs, targets=targets, update_mlp_bias=update_mlp_bias)
        else:
            # Use autocast for CUDA and CPU
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs, loss = model(inputs, targets=targets, update_mlp_bias=update_mlp_bias)
        # print(f"outputs: {outputs.shape}, loss: {loss}")

        # Add gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer_clip_grad)

        # Backward pass
        loss.backward()
        # What is the following line and who added it?
        #torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer_clip_grad)
        if step % 10 == 0:  # Print every 100 steps
            print(f"Current learning rate: {scheduler.get_last_lr()[0]:.2e}")
        # Accumulate Gradient until meet the intended batch size or if its the last step
        if ((step + 1) % gradient_accumalate_steps == 0 or step >= max_steps):
            optimizer.step()
            optimizer.zero_grad()   
            scheduler.step()

        end_time = time.time()
        token_per_second = (inputs.shape[1] * inputs.shape[0]) / (end_time - start_time)
        print(f"Epoch: {epoch}, Step: {step}, Batch(micro): {step}, Batch (considering grad accum): {step // gradient_accumalate_steps},  Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s, Token/s: {token_per_second:.2f}")
        
        # Save chceckpoitn and Test the model every 500 steps
        if step % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, step, loss, f'{config.checkpoints_path}/checkpoint_step_{step}.pt')
            print(f"Saved checkpoint at step {step}")
            test(model, tokenizer, device, config)
        
        if (step  >= max_steps):
            #   Save final checkpoint 
            save_checkpoint(model, optimizer, scheduler, epoch, step, loss, f'{config.checkpoints_path}/checkpoint_final.pt')
            print("Saved final checkpoint")
            test(model, tokenizer, device, config)
            # Save the model
            torch.save(model.state_dict(), f'{config.checkpoints_path}/model_final_new.pt')
            print("Saved the trained model")
            print("Training complete!")
            return

    print("Training complete!!")



if __name__ == "__main__":
    train_model()