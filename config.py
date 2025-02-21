from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 67
    vocab_size: int = 49152 # it should match the vocab size of the tokenizer
    num_hidden_layers: int = 30 # number of layers
    num_attention_heads: int = 8 # number of heads TODO
    # num_key_value_heads: int = 3 # number of key and value heads
    nn_embed: int = 768 # embedding dimension or hidden_size
    max_sequence_len: int = 2048 # max token sequence length (for pos embedding) # Block size
    ffn_intermediate_size: int = 1536 # intermediate size for the FFN layer
    latent_compression_ratio: int = 4 # Compression ratio for the multi head latent attention layer
    num_experts: int = 8 # number of experts
    num_shared_experts: int = 1 # number of shared experts
    experts_top_k: int = 2 # top k for the model
    expert_load_update_interval: int = 100 # Check and Update the bias terms every 100 steps
    rms_norm_eps: float = 1.0e-05
    nn_top_k: int = 50 # top k for the model
    nn_temperature: float = 1.0 # temperature for the model
    tokenizer_name_or_path: str = "HuggingFaceTB/cosmo2-tokenizer"
    checkpoint_interval: int = 1000
    checkpoints_path = "checkpoints"
    init_method_std = 0.041666666666666664
    nn_train_tok_seq: int = 64 # 1024 # 2048 64 
    micro_batch_size: int = 8
    intended_batch_size: int = 64  # 8
    train_steps: int = 2000000 # SmolLM2-135M steps. We are not using it
    optimizer_learning_rate_scheduler_learning_rate: float = 0.003
    optimizer_learning_rate_scheduler_lr_decay_starting_step: int = 1600000
    optimizer_learning_rate_scheduler_lr_decay_steps: int = 400000
    optimizer_learning_rate_scheduler_lr_decay_style: str = "linear"
    optimizer_learning_rate_scheduler_lr_warmup_steps: int = 2000
    optimizer_learning_rate_scheduler_lr_warmup_style: str = "linear"
    optimizer_learning_rate_scheduler_min_decay_lr: float = 0
    optimizer_factory_adam_beta1: float = 0.9
    optimizer_factory_adam_beta2: float = 0.95
    optimizer_factory_adam_eps: float = 1.0e-08
    optimizer_factory_name: str = "adamW"
    optimizer_factory_torch_adam_is_fused: bool = True
    optimizer_weight_decay: float = 0.01
    optimizer_zero_stage: int = 0
    optimizer_clip_grad: float = 1.0
 