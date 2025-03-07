import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaConfig,
    apply_rotary_pos_emb
)
from rope import CustomLlamaRotaryEmbedding, LlamaDeepseekRotaryEmbedding

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_heads = config.num_attention_heads
        self.hidden_dim = config.nn_embed 
        self.head_dim = self.hidden_dim // self.num_heads

        self.latent_dim = self.hidden_dim // config.latent_compression_ratio

        # Matrix demposition is made of Down and Up projection matrices
        # Down projection matrices are used to project the hidden dimension to the latent dimension
        # k & v shares the same down projection matrix, but not the up matrix
        # print(f"self.hidden_dim: {self.hidden_dim}, self.latent_dim: {self.latent_dim}")
        self.kv_proj_d = nn.Linear(self.hidden_dim, self.latent_dim)
        self.q_proj_d = nn.Linear(self.hidden_dim, self.latent_dim)

        # Up projection matrices are used to project the latent dimension back to the hidden dimension.
        # However, as the Rope is applied on the hidden dimension, we need to split the latent dimension 
        # into two halves for the query and key projections.
        self.k_proj_u = nn.Linear(self.latent_dim, self.hidden_dim // 2)
        self.q_proj_u = nn.Linear(self.latent_dim, self.hidden_dim // 2)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_dim)

        # Rope-K is calculated on the input x (hidden dimension)
        self.rope_k = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        # Rope-Q is calculated on the q_down (latent dimension)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_dim // 2)

        # Apply Rope to the query and key projections
        # self.rotatry_embedding = CustomLlamaRotaryEmbedding(self.head_dim // 2)
        self.rotatry_embedding = LlamaDeepseekRotaryEmbedding(self.head_dim // 2)
        # self.rotatry_embedding = LlamaRotaryEmbedding(LlamaConfig(hidden_size=self.head_dim // 2)) # XXX
        
        # output projection
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.o_proj.NANGPT_SCALE_INIT = 1 # TODO do we need weight initialization scaling?
        self.register_buffer("rotatory_embed_cached", None, persistent=False)

    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq_len, hidden_dim] hidden_dim = embed
        # attention_mask: [batch_size, seq_len]

        B, seq_len, embed = x.shape

        # Down projection followed by up projection
        kv_d = self.kv_proj_d(x) # [B, seq_len, embed] -> [B, seq_len, latent_dim]
        q_d = self.q_proj_d(x) # [B, seq_len, embed] -> [B, seq_len, latent_dim]
        k_proj_1 = self.k_proj_u(kv_d) # [B, seq_len, latent_dim] -> [B, seq_len, hidden_dim // 2]
        q_proj_1 = self.q_proj_u(q_d) # [B, seq_len, latent_dim] -> [B, seq_len, hidden_dim // 2]
        v_proj_1 = self.v_proj_u(kv_d) # [B, seq_len, latent_dim] -> [B, seq_len, hidden_dim]

        # k_proj_1 & q_proj_1 makes the first half of the k & q
        # k_rope_2 & q_rope_2 makes the second half of the k & q after the rope embedding 
        # Rope-K is calculated on the input x (hidden dimension)
        k_rope_2 = self.rope_k(x) # [B, seq_len, embed] -> [B, seq_len, hidden_dim // 2]
        # Rope-Q is calculated on the q_down (latent dimension)
        q_rope_2 = self.rope_q(q_d) # [B, seq_len, latent_dim] -> [B, seq_len, hidden_dim // 2]
        
        # Reshape k,q,v n rope_k, rope_q to split into heads before applying Rope
        # print(f"k_proj_1.shape: {k_proj_1.shape}, q_proj_1.shape: {q_proj_1.shape}, v_proj_1.shape: {v_proj_1.shape}, k_rope_2.shape: {k_rope_2.shape}, q_rope_2.shape: {q_rope_2.shape}")
        k_proj_1 = k_proj_1.view(B, seq_len, self.num_heads, self.head_dim // 2) # [B, seq_len, hidden_dim // 2] -> [B, seq_len, num_heads, head_dim // 2]
        q_proj_1 = q_proj_1.view(B, seq_len, self.num_heads, self.head_dim // 2) # [B, seq_len, hidden_dim // 2] -> [B, seq_len, num_heads, head_dim // ]
        v_proj_1 = v_proj_1.view(B, seq_len, self.num_heads, self.head_dim) # [B, seq_len, hidden_dim] -> [B, seq_len, num_heads, head_dim]
        k_rope_2 = k_rope_2.view(B, seq_len, self.num_heads, self.head_dim // 2) # [B, seq_len, hidden_dim // 2] -> [B, seq_len, num_heads, head_dim // 2]
        q_rope_2 = q_rope_2.view(B, seq_len, self.num_heads, self.head_dim // 2) # [B, seq_len, hidden_dim // 2] -> [B, seq_len, num_heads, head_dim // 2]
        
        # Apply Rope to the query and key projections
        # print(f"k_rope_2.shape: {k_rope_2.shape}, q_rope_2.shape: {q_rope_2.shape}")
        # k_rope_2 = self.rotatry_embedding.apply_rotary_emb(k_rope_2, seq_len) # [B, seq_len, num_heads, head_dim // 2]
        # q_rope_2 = self.rotatry_embedding.apply_rotary_emb(q_rope_2, seq_len) # [B, seq_len, num_heads, head_dim // 2]
        # Transpose to (batch, num_heads, seq_len, head_dim//2) for rotary application.
        # =====XXX Following is only for LlamaDeepseekRotaryEmbedding=====
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)
        if self.rotatory_embed_cached is None or self.rotatory_embed_cached.shape[2] != seq_len:
            # print(f"Creating new rotatory_embed")
            rotatory_embed = self.rotatry_embedding(seq_len, x.device) 
            self.register_buffer("rotatory_embed_cached", rotatory_embed, persistent=False)
        else:
            # print(f"Using cached rotatory_embed")
            rotatory_embed = self.rotatory_embed_cached
        # print(f"rotatory_embed.shape: {rotatory_embed.shape}")
        # print(f"q_rope_2.shape: {q_rope_2.shape}")
        q_rope_2 = self.rotatry_embedding.apply_rotary_emb(q_rope_2, rotatory_embed) # [B, seq_len, num_heads, head_dim // 2]
        k_rope_2 = self.rotatry_embedding.apply_rotary_emb(k_rope_2, rotatory_embed) # [B, seq_len, num_heads, head_dim // 2]
        # Transpose back to (batch, seq_len, num_heads, head_dim//2)
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)
        # =====XXX End of LlamaDeepseekRotaryEmbedding=====
        
        # Concatenate the first half of the k & q (i.e. k_proj_1 & q_proj_1) with the second half of the k & q (i.e. k_rope_2 & q_rope_2)
        k = torch.cat([k_proj_1, k_rope_2], dim=-1) # [B, seq_len, num_heads, head_dim // 2] + [B, seq_len, num_heads, head_dim // 2] -> [B, seq_len, num_heads, head_dim]
        q = torch.cat([q_proj_1, q_rope_2], dim=-1) # [B, seq_len, num_heads, head_dim // 2] + [B, seq_len, num_heads, head_dim // 2] -> [B, seq_len, num_heads, head_dim]
        v = v_proj_1 # [B, seq_len, num_heads, head_dim]

        # Reshape to bring the seq_len to the next to head dimension for attention calculation
        k = k.transpose(1, 2) # [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2) # [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2) # [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]

        # Compute the attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, num_heads, seq_len, head_dim] @ [B, num_heads, head_dim, seq_len] -> [B, num_heads, seq_len, seq_len]
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask[:, None, :, :] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1) # [B, num_heads, seq_len, seq_len]
        out = attn @ v # [B, num_heads, seq_len, seq_len] @ [B, num_heads, seq_len, head_dim] -> [B, num_heads, seq_len, head_dim]

        # Reshape back to the original shape``
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_dim) # [B, num_heads, seq_len, head_dim] -> [B, seq_len, num_heads, head_dim] -> [B, seq_len, hidden_dim] (as hidden_dim = num_heads * head_dim)

        # Output projection
        out = self.o_proj(out) # [B, seq_len, hidden_dim] -> [B, seq_len, hidden_dim]

        return out
        
class DeepSeekExpert(nn.Module):
    # Every Expert is like a typicall FFN layer
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        
        self.gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.up = nn.Linear(dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, dim, bias=False)
        self.down.NANGPT_SCALE_INIT = 1 # TODO do we need weight initialization scaling - Optimization ?
        self.act_fn = nn.SiLU() # SwiGLU activation function

    def forward(self, x):
        return self.down(self.act_fn(self.gate(x)) * self.up(x))

class DeepSeekFFN(nn.Module):
    # Feedforward layer with Mixure of Experts
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.nn_embed
        self.intermediate_dim = config.ffn_intermediate_size 
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = self.num_experts - self.num_shared_experts
        self.top_k = config.experts_top_k

        # Shared Experts
        self.shared_experts = nn.ModuleList([DeepSeekExpert(self.dim, self.intermediate_dim) for _ in range(self.num_shared_experts)])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([DeepSeekExpert(self.dim, self.intermediate_dim) for _ in range(self.num_routed_experts)])
        
        # Routing to decide which Routing expert to use
        self.routing = nn.Linear(self.dim, self.num_routed_experts)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
        

    def forward(self, x, update_bias=False):
        B, seq_len, embed = x.shape

        # Process through the Shared Experts
        shared_out = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_out = shared_out / self.num_shared_experts

        # Calculate routing logits
        routing_logits = self.routing(x) + self.routing_bias
        routing_probs = torch.sigmoid(routing_logits)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1) #top_k_probs: [B, seq_len, top_k], top_k_indices: [B, seq_len, top_k]
        
        # Normalize the top k probabilities
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True) # [B, seq_len, top_k]

        # Process through the top_k routed experts
        routed_out = torch.zeros_like(x)
        expert_load = torch.zeros(self.num_routed_experts, device=x.device)
        # iterate over the top_k to process each one by one
        for k in range(self.top_k):
            # Kth expert index and probability for each token in the batch
            expert_indices = top_k_indices[..., k]  # [B, seq_len]
            expert_probs = top_k_probs[..., k].unsqueeze(-1)  # [B, seq_len, 1]
            # print(f"expert_indices.shape: {expert_indices.shape}, expert_probs.shape: {expert_probs.shape}")
            
            # Process each expert index to see which tokens should use this expert
            for expert_idx in range(self.num_routed_experts):
                # Create a mask for tokens that should use this expert and apply the expert to all inputs and mask out the ones that don't use it
                # mask = (expert_indices == expert_idx).unsqueeze(-1)  # [B, seq_len, 1]
                # expert_output = self.routed_experts[expert_idx](x) # [B, seq_len, dim]
                # print(f"mask.shape: {mask.shape}")
                # print(f"expert_output.shape: {expert_output.shape}")
                # print(f"mask : {mask}")
                # print(f"mask*expert_output : {mask*expert_output}")
                # routed_out += mask * expert_probs * expert_output # [B, seq_len, 1] * [B, seq_len, 1] * [B, seq_len, dim] -> [B, seq_len, dim]
                
                """ 
                Note: 
                The above code is generated by the cursor. It applies the mask after the expert is applied. As it is calculating over whole x and then masking it out by multiplying 3 matrices, it is slower
                The below code is copied from the class notes. This one applies the mask to the input itself and then only used the masked output for the calculation. As it is only considering the masked input for the calculation and 2 matrices for multiplication, it is faster.
                """
                mask = (expert_indices == expert_idx)  # [B, seq_len]
                masked_input = x[mask]  # [true_count, dim] # As mask is a 2D tensor being applied to a 3D tensor x, x[mask] yeilds a 2D tensor keeping the last dimension same as x.
                expert_output = self.routed_experts[expert_idx](masked_input) # [true_count, dim]
                routed_out[mask] += expert_output * expert_probs[mask] # [true_count, dim] * [true_count, 1] -> [true_count, dim]
                
                expert_load[expert_idx] += torch.sum(mask) # Update the expert load for the expert_idx  with the number of tokens that should use this expert for.

        if update_bias:
            expert_load = expert_load / (x.shape[0] * x.shape[1] * self.top_k) # Normalize the expert load 
            self.update_bias_terms(expert_load)

        # Return the sum of the shared and routed experts
        return shared_out + routed_out
    
    def update_bias_terms(self, expert_load):
        # Adjust the routing bias terms based on the expert load
        # expert_load: [num_routed_experts] is the load on each expert
        target_load = 1.0 / self.num_routed_experts # scalar constant number
        load_diff = expert_load - target_load #  [num_routed_experts]

        # Update rate (dynamic - based on the magnitude of the load difference)
        # update_rate = 0.1 * (torch.abs(load_diff) ** 0.5)
        update_rate = 0.1 * (torch.abs(load_diff)) # [num_routed_experts]

        # Update the routing bias terms TODO Why it is a scalar number? It should be a vector of size [num_routed_experts]
        self.routing_bias.data -= update_rate * load_diff # [num_routed_experts] * [num_routed_experts] -> scalar number
            

class DeepSeekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # RMSNorm
        self.attn_norm = nn.RMSNorm(config.nn_embed) # nn.RMSNorm(config.nn_embed, eps=config.rms_norm_eps)
        # Attention - Multi Head Latent Attention
        self.attn = MultiHeadLatentAttention(config)
        
        self.ffn_norm = nn.RMSNorm(config.nn_embed ) #nn.RMSNorm(config.nn_embed, eps=config.rms_norm_eps)

        # Feedforward
        self.ffn = DeepSeekFFN(config)

         # Add layer scale parameters
        # self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
        # self.ffn_scale = nn.Parameter(torch.ones(1) * 0.1)
     

    def forward(self, x, attention_mask=None, update_mlp_bias=False):
        # x: [batch_size, seq_len, nn_embed]
        # attention_mask: [batch_size, seq_len]

        # Multi Head Latent Attention
        x = x + self.attn(self.attn_norm(x), attention_mask)

        # Feedforward
        x = x + self.ffn(self.ffn_norm(x), update_mlp_bias)

        # Scale the residual connections
        # x = x + self.attention_scale * self.attn(self.attn_norm(x), attention_mask)
        # x = x + self.ffn_scale * self.ffn(self.ffn_norm(x), update_mlp_bias)
        

        return x

class DeepSeekTransfomerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # embedding layer (Normal Embedding as position embedding will be part of Attention layer)
        self.embedding = nn.Embedding(config.vocab_size, config.nn_embed)

        # total num_hidden_layers Blocks (Each block has attention and feedforward layer)
        self.layers = nn.ModuleList([
            DeepSeekBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # final RMS norm
        self.norm = nn.RMSNorm(config.nn_embed) #nn.RMSNorm(config.nn_embed, eps=config.rms_norm_eps)

        # head
        self.head = nn.Linear(config.nn_embed, config.vocab_size)

         # Optimization Weight sharing between lm_head and embedding
        self.head.weight = self.embedding.weight

        self.std = config.init_method_std
        # initialize weights
        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, update_mlp_bias=False, use_cache: bool = False, return_flatten_logits: bool = True):
        # TODO Should the mask be created for inference? if yes, would it be same?
        if (mask is None):
            mask = self.create_causal_mask(input_ids.shape[1], device=input_ids.device)
        # input_ids: [batch_size, seq_len] -> [batch_size, seq_len, nn_embed]
        x = self.embedding(input_ids)

        # [batch_size, seq_len, nn_embed] -> [batch_size, seq_len, nn_embed]
        for layer in self.layers:
            x = layer(x, mask, update_mlp_bias)
        
        # [batch_size, seq_len, nn_embed] -> [batch_size, seq_len, vocab_size]
        logits = self.head(self.norm(x))
        # print(f"logits.shape: {logits.shape}")

        if targets is not None:
            # print(f"targets.shape: {targets.shape}")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # print(f"logits.view(-1, logits.size(-1)).shape: {logits.view(-1, logits.size(-1)).shape} AND targets.view(-1).shape: {targets.view(-1).shape}")
            return logits, loss
        else:
            if return_flatten_logits:
                return logits.view(-1, logits.size(-1))
            else:
                return logits

    # Linear layers (attention projections, FFN layers, lm_head) are initialized from N(0, 0.02)
    # Embedding layer is initialized from N(0, 0.02)
    # All RMSNorm weights are initialized to 1.0
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = self.std
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.num_hidden_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = self.std) # Should it always be .02?

    def create_causal_mask(self, seq_len, device):
        """Creates a causal attention mask where each position can only attend to previous positions"""
        # Create lower triangular matrix (including diagonal)
        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1).bool()
        # # Invert and convert to float
        # return (~mask).float()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(device)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Generate text using the model
        Args:
            input_ids: Starting token ids (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness (1.0 = neutral, <1.0 = more deterministic, >1.0 = more random)
            top_k: Number of highest probability tokens to consider for sampling
        Returns:
            Generated token ids (B, T+max_new_tokens)
        """
        batch_size, seq_len = input_ids.shape
        
        # clear existing KV caching TODO enable after implementing KV caching
        # self.clear_cache()
        
        # Create a new tensor to store the generated tokens
        input_ids = torch.cat([input_ids, torch.zeros((batch_size, max_new_tokens), 
                            dtype=torch.long, device=input_ids.device)], dim=1)
        
        # Generate tokens one at a time
        for idx in range(max_new_tokens):
            # print(f"Generating token {idx+1} of {max_new_tokens}")
            
            # Get the current sequence length including cached tokens
            current_seq_len = seq_len + idx

            # TODO is it correct? Should the mask be created for inference? if yes, would it be same?
            next_mask = self.create_causal_mask(current_seq_len, device=input_ids.device)
            
            # Create mask that includes both the current input and cached tokens
            # if idx == 0:
            #     # First iteration - create mask for the full input sequence
            #     next_mask = self.create_causal_mask(current_seq_len, device=input_ids.device)
            # else:
            #     # Subsequent iterations - create mask for the new token attending to all previous tokens
            #     next_mask = torch.ones((1, 1, 1, current_seq_len), device=input_ids.device)

            # Process including the new tokens
            logits = self(input_ids[:, :current_seq_len], next_mask, use_cache=False, return_flatten_logits=False)
            
            # Get the last token's logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the filtered distribution
            next_token = top_k_indices[
                torch.arange(batch_size, device=input_ids.device),
                torch.multinomial(probs, num_samples=1).squeeze(1)
            ]
            
            # Update input_ids with the new token
            input_ids[:, current_seq_len] = next_token
        
        return input_ids
        
        


        
        



            


        
        



            