import torch
import torch.nn as nn

class LlamaDeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:

        positions = torch.arange(seq_len, device=device)
        sincos = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        # Rearranged so that seq_len is in dimension 2
        return emb[None, None, :, :]

    def apply_rotary_emb(self, x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        head_dim = x.shape[-1]
        x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
        sin, cos = rotary_emb[..., :head_dim // 2], rotary_emb[..., head_dim // 2:]
        rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated_x
    

# TODO: Check if the implementation is correct as it was updated by me to correct the shape of the sin and cos and the cache to match the shape of the q_k (Cursor was not able to correct it). 
class CustomLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        # Initializes the rotary embeddings with the given dimension and maximum sequence length. 
        # It creates inverse frequency bands and caches the position embeddings.
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position embeddings cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        # Creates and caches the sine and cosine embeddings for the given sequence length.
        self.max_seq_len_cached = seq_len
        
        # Calculate position embeddings
        t = torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Cache the embeddings
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        # self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        # self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def apply_rotary_emb(self, q_k, seq_len=None):
        """
        Applies the rotary embeddings to the query and key tensors. It:
        - Ensures the cache is large enough for the sequence length
        - Gets the cached sine and cosine values
        - Reshapes q and k to match the cache dimensions
        - Applies the rotation using complex multiplication
        - Returns the rotated query and key tensors

        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        # Get cached values for the sequence length
        cos = self.cos_cached[:seq_len, :].to(q_k.device)  # (seq_len, head_dim//4)
        sin = self.sin_cached[:seq_len, :].to(q_k.device)   # (seq_len, head_dim//4)

        # Reshape sin and cos for broadcasting
        sin = sin.view(1, sin.shape[0], 1, sin.shape[1])  # (1, seq_len, 1, head_dim//4)
        cos = cos.view(1, cos.shape[0], 1, cos.shape[1])  # (1, seq_len, 1, head_dim//4)

        # q_k: Input tensor of shape (batch_size, seq_len, num_heads, head_dim // 2) 
        # sin: Sine tensor of shape (1, seq_len, 1, head_dim//4)
        # cos: Cosine tensor of shape (1, seq_len, 1, head_dim//4)
        
        # Reshape q and k to match the cache dimensions
        q_k_embed = q_k.float()
        # Split channels for rotation
        q_k_rot, q_k_pass = q_k_embed[..., :q_k_embed.shape[-1] // 2], q_k_embed[..., q_k_embed.shape[-1] // 2:]


        # sin: Sine tensor of shape (1, seq_len, 1, head_dim//4)
        # cos: Cosine tensor of shape (1, seq_len, 1, head_dim//4)
        # q_k_rot: (batch_size, seq_len, num_heads, head_dim//4)
        # q_k_pass: (batch_size, seq_len, num_heads, head_dim//4)
        
        # Apply rotation using complex multiplication
        # [cos_θx - sin_θy, sin_θx + cos_θy]
        q_k_rotated = torch.cat(
            [
                q_k_rot * cos - q_k_pass * sin,
                q_k_rot * sin + q_k_pass * cos,
            ],
            dim=-1,
        )

        return q_k_rotated.type_as(q_k)