# Positional Encoding
import torch
import torch.nn as nn 
from utils import generate_causal_mask
from .vector_quantizers import EMAVectorQuantizer

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]
    

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_embeddings):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, model_dim))  # Learnable EOS embedding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.vq_layer = EMAVectorQuantizer(num_embeddings, model_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding(seq_len)
        
        # Append EOS embedding at the end of sequence
        eos_embedding_expanded = self.eos_embedding.expand(x.size(0), -1, -1)  # Match batch size
        x = torch.cat([x, eos_embedding_expanded], dim=1)  # Append EOS token to each sequence
        
        # Apply causal mask to encoder
        causal_mask = generate_causal_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x.permute(1, 0, 2), mask=causal_mask).permute(1, 0, 2)
        
        # Apply EMA VQ layer to quantize encoder outputs and get VQ loss
        quantized, vq_loss, encidx = self.vq_layer(x)
        
        return quantized[:, -1, :], vq_loss, encidx 



class TransformerEncoderPerStates(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_embeddings):
        super(TransformerEncoderPerStates, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, model_dim))  # Learnable EOS embedding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.vq_layer = EMAVectorQuantizer(num_embeddings, model_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding(seq_len)
        eos_embedding_expanded = self.eos_embedding.expand(x.size(0), -1, -1)  # Match batch size
        x = torch.cat([x, eos_embedding_expanded], dim=1)  # Append EOS token to each sequence
        causal_mask = generate_causal_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x.permute(1, 0, 2), mask=causal_mask).permute(1, 0, 2)
        quantized, vq_loss, encidx = self.vq_layer(x)
        return quantized, vq_loss, encidx 

class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, tgt, memory):
        seq_len = tgt.size(1)
        tgt = self.embedding(tgt) + self.positional_encoding(seq_len)
        
        # Apply causal mask to decoder
        causal_mask = generate_causal_mask(seq_len).to(tgt.device)
        tgt = self.transformer_decoder(tgt.permute(1, 0, 2), memory.unsqueeze(0), tgt_mask=causal_mask)
        
        return self.fc_out(tgt.permute(1, 0, 2))

class TransformerDecoderPerState(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads, num_layers):
        super(TransformerDecoderPerState, self).__init__()
        self.embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, tgt, memory):
        seq_len = tgt.size(1)
        tgt = self.embedding(tgt) + self.positional_encoding(seq_len)
        
        # Apply causal mask to decoder
        causal_mask = generate_causal_mask(seq_len).to(tgt.device)
        tgt = self.transformer_decoder(tgt.permute(1, 0, 2), memory.permute(1,0,2), tgt_mask=causal_mask)
        
        return self.fc_out(tgt.permute(1, 0, 2))

    def generate_sequence(encoder, decoder, input_seq, max_length):
        memory, vq_loss, _ = encoder(input_seq)
        generated_seq = []
        next_input = memory.unsqueeze(1)  # Start with the end-of-sequence embedding from the encoder

        for _ in range(max_length):
            output = decoder(next_input, memory.unsqueeze(0))  # Shape: (batch_size, 1, output_dim)
            next_token = output[:, -1, :]  # Shape: (batch_size, output_dim)
            generated_seq.append(next_token)
            next_input = next_token.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)
        generated_seq = torch.cat(generated_seq, dim=1)  # Shape: (batch_size, max_length, output_dim)
        return generated_seq, vq_loss
