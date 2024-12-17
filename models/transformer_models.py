# Positional Encoding
import torch
import torch.nn as nn 
from utils import generate_causal_mask
from .vector_quantizers import EMAVectorQuantizer, EnhancedEMAVectorQuantizer

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
        self.vq_layer = EnhancedEMAVectorQuantizer(num_embeddings, model_dim)

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
        quantized, vq_loss, encidx = self.vq_layer(x[:,-1,:])
        
        return quantized, vq_loss, encidx 



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
        self.vq_layer = EnhancedEMAVectorQuantizer(num_embeddings, model_dim)

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

class TransformerEncoderAtari(nn.Module):
    def __init__(self, input_channels, action_dim, model_dim, num_heads, num_layers, num_embeddings, frame_size=(84, 84)):
        super(TransformerEncoderAtari, self).__init__()

        # CNN for processing Atari frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_output_dim = 64 * (frame_size[0] // 8) * (frame_size[1] // 8)

        # Linear layers to embed states and actions
        self.state_embedding = nn.Linear(conv_output_dim, model_dim)
        self.action_embedding = nn.Embedding(action_dim, model_dim)

        # Positional encoding with sinusoidal encoding
        self.positional_encoding = PositionalEncoding(model_dim)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, model_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Quantization layer
        self.vq_layer = EMAVectorQuantizer(num_embeddings, model_dim)

    def forward(self, states, actions):
        batch_size, seq_len, channels, height, width = states.size()

        # Process state frames through CNN
        states = states.view(-1, channels, height, width)
        states = self.conv(states)
        states = states.view(batch_size, seq_len, -1)  # Flatten to [batch, seq_len, conv_output_dim]
        
        # Embedding states and actions
        state_embeds = self.state_embedding(states)  # Shape: [batch, seq_len, model_dim]
        action_embeds = self.action_embedding(actions.long())  # Shape: [batch, seq_len, model_dim]

        # Combine state and action embeddings
        embeddings = state_embeds + action_embeds

        # Add positional encodings
        pos_embeds = self.positional_encoding(seq_len)
        embeddings += pos_embeds

        # Append EOS embedding
        eos_embedding_expanded = self.eos_embedding.expand(batch_size, -1, -1)
        embeddings = torch.cat([embeddings, eos_embedding_expanded], dim=1)

        # Generate and apply causal mask
        causal_mask = generate_causal_mask(embeddings.size(1)).to(embeddings.device)
        transformer_output = self.transformer_encoder(embeddings.permute(1, 0, 2), mask=causal_mask).permute(1, 0, 2)

        # Quantize final output
        quantized, vq_loss, encidx = self.vq_layer(transformer_output[:, -1, :])
        return quantized, vq_loss, encidx


class TransformerDecoderAtari(nn.Module):
    def __init__(self, model_dim, output_channels, num_heads, num_layers, frame_size=(84, 84)):
        super(TransformerDecoderAtari, self).__init__()
        
        # CNN layers for initial state processing before transformer
        self.conv = nn.Sequential(
            nn.Conv2d(output_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_output_dim = 64 * (frame_size[0] // 8) * (frame_size[1] // 8)

        # Embedding layer for the processed state vector
        self.state_embedding = nn.Linear(conv_output_dim, model_dim)
        
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Reconstruction layers using Upsample and Conv2d
        self.reconstruction = nn.Sequential(
            nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False),  # Upsample 1
            nn.Conv2d(model_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upsample 2
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=6, mode='bilinear', align_corners=False),  # Upsample 3
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()  # Sigmoid to constrain pixel values (assuming normalization)
        )

    def forward(self, tgt_states, memory):
        batch_size, num_frames, channels, height, width = tgt_states.size()

        # Process target states through initial CNN
        tgt_states = tgt_states.view(-1, channels, height, width)
        tgt_states = self.conv(tgt_states)
        tgt_states = tgt_states.view(batch_size, num_frames, -1)
        
        # Embed processed state vector and apply positional encoding
        tgt = self.state_embedding(tgt_states) + self.positional_encoding(num_frames)

        # Apply causal mask to transformer decoder
        causal_mask = generate_causal_mask(num_frames).to(tgt.device)
        tgt = self.transformer_decoder(tgt.permute(1, 0, 2), memory.unsqueeze(0), tgt_mask=causal_mask)

        # Reshape to [batch_size * num_frames, model_dim, 1, 1] for reconstruction
        tgt = tgt.permute(1, 0, 2).reshape(batch_size * num_frames, -1, 1, 1)
        output = self.reconstruction(tgt)

        # Reshape back to [batch_size, num_frames, channels, height, width]
        output = output.view(batch_size, num_frames, channels, height, width)
        return output
