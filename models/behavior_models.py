from .transformer_models import TransformerEncoder, TransformerDecoder, TransformerEncoderPerStates, TransformerDecoderPerState
import torch.nn as nn
import torch

class BeXRLSequence(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings):
        super(BeXRLSequence, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_encoder_layers, num_embeddings)
        self.decoder = TransformerDecoder(model_dim, output_dim, num_heads, num_decoder_layers)
    
    def forward(self, src, tgt):
        memory, vq_loss, encidx = self.encoder(src)  # Only the final quantized state and VQ loss from encoder
        output = self.decoder(tgt, memory)  # Decode future states from the quantized encoder output
        return output, vq_loss, encidx 

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

class BeXRLState(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings):
        super(BeXRLState, self).__init__()
        self.encoder = TransformerEncoderPerStates(input_dim, model_dim, num_heads, num_encoder_layers, num_embeddings)
        self.decoder = TransformerDecoderPerState(model_dim, output_dim, num_heads, num_decoder_layers)
    
    def forward(self, src, tgt):
        memory, vq_loss, encidx = self.encoder(src)  # Only the final quantized state and VQ loss from encoder
        output = self.decoder(tgt, memory)  # Decode future states from the quantized encoder output
        return output, vq_loss, encidx  