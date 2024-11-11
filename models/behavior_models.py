from .transformer_models import TransformerEncoder, TransformerDecoder
import torch.nn as nn

class BeXRLSequence(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings):
        super(BeXRLSequence, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_encoder_layers, num_embeddings)
        self.decoder = TransformerDecoder(model_dim, output_dim, num_heads, num_decoder_layers)
    
    def forward(self, src, tgt):
        memory, vq_loss, encidx = self.encoder(src)  # Only the final quantized state and VQ loss from encoder
        output = self.decoder(tgt, memory)  # Decode future states from the quantized encoder output
        return output, vq_loss, encidx  

class BeXRLState(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings):
        super(BeXRLState, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_encoder_layers, num_embeddings)
        self.decoder = TransformerDecoder(model_dim, output_dim, num_heads, num_decoder_layers)
    
    def forward(self, src, tgt):
        memory, vq_loss, encidx = self.encoder(src)  # Only the final quantized state and VQ loss from encoder
        output = self.decoder(tgt, memory)  # Decode future states from the quantized encoder output
        return output, vq_loss, encidx  