import torch
import torch.nn as nn
import torch.optim as optim

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell 

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x, hidden, cell):
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(lstm_out) 
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs, hidden, cell = self.decoder(trg, hidden, cell)
        return outputs
