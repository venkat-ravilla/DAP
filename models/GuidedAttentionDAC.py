import torch.nn as nn
import torch
from models.Encoder import Encoder
from models.Decoder import Decoder

class GuidedAttentionDAC(nn.Module):
    
    def __init__(self, config, device=torch.device("cpu")):
        
        super(GuidedAttentionDAC, self).__init__()
        
        self.device = device
        
        # encoder model
        self.encoder = Encoder(config, device)
        # decoder model
        self.decoder = Decoder(config, device)
        
    
    def forward(self, batch, batch_size):
        """
            batch.shape = [batch, conv_window, seq_len]
        """
        
        encoded_hidden, encoder_output = self.encoder(batch)
        encoder_output = torch.cat((encoder_output[-2,:,:],encoder_output[-1,:,:]), dim=1)
        logits = self.decoder(encoder_output, encoded_hidden, batch_size)
        
        return logits
          