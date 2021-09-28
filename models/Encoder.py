from models.PersonaGRU import PersonaGRU
import torch.nn as nn
import torch
from models.UtteranceGRU import UtteranceGRU

class Encoder(nn.Module):
    
    def __init__(self, config, device=torch.device("cpu")):
        
        super(Encoder, self).__init__()
        
        self.persona_input = 2*config['utterance_hidden']

        self.conversation_input = 2*self.persona_input
        
        self.device = device
        
        # utterance class instance
        self.utterance_rnn = UtteranceGRU(config)
                
        # persona level class instance
        self.persona = PersonaGRU(config, device)

        self.gru = nn.GRU(
            input_size=self.conversation_input,
            hidden_size=config["conversation_hidden"], 
            num_layers=config["conversation_num_layers"], 
            bidirectional=True,
            batch_first=True
        )
                
    
    def forward(self, batch):
        """
            x.shape = [batch, seq_len, hidden_size]
        """
        batch_inputid = batch['inputid']
        
        # Utterance layer embedding
        batch_sequences = torch.empty((0, 10, self.persona_input), device=self.device)
        for idx, inputid in enumerate(batch_inputid):
            seq_of_utterance = self.utterance_rnn(input_ids=inputid)
            batch_sequences = torch.cat((batch_sequences, seq_of_utterance.unsqueeze(0)), dim=0)
        
        # providing utterance layer inputs to persona layer
        output = self.persona(batch, batch_sequences)

        # conversational level embedding  
        encoded_hidden, encoder_output = self.gru(output)
        
        return encoded_hidden, encoder_output
          
