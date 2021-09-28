import torch.nn as nn
import torch

class PersonaGRU(nn.Module):
    
    def __init__(self, config, device=torch.device("cpu")):
        
        super(PersonaGRU, self).__init__()
        # vector size of persona layer is double the utterance hidden size due to bidirectional GRU        
        self.persona_input = 2*config["utterance_hidden"]
        
        self.device = device
        
        # GRU for detecting persona info 
        self.gru = nn.GRU(
            input_size=self.persona_input,
            hidden_size=config["persona_hidden"], 
            num_layers=config["persona_num_layers"], 
            bidirectional=True,
            batch_first=True
        )
        
        # initial hidden_states
        self.hx = torch.randn((2, 1, config["utterance_hidden"]), device=self.device)
        
    
    def forward(self, batch_input, batch_sequences):
        """
            batch_input.shape = [batch, seq_len, hidden_size]
            batch_sequences.shape = [batch, seq_len, persona_input]
        """
        
        batch_sequencemask_fr = batch_input['sequencemask_fr']
        batch_sequencemask_bk = batch_input['sequencemask_bk']
        hx = self.hx.to("cuda")
        # create an empty feature vector 
        batch_personafeatures = torch.empty((0, 10, 2*self.persona_input), device=self.device)      
        
        for seq, fr_mask, bk_mask in zip(batch_sequences, batch_sequencemask_fr, batch_sequencemask_bk): #seq
            personafeatures_forward = torch.empty((0, self.persona_input), device=self.device)
            personafeatures_backword = torch.empty((0, self.persona_input), device=self.device)
            seq_tmp = seq.unsqueeze(0)
            seq_rev = seq_tmp.flip([0,1])
            backseq = seq_rev.squeeze()
            hidden = hx
            for sent, fr_hid_mask in zip(seq, fr_mask):
                sent = sent.unsqueeze(0)
                sent = sent.unsqueeze(0)
                outputs, hidden = self.gru(sent, torch.where(fr_hid_mask==1, hidden, hx))
                personafeatures_forward = torch.cat((personafeatures_forward, outputs.squeeze(0)), dim=0)
            
            hidden = hx
            for sent, bk_hid_mask in zip(backseq, bk_mask):
                sent = sent.unsqueeze(0)
                sent = sent.unsqueeze(0)
                outputs, hidden = self.gru(sent, torch.where(bk_hid_mask==1, hidden, hx))
                personafeatures_backword = torch.cat((personafeatures_backword, outputs.squeeze(0)), dim=0)
                
            personafeatures_backword = personafeatures_backword.unsqueeze(0).flip([0,1])
            personafeatures = torch.cat((personafeatures_forward, personafeatures_backword.squeeze(0)), dim=1)
            personafeatures.size()
            batch_personafeatures = torch.cat((batch_personafeatures, personafeatures.unsqueeze(0)), dim=0)
            
        return batch_personafeatures
