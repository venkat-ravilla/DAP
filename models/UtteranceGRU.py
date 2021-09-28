import torch
import  torch.nn as nn
from transformers import AutoModel


class UtteranceGRU(nn.Module):
    
    def __init__(self, config):
        super(UtteranceGRU, self).__init__()  
        # embedding layer is replaced by pretrained roberta's embedding
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=config["model_name"])
        # freeze the model parameters
        for param in self.base.parameters():
            param.requires_grad = False
        
        # GRU for sentence embedding
        self.gru = nn.GRU(
            input_size=768, 
            hidden_size=config['utterance_hidden'], 
            num_layers=config['utterance_num_layers'], 
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, input_ids, attention_mask):
        """
            input_ids.shape = [seq_sentences, seq_len]
            attention_mask.shape = [seq_sentences, seq_len]
        """
        
        # hidden_states.shape = [batch, max_len, hidden_size]
        word_embeddings, _ = self.base(input_ids, attention_mask) 
                
        # sentence embeddings sentence_embeddings.shape = [bidirectional?2:1, batch, hidden_size]
        word_hidden, sentence_embeddings = self.gru(word_embeddings)
        # Concatenate the sentence embeddings in both the directions of GRU shape = [1, batch, hidden_size*2]
        return torch.cat((sentence_embeddings[-2,:,:],sentence_embeddings[-1,:,:]), dim=1)