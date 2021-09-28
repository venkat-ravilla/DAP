import torch.nn as nn
import torch

class Decoder(nn.Module):
    
    def __init__(self, config, device=torch.device("cpu")):
        
        super(Decoder, self).__init__()
        self.device = device
        
        self.num_classes = config['num_classes']
        # Initial sos token
        #self.sostoken = torch.tensor([[43]]*64, dtype=torch.long, device=device)
        
        # output tag embedding input with vocal=num_classes+sostoken
        self.embedding = nn.Embedding(44, config['decoder_embedding_output'])        
        
        self.gru = nn.GRU(
            input_size=config['decoder_embedding_output'],
            hidden_size=config['decoder_hidden'], 
            num_layers=config['decoder_num_layers'], 
            batch_first=True
        )
        self.classifierinput_1 = config['decoder_hidden']*2
        self.classifierinput_2 = config['decoder_hidden']
        self.classifierinput_3 = self.classifierinput_2/2

        # classifier on top of feature extractor
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=self.classifierinput_1, out_features=self.classifierinput_2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.classifierinput_2, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=self.num_classes)
        ])
        
    
    def forward(self, final_hidden, hidden_seq, batch_size):
        """
            final_hidden.shape = [batch, seq_len, hidden_size]
            hidden_seq.shape = [batch, seq_len, hidden_size]
        """
        
        inputtoken = torch.tensor([[43]]*batch_size, dtype=torch.long, device=self.device)
        outputs = torch.empty((batch_size, 0, 43), device=self.device)
        final_hidden = final_hidden.unsqueeze(0)
        for i, x in enumerate(torch.tensor([i for i in range(10)], device=self.device)):
            embedded = self.embedding(inputtoken)
            #embedded = self.dropout(embedded)
            finaloutput, finalhidden = self.gru(embedded, final_hidden) 
            ctx = torch.cat((finaloutput[:,0,:], hidden_seq[:,i,:]), dim=1)
            output = self.classifier(ctx)
            output_index = torch.argmax(output, dim=1)
            output_index = output_index.unsqueeze(1)
            outputs = torch.cat((outputs,output.unsqueeze(1)), dim=1)
            inputtoken = output_index
            final_hidden = finalhidden
        
        return outputs
