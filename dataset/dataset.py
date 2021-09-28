from torch.utils.data import Dataset
import torch

class DADataset(Dataset):
    
    __label_dict = dict()
    
    def __init__(self, tokenizer, data, text_field = "clean_text", label_field="act_label_1", max_len=20):
        
        self.text = list(data[text_field]) #data['train'][text_field]
        self.acts = list(data[label_field]) #['train'][label_field]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.index_label_map = {}
        
        
        # build/update the label dictionary 
        classes = sorted(set([item for row in self.acts for item in row]))
        
        for cls in classes:
            if cls not in DADataset.__label_dict.keys():
                DADataset.__label_dict[cls]=len(DADataset.__label_dict.keys())
        
        for key in DADataset.__label_dict:
            index = DADataset.__label_dict[key]
            self.index_label_map[index] = key
    
    def __len__(self):
        return len(self.text)
    
    def label_dict(self):
        return DADataset.__label_dict
    
    def tokenize(self, input):
        input_encoding = self.tokenizer.encode_plus(
            text=input,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            padding="max_length",
        )
        return input_encoding

    
    def __getitem__(self, index):
        
        text = self.text[index]
        #act = self.acts[index]
        label = [DADataset.__label_dict[act] for act in self.acts[index]]
        inputid = []
        attention = []
        sequencemask_fr = []
        sequencemask_bk = []
        for persona in text:
            start = True
            for sent in persona:
                tkr = self.tokenize(sent)
                inputid.append(tkr['input_ids'])
                attention.append(tkr['attention_mask'])
                sequencemask_bk.append(1)
                if start:
                    sequencemask_fr.append(0)
                    start = False
                else:
                    sequencemask_fr.append(1)
            sequencemask_bk[len(sequencemask_bk)-1]=0

        return {
            "sequencemask_fr": torch.tensor(sequencemask_fr),
            "sequencemask_bk": torch.tensor(sequencemask_bk),
            "inputid":torch.tensor(inputid),
            "attention":torch.tensor(attention),
            "label":torch.tensor(label, dtype=torch.long),
        }
