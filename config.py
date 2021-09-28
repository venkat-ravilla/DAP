import torch
import os

config = {
    
    # data 
    "data_dir":os.path.join(os.getcwd(), 'data'),
    "dataset":"swda",
    "text_field":"Text",
    "label_field":"DamslActTag",

    "max_len":20,
    "batch_size":128,
    "num_workers":4,
    
    # model
    "model_name":"roberta-base", #roberta-base
    "utterance_hidden":128, 
    "utterance_num_layers":1,
    "persona_hidden":128, 
    "persona_num_layers":1,
    "conversation_hidden": 256, 
    "conversation_num_layers":1,
    "decoder_embedding_output":64,
    "decoder_hidden":512, 
    "decoder_num_layers":1, 
    "num_classes":43, # there are 43 classes in switchboard corpus
    
    # training
    "save_dir":"./",
    "project":"dialogue-act-classification",
    "run_name":"guiding-attention-seq2seq-dac",
    "lr":1e-2,
    "monitor":"val_accuracy",
    "min_delta":0.001,
    "dirpath":"./checkpoints/",
    "filename":"{epoch}-{val_accuracy:4f}",
    "precision":32,
    "average":"micro",
    "epochs":10,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "restart":False,
    "restart_checkpoint":"./checkpoints/epoch=9-step=6569.ckpt"
    
}
