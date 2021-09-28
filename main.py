from config import config
from Trainer import LightningModel
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
import os


if __name__=="__main__":

    # create the checkpoints dir
    path = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.isdir(path):
        os.mkdir(path)

        
    logger = WandbLogger(
        name="guiding-attention",
        save_dir=config["save_dir"],
        project=config["project"],
        log_model=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=5,
    )
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=config["dirpath"],
        every_n_epochs=1
    )

    model = LightningModel(config=config)

    if config['restart'] and config['restart_checkpoint']:
        trainer = pl.Trainer(
            resume_from_checkpoint=config['restart_checkpoint'],
            logger=logger,
            gpus=[0,1],
            gradient_clip_val=5.0,
            checkpoint_callback=True,
            callbacks=[model_checkpoint_callback],
            default_root_dir="./models/",
            max_epochs=config["epochs"],
            precision=config["precision"],
            #distributed_backend="ddp2",
            accelerator="ddp",
            plugins=DDPPlugin(find_unused_parameters=False)
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            gpus=[0,1],
            gradient_clip_val=5.0,
            checkpoint_callback=True,
            callbacks=[model_checkpoint_callback],
            default_root_dir="./models/",
            max_epochs=config["epochs"],
            precision=config["precision"],
            #distributed_backend="ddp2",
            accelerator="ddp",
            plugins=DDPPlugin(find_unused_parameters=False)
        )

        
    trainer.fit(model)
    
    trainer.test(model)

    trainer.save_checkpoint("final.ckpt")
