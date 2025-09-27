from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl
import torch


# @hydra.main(config_path="./confs", config_name="gala", version_base="1.1")
@hydra.main(config_path="./confs", config_name="peoplesnapshot", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    model = NeRFModel(opt)
    datamodule = hydra.utils.instantiate(opt.dataset)
    trainer = pl.Trainer(accelerator='gpu',
                         **opt.trainer_args)

    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     trainer.fit(model, datamodule=datamodule)
    # prof.export_chrome_trace("/root/trobr/code/p0922/trace.json")
    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint('model.ckpt')


if __name__ == "__main__":
    main()
