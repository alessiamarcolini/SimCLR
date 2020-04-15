print("partito")
import yaml

from data_aug.dataset_wrapper import DataSetWrapper
from simclr import SimCLR

print("ciao dal main")
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
print("ok yaml")
dataset = DataSetWrapper(config["batch_size"], **config["dataset"])
print("ok dataset")
simclr = SimCLR(dataset, config)
simclr.train()
