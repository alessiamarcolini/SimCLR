print('ciao importing')
import yaml

from data_aug.dataset_wrapper import DataSetWrapper
from simclr import SimCLR
print('ciao imported')

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(
        config["dataset_dir"], config["batch_size"], **config["dataset"]
    )
    simclr = SimCLR(dataset, config)
    simclr.train()
    simclr.extract_features()


if __name__ == "__main__":
    print('ciao da main')
    main()
