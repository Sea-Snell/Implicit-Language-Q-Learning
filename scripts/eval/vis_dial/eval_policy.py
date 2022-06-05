import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import hydra
from omegaconf import DictConfig, OmegaConf
import visdial.load_objects
from evaluate import eval

@hydra.main(config_path="../../../config/vis_dial", config_name="eval_policy")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    eval(cfg)

if __name__ == "__main__":
    main()
