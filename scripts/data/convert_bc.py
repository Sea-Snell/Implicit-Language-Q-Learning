import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    state_dict = torch.load(args.load, map_location=torch.device('cpu'))
    for k in list(state_dict.keys()):
        if 'model.' in k:
            if k.startswith('model'):
                state_dict['lm_policy'+k[len('model'):]] = state_dict.pop(k)
    torch.save(state_dict, args.save)


