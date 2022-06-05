import argparse
import pickle as pkl
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    args = parser.parse_args()

    with open(args.eval_file, 'rb') as f:
        d = pkl.load(f)
    

    rs = [sum(map(lambda x: x[2], item[1])) for item in d['eval_dump']]
    mean_r = np.mean(rs)
    std_r = np.std(rs)
    st_err_r = std_r / np.sqrt(len(rs))
    print(d['config'])
    print(f'{mean_r} +- {st_err_r}')

