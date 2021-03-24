import torch
import argparse

save_dir_docker = 'your log dir'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cross_str", type=str, default="Null")
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor_srnet", type=float, default=0.0)
    parser.add_argument("--dual_ratio", type=float, default=1.0)
    parser.add_argument("--regular_ratio", type=float, default=1.0)
    parser.add_argument("--rank_ratio", type=float, default=1.0)
    parser.add_argument("--ensemble_ce_ratio", type=float, default=1.0)
    parser.add_argument("--super_ce_ratio", type=float, default=1.0)
    parser.add_argument("--low_ce_ratio", type=float, default=1.0)
    parser.add_argument("--aux_ce_ratio", type=float, default=1.0)
    parser.add_argument("--lr_decay_interval", type=int, default=10)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momen", type=float, default=0.9)
    parser.add_argument("--log_path", type=str, default=save_dir_docker)
    parser.add_argument("--theme", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--job_type", type=str, default='S')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import json

    args = get_args()
    print(args.__dict__)
    print(type(args.__dict__))

    with open('./args.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))

