import argparse
import logging
import logging.config
import os
import sys
import time
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from itertools import chain
from logging import critical, debug, error, info, warning
from multiprocessing import Process
from pathlib import Path
from typing import Iterator, Optional

import mlflow
import numpy as np
import scipy.special
import torch
import torch.distributions as D
import torch.nn as nn
from torch import Tensor, tensor
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader

import generator
from pydreamer.preprocessing import to_image
from pydreamer import tools
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from pydreamer.models import *
from pydreamer.models.functions import map_structure, nanmean
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from pydreamer.tools import *

from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscrete

torch.distributions.Distribution.set_default_validate_args(False)
torch.backends.cudnn.benchmark = True  # type: ignore


def run(conf):
#     mlflow_start_or_resume(conf.run_name or conf.resume_id, conf.resume_id)
#     try:
#         mlflow.log_params({k: v for k, v in vars(conf).items() if not len(repr(v)) > 250})  # filter too long
#     except Exception as e:
#         # This happens when resuming and config has different parameters - it's fine
#         error(f'ERROR in mlflow.log_params: {repr(e)}')

    device = torch.device(conf.device)
    device = torch.device('cpu')

    # MODEL

    if conf.model == 'dreamer':
        model = Dreamer(conf)
    else:
        assert False, conf.model
    model.to(device)

    print(model)

    # Training

    print(conf.run_name)
    print(conf.resume_id)
    run_id = conf.resume_id
    if conf.resume_id is None:
#         run_id = "f3f47a18b9334a4baa97c728143a00c6" # "./alternate.x86_64"
#         run_id = "0657e4d7a0f14c6ea301017f6774402b" # "./alternate.x86_64"
#         run_id = "a1ec5269279f46f79af2884526590592" # "staticasl" (fixed)
        run_id = "3aaa8d09bce64dd888240a04b714aec7" # "kozehd" (kozehdrs)
    print("run_id: " + run_id)

    optimizers = model.init_optimizers(conf.adam_lr, conf.adam_lr_actor, conf.adam_lr_critic, conf.adam_eps)
    resume_step = tools.mlflow_load_checkpoint(model, optimizers, map_location=device, run_id=run_id)
    info(f'Loaded model from checkpoint epoch {resume_step}')

    last_action = np.array([1, 0, 0])

#     build_name = "./alternate.x86_64"
    build_name = "kozehd"
#     difficulty_mode = "progressive"
    difficulty_mode = "easiest"
    env = NavRep3DAnyEnvDiscrete(build_name=build_name,
                                 debug_export_every_n_episodes=0,
                                 difficulty_mode=difficulty_mode)

    A = 3
    CH = 3
    H = 64
    W = 64
    V = 5
    T = 1
    B = 1

    render = True
    successes = []
    N = 1000
    pbar = tqdm(range(N))
    for i in pbar:
        (img, vecobs) = env.reset() # ((64, 64, 3) [0-255], (5,) [-inf, inf])
        rnn_state = model.init_state(B)
        while True:
            obs = {}
            obs["action"] = torch.tensor(last_action, dtype=torch.float).view((T, B, A))
            obs["terminal"] = torch.tensor(0, dtype=torch.bool).view((T, B))
            obs["reset"] = torch.tensor(0, dtype=torch.bool).view((T, B))
            obs["image"] = torch.tensor(to_image(img.reshape((T, B, H, W, CH))), dtype=torch.float).view(
                (T, B, CH, H, W))
            obs["vecobs"] = torch.tensor(vecobs, dtype=torch.float).view((T, B, V))
            obs["reward"] = torch.tensor(0, dtype=torch.float).view((T, B))

            action_distr, rnn_state, metrics = model.forward(obs, rnn_state)

            deter_action = np.argmax(action_distr.mean.view((A,)).detach().numpy())
            deter_action = np.argmax(action_distr.sample().view((A,)).detach().numpy())
            last_action[:] = 0
            last_action[deter_action] = 1

            (img, vecobs), reward, done, inf = env.step(deter_action)

            if render:
                env.render(save_to_file=True)
            if done:
                if reward > 50.:
                    if render:
                        print("Success!")
                    successes.append(1.)
                else:
                    if render:
                        print("Failure.")
                    successes.append(0.)
                pbar.set_description(f"Success rate: {sum(successes)/len(successes):.2f}")
                break


if __name__ == '__main__':
    configure_logging(prefix='[TRAIN]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    conf = parser.parse_args(remaining)

    run(conf)
