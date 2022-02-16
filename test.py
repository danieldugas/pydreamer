import os
import argparse
from tqdm import tqdm
from strictfire import StrictFire
from pyniel.python_tools.path_tools import make_dir_if_not_exists
import numpy as np
import torch
from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscrete

from pydreamer.preprocessing import to_image
from pydreamer import tools
from pydreamer.models import Dreamer

from convert_dreamer_metrics_to_n3d import find_runs

torch.distributions.Distribution.set_default_validate_args(False)
torch.backends.cudnn.benchmark = True  # type: ignore

def main(gpu=False, build_name="./alternate.x86_64", render=True, difficulty_mode="medium",
         run_id=None, n_episodes=1000):
    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    confnames = ["defaults", "navrep3dtrain"]
    for name in confnames:
        conf.update(configs[name])
    # Dict to namepsace
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    conf = parser.parse_args([])

    if gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    if conf.model == 'dreamer':
        model = Dreamer(conf)
    else:
        assert False, conf.model
    model.to(device)
    print(model)

    if run_id is None:
#         run_id = "f3f47a18b9334a4baa97c728143a00c6" # "./alternate.x86_64"
        run_id = "0657e4d7a0f14c6ea301017f6774402b" # "./alternate.x86_64"
#         run_id = "a1ec5269279f46f79af2884526590592" # "staticasl" (staticaslfixed)
#         run_id = "3aaa8d09bce64dd888240a04b714aec7" # "kozehd" (kozehdrs)
    print("Selecting run_id: " + run_id)

    runs = find_runs()
    found = False
    for run, rundir, run_env, run_len in runs:
        if run_id == run:
            found = True
            break
    assert found, f"Run {run_id} not found"

    optimizers = model.init_optimizers(conf.adam_lr, conf.adam_lr_actor, conf.adam_lr_critic, conf.adam_eps)
#     resume_step = tools.mlflow_load_checkpoint(model, optimizers, map_location=device, run_id=run_id)
    modelpath = os.path.join(rundir, "artifacts/checkpoints/latest.pt")
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    for i, opt in enumerate(optimizers):
        opt.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])
    resume_step = checkpoint['epoch']
    print(f'Loaded model from checkpoint epoch {resume_step}')

    last_action = np.array([1, 0, 0])

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

    successes = []
    difficulties = []
    pbar = tqdm(range(n_episodes))
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
                difficulty = inf["episode_scenario"]
                difficulties.append(difficulty)
                pbar.set_description("Success rate: {:.2f}, avg dif: {:.2f}".format(
                    sum(successes)/len(successes), np.mean(difficulties)))
                break

    bname = build_name.replace(".x86_64", "").replace("./", "")
    SAVEPATH = "~/navrep3d/test/{}_{}_DREAMER".format(run_env, run_id) + "_{}_{}_{}.npz".format(
        bname, difficulty_mode, n_episodes)
    SAVEPATH = os.path.expanduser(SAVEPATH)
    make_dir_if_not_exists(os.path.dirname(SAVEPATH))
    np.savez(SAVEPATH, successes=np.array(successes), difficulties=np.array(difficulties))
    print("Saved to {}".format(SAVEPATH))


if __name__ == '__main__':
    StrictFire(main)
