import os
import time
import numpy as np
import pandas as pd
from strictfire import StrictFire

def find_runs(mlruns_dir="~/aws_mlruns/mlruns"):
    mlruns_dir = os.path.expanduser(mlruns_dir)
    runs_parent = os.path.join(mlruns_dir, "0")
    maybe_runs = os.listdir(runs_parent)

    runs = []
    for run in maybe_runs:
        rundir = os.path.join(runs_parent, run)
        if not os.path.isdir(rundir):
            continue
        logpath = os.path.join(rundir, "metrics/agent/scenario")
        envidpath = os.path.join(rundir, "params/env_id")
        # test that run is loadable
        S = pd.read_csv(logpath, sep=" ", names=["wall_time", "num_walls", "total_steps"])
        run_len = len(S["num_walls"])
        env_id_df = pd.read_csv(envidpath)
        env_id = env_id_df.columns[0]
        runs.append((run, rundir, env_id, run_len))

    print("Found {} runs:".format(len(runs)))
    for run, rundir, env_id, run_len in runs:
        print((run, env_id, run_len))

    return runs

def main(mlruns_dir="~/aws_mlruns/mlruns",
         dry_run=False,
         ):

    runs = find_runs(mlruns_dir=mlruns_dir)

    for run, rundir, env_id, run_len in runs:
        if not os.path.isdir(rundir):
            continue
        logpath = os.path.join(rundir, "metrics/agent/scenario")
        print("Reading {}".format(logpath))
        S = pd.read_csv(logpath, sep=" ", names=["wall_time", "num_walls", "total_steps"])

        # read environment
        envidpath = os.path.join(rundir, "params/env_id")
        env_id_df = pd.read_csv(envidpath)
        env_id = env_id_df.columns[0]
        if env_id == "NavRep3DStaticASLEnv":
            envname = "navrep3daslfixedenv"
            scenario = "navrep3dasl"
        elif env_id == "NavRep3DKozeHDEnv":
            envname = "navrep3dkozehdrenv"
            scenario = "navrep3dkozehd"
        elif env_id == "NavRep3DKozeHDRSEnv":
            envname = "navrep3dkozehdrsenv"
            scenario = "navrep3dkozehd"
        elif env_id == "NavRep3DTrainEnv":
            envname = "navrep3daltenv"
            scenario = "navrep3dalt"
        else:
            raise NotImplementedError

        S = pd.read_csv(logpath, sep=" ", names=["wall_time", "num_walls", "total_steps"])
        S["scenario"] = scenario
        S["damage"] = np.nan
        S["steps"] = np.nan
        S["goal_reached"] = np.nan
        S["reward"] = np.nan
        S["num_agents"] = S["num_walls"]
        if env_id == "NavRep3DTrainEnv":
            S["num_walls"] = S["num_agents"] * 2. - 2.

        # post-processing
        S["wall_time"] = S["wall_time"] / 1000.
        if False: # why are step values not integers? why are reward values integers?
            logpath2 = os.path.expanduser("~/aws_mlruns/mlruns/0/c88db86657a646d180028f28c55f3887/metrics/agent/return") # noqa
            logpath3 = os.path.expanduser("~/aws_mlruns/mlruns/0/c88db86657a646d180028f28c55f3887/metrics/agent/episode_length") # noqa
            S2 = pd.read_csv(logpath2, sep=" ", names=["wall_time", "reward", "total_steps"])
            S3 = pd.read_csv(logpath3, sep=" ", names=["wall_time", "episode_length", "total_steps"])
            S["reward"] = S2["reward"]
            S["steps"] = S3["episode_length"]

        print(S)

        # fix log start time
        start_time = S["wall_time"][0]
        time_string = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime(start_time))
        outpath = "~/navdreams_data/results/logs/gym/{}_{}_DISCRETE_DREAMER_E2E_VCARCH_C1024.csv".format(
            envname, time_string
        )
        outpath = os.path.expanduser(outpath)

        if len(S["wall_time"]) == 0:
            print("EMPTY!")
        else:
            if not dry_run:
                S.to_csv(outpath)
            print("{} {} written.".format(outpath, "would have been" if dry_run else ""))


if __name__ == "__main__":
    StrictFire(main)
