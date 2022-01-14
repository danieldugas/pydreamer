import os
import time
import numpy as np
import pandas as pd
from strictfire import StrictFire

def main(envname="navrep3daslfixedenv",
         scenario="navrep3dasl",
         mlruns_dir="~/aws_mlruns/mlruns",
         ):
    print("")
    print("ASSUMING SCENARIO IS {}".format(scenario))
    print("")

    mlruns_dir = os.path.expanduser(mlruns_dir)
    runs_parent = os.path.join(mlruns_dir, "0")
    maybe_runs = os.listdir(runs_parent)

    runs = []
    for run in maybe_runs:
        rundir = os.path.join(runs_parent, run)
        if not os.path.isdir(rundir):
            continue
        logpath = os.path.join(rundir, "metrics/agent/scenario")
        S = pd.read_csv(logpath, sep=" ", names=["wall_time", "num_walls", "total_steps"])
        runs.append(run)

    print("Found {} runs:".format(len(runs)))
    for run in runs:
        print(run)

    time.sleep(5.)

    for run in runs:
        rundir = os.path.join(runs_parent, run)
        if not os.path.isdir(rundir):
            continue
        logpath = os.path.join(rundir, "metrics/agent/scenario")
        print("Reading {}".format(logpath))
        S = pd.read_csv(logpath, sep=" ", names=["wall_time", "num_walls", "total_steps"])

        outpath = os.path.expanduser("~/navrep3d/logs/gym/navrep3daslfixedenv_2022_01_01__01_01_01_DISCRETE_DREAMER_E2E_VCARCH_C1024.csv") # noqa

        S = pd.read_csv(logpath, sep=" ", names=["wall_time", "num_walls", "total_steps"])
        S["scenario"] = scenario
        S["damage"] = np.nan
        S["steps"] = np.nan
        S["goal_reached"] = np.nan
        S["reward"] = np.nan
        S["num_agents"] = S["num_walls"]

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
        outpath = outpath.replace("2022_01_01__01_01_01", time_string)

        S.to_csv(outpath)
        print("{} written.".format(outpath))


if __name__ == "__main__":
    StrictFire(main)
