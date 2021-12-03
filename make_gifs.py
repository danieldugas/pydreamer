from typing import Dict
import tempfile
from pathlib import Path
import numpy as np
from mlflow.tracking import MlflowClient

FPS = 10
B, T = 5, 50


def download_artifact_npz(run_id, artifact_path) -> Dict[str, np.ndarray]:
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, artifact_path, tmpdir)
        with Path(path).open('rb') as f:
            data = np.load(f)
            return {k: data[k] for k in data.keys()}  # type: ignore


def encode_gif(frames, fps):
    # Copyright Danijar
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())  # type: ignore
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def make_gif(env_name, run_id, step, fps=FPS):
    dest_path = f'/tmp/dream_{env_name}_{step}.gif'
    artifact = f'd2_wm_dream/{step}.npz'
    data = download_artifact_npz(run_id, artifact)
    img = data['image_pred']
    img = img[:B, :T].reshape((-1, 64, 64, 3))
    gif = encode_gif(img, fps)
    with Path(dest_path).open('wb') as f:
        f.write(gif)
    print(dest_path)
    print(img.shape)


if __name__ == "__main__":
    for i in range(100)[::-1]:
        try:
            make_gif('navrep3dtrain', '15a44ae71c9b4fcfb862eeb7c05a46b2', "{:04}".format(i) + '001')
        except Exception as e:
            print(e)
