# from configs import cfg

import pickle
import sys
from pathlib import Path
import yaml
import os
# from core.utils.log_util import Logger
# from core.data import create_dataloader
# from core.nets import create_network
# from core.train import create_trainer, create_optimizer
import numpy as np

from absl import app
from absl import flags
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387.yaml',
                    'the path of config file')


# train_loader = create_dataloader('train')

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir

# for idx, batch in enumerate(train_loader):
#     print(idx)
#     if idx == 1:
#         for key in batch:
#             print(key, batch[key])
#         break
    

# with open('/home/wenhao/CV/humannerf/dataset/zju_mocap/387/mesh_infos.pkl', 'rb') as f:
#     mesh_infos = pickle.load(f)

#     for frame_name in mesh_infos:
#         print(mesh_infos[frame_name])
#         break


# smpl_params = np.load('/home/wenhao/package/zju-mocap/CoreView_387/me')
with open('tools/prepare_zju_mocap/387.yaml', 'r') as file:
        cfg = yaml.full_load(file)
# cfg = parse_config()
annots = np.load('/home/wenhao/package/zju-mocap/CoreView_387/annots.npy', allow_pickle=True).item()
img_path_frames_views = annots['ims']
select_view = cfg['training_view']
img_paths = np.array([
    np.array(multi_view_paths['ims'])[select_view] \
        for multi_view_paths in img_path_frames_views
])
for idx, ipath in enumerate(img_paths):
    smpl_idx = idx
    smpl_params = np.load(
            os.path.join('/home/wenhao/package/zju-mocap/CoreView_387/new_params', f"{smpl_idx}.npy"),
            allow_pickle=True).item()
    print(smpl_params)
