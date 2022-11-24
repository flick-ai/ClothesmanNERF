import os

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg, args

EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def load_network():
    model = create_network()
    ckpt_path = '/home/wenhao/CV/humannerf/experiments/human_nerf/zju_mocap/p387/single_gpu/latest.tar'
    # ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None):
    cfg.perturb = 0.
    # pose = torch.tensor([-0.0126, -0.0251,  0.0797,  0.0303, -0.3528, -0.2926,  0.0100,  0.0100,
    #       0.0100,  0.1286,  0.0133,  0.0096,  0.2766, -0.3499,  0.3004,  0.0105,
    #       0.0099,  0.0101, -0.0161,  0.0043,  0.0115,  0.0684, -0.2980, -0.0491,
    #       0.0104,  0.0100,  0.0100,  0.0100,  0.0101,  0.0100,  0.0100,  0.0100,
    #       0.0100, -0.0222, -0.2791,  0.1108,  0.0101,  0.0100,  0.0097,  0.0099,
    #       0.0100,  0.0100,  0.0268, -0.2397,  0.0183,  0.1582,  0.0569, -0.6302,
    #      -0.0296,  0.7069,  0.0514, -0.0162, -1.2805, -0.9353,  0.0025,  0.6423,
    #      -0.0952,  0.0100,  0.0100,  0.0100,  0.0100,  0.0100,  0.0100,  0.0099,
    #       0.0100,  0.0100,  0.0100,  0.0100,  0.0100]).cuda()
    # dst_Rs = torch.tensor([[[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
    #       [ 0.0000e+00,  1.0000e+00,  0.0000e+00],
    #       [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

    #      [[ 9.9695e-01, -6.9245e-02, -3.5840e-02],
    #       [ 7.0039e-02,  9.9731e-01,  2.1390e-02],
    #       [ 3.4263e-02, -2.3835e-02,  9.9913e-01]],

    #      [[ 8.9047e-01,  2.8781e-01, -3.5246e-01],
    #       [-2.9504e-01,  9.5486e-01,  3.4321e-02],
    #       [ 3.4643e-01,  7.3428e-02,  9.3519e-01]],

    #      [[ 1.0000e+00, -5.1424e-06, -2.9654e-05],
    #       [ 5.1412e-06,  1.0000e+00, -3.9716e-05],
    #       [ 2.9654e-05,  3.9716e-05,  1.0000e+00]],

    #      [[ 9.9999e-01,  6.2279e-04,  3.2905e-03],
    #       [-2.2915e-04,  9.9298e-01, -1.1830e-01],
    #       [-3.3411e-03,  1.1830e-01,  9.9297e-01]],

    #      [[ 8.9557e-01, -3.2365e-01, -3.0527e-01],
    #       [ 2.2997e-01,  9.2413e-01, -3.0510e-01],
    #       [ 3.8086e-01,  2.0304e-01,  9.0206e-01]],

    #      [[ 1.0000e+00, -5.2669e-05, -7.2056e-05],
    #       [ 5.2633e-05,  1.0000e+00, -4.9803e-04],
    #       [ 7.2083e-05,  4.9802e-04,  1.0000e+00]],

    #      [[ 9.9998e-01, -1.4718e-03, -5.7166e-03],
    #       [ 1.6203e-03,  9.9966e-01,  2.6061e-02],
    #       [ 5.6763e-03, -2.6070e-02,  9.9964e-01]],

    #      [[ 9.5122e-01,  4.9189e-02, -3.0455e-01],
    #       [-6.7039e-02,  9.9657e-01, -4.8430e-02],
    #       [ 3.0112e-01,  6.6484e-02,  9.5126e-01]],

    #      [[ 1.0000e+00, -7.0894e-06, -9.2461e-06],
    #       [ 7.0860e-06,  1.0000e+00, -3.6149e-04],
    #       [ 9.2486e-06,  3.6149e-04,  1.0000e+00]],

    #      [[ 1.0000e+00,  1.6575e-06,  6.7634e-05],
    #       [-1.6586e-06,  1.0000e+00,  1.7619e-05],
    #       [-6.7634e-05, -1.7619e-05,  1.0000e+00]],

    #      [[ 1.0000e+00, -2.6192e-06,  8.5180e-06],
    #       [ 2.6192e-06,  1.0000e+00,  1.0215e-06],
    #       [-8.5180e-06, -1.0215e-06,  1.0000e+00]],

    #      [[ 9.5349e-01, -9.4641e-02, -2.8618e-01],
    #       [ 1.0388e-01,  9.9444e-01,  1.7227e-02],
    #       [ 2.8295e-01, -4.6152e-02,  9.5802e-01]],

    #      [[ 1.0000e+00,  2.8721e-04,  3.3768e-05],
    #       [-2.8720e-04,  1.0000e+00, -8.7676e-05],
    #       [-3.3793e-05,  8.7667e-05,  1.0000e+00]],

    #      [[ 1.0000e+00,  4.1444e-05, -4.1181e-05],
    #       [-4.1441e-05,  1.0000e+00,  7.8546e-05],
    #       [ 4.1185e-05, -7.8544e-05,  1.0000e+00]],

    #      [[ 9.6895e-01, -1.0295e-02, -2.4702e-01],
    #       [ 6.1123e-03,  9.9982e-01, -1.7693e-02],
    #       [ 2.4716e-01,  1.5634e-02,  9.6885e-01]],

    #      [[ 8.0131e-01,  5.9824e-01, -2.1927e-03],
    #       [-5.9153e-01,  7.9177e-01, -1.5222e-01],
    #       [-8.9329e-02,  1.2328e-01,  9.8834e-01]],

    #      [[ 7.6609e-01, -5.1372e-02,  6.4067e-01],
    #       [ 2.4901e-02,  9.9842e-01,  5.0282e-02],
    #       [-6.4225e-01, -2.2567e-02,  7.6616e-01]],

    #      [[-2.8801e-02,  6.0419e-01, -7.9631e-01],
    #       [-5.7703e-01,  6.4045e-01,  5.0680e-01],
    #       [ 8.1621e-01,  4.7409e-01,  3.3019e-01]],

    #      [[ 8.0149e-01,  9.5851e-02,  5.9027e-01],
    #       [-1.0044e-01,  9.9462e-01, -2.5132e-02],
    #       [-5.8950e-01, -3.9143e-02,  8.0681e-01]],

    #      [[ 1.0000e+00,  1.5239e-05, -2.3897e-05],
    #       [-1.5238e-05,  1.0000e+00,  2.1453e-05],
    #       [ 2.3897e-05, -2.1452e-05,  1.0000e+00]],

    #      [[ 1.0000e+00, -1.3143e-05, -1.1876e-05],
    #       [ 1.3143e-05,  1.0000e+00,  2.2362e-05],
    #       [ 1.1875e-05, -2.2362e-05,  1.0000e+00]],

    #      [[ 1.0000e+00, -2.1657e-05, -9.0978e-06],
    #       [ 2.1658e-05,  1.0000e+00,  5.4617e-05],
    #       [ 9.0966e-06, -5.4617e-05,  1.0000e+00]],

    #      [[ 1.0000e+00, -3.6781e-05, -1.3559e-05],
    #       [ 3.6781e-05,  1.0000e+00,  3.3286e-06],
    #       [ 1.3559e-05, -3.3291e-06,  1.0000e+00]]]).cuda()
    # dst_Ts = torch.tensor([[-1.7419e-03, -2.2306e-01,  2.9133e-02],
    #      [ 6.9915e-02, -9.0854e-02, -6.4179e-03],
    #      [-6.8082e-02, -9.0024e-02, -3.7824e-03],
    #      [-2.4838e-03,  1.0803e-01, -2.6848e-02],
    #      [ 3.2440e-02, -3.6617e-01, -5.2150e-03],
    #      [-3.6504e-02, -3.7359e-01, -9.1712e-03],
    #      [ 5.2442e-03,  1.3388e-01,  3.3602e-04],
    #      [-1.3178e-02, -3.8776e-01, -4.3516e-02],
    #      [ 1.5284e-02, -3.8827e-01, -4.2211e-02],
    #      [ 1.5624e-03,  5.2418e-02,  2.6165e-02],
    #      [ 2.5399e-02, -5.4661e-02,  1.1743e-01],
    #      [-2.4706e-02, -4.7028e-02,  1.2133e-01],
    #      [-2.5790e-03,  2.1211e-01, -4.5001e-02],
    #      [ 7.8646e-02,  1.2092e-01, -3.4625e-02],
    #      [-8.1403e-02,  1.1813e-01, -3.9274e-02],
    #      [ 4.8052e-03,  6.3714e-02,  5.0699e-02],
    #      [ 9.0610e-02,  2.9985e-02, -9.0147e-03],
    #      [-9.5450e-02,  3.1860e-02, -9.6813e-03],
    #      [ 2.5414e-01, -1.1567e-02, -2.7337e-02],
    #      [-2.4857e-01, -1.2153e-02, -2.1210e-02],
    #      [ 2.4382e-01,  8.6154e-03, -8.6047e-04],
    #      [-2.4971e-01,  7.5363e-03, -5.0088e-03],
    #      [ 8.2796e-02, -8.2258e-03, -1.4568e-02],
    #      [-8.3381e-02, -6.1825e-03, -1.0227e-02]]).cuda()
    model = load_network()
    test_loader = create_dataloader(data_type)
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)
    print(f"save to {os.path.join(cfg.logdir, cfg.load_net)}")
    model.eval()
    for batch in tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v[0]
        
        # batch['dst_posevec'] = pose
        # batch['dst_Ts'] = dst_Ts
        # batch['dst_Rs'] = dst_Rs

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out)

    writer.finalize()


def run_freeview():
    _freeview(
        data_type='freeview',
        folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net+"1"),
        exp_name=render_folder_name)

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=f"{idx:06d}")
    
    writer.finalize()

        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
