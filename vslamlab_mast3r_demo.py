import os
import torch
import argparse
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_reconstructed_scene


if __name__ == '__main__':

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--sequence_path", type=str, help="path to image directory")
    parser.add_argument("--calibration_yaml", type=str, help="path to calibration file")
    parser.add_argument("--rgb_txt", type=str, help="path to image list")
    parser.add_argument("--exp_folder", type=str, help="path to save results")
    parser.add_argument("--exp_it", type=str, help="experiment iteration")
    parser.add_argument("--settings_yaml", type=str, help="settings_yaml")
    parser.add_argument("--verbose", type=str, help="verbose")
   
    args, unknown = parser.parse_known_args()
    sequence_path = args.sequence_path
    rgb_txt = args.rgb_txt
    exp_folder = args.exp_folder
    exp_id = args.exp_it
    verbose = bool(int(args.verbose))

    # Verify if PyTorch is compiled with CUDA
    print("\nVerify if PyTorch is compiled with CUDA: ")
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print(output)
    except FileNotFoundError:
        print("    nvcc (NVIDIA CUDA Compiler) not found, CUDA may not be installed.")

    print(f"    Torch with CUDA is available: {torch.cuda.is_available()}")
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    Device capability: {torch.cuda.get_device_capability(0)}")

    # Load image paths and timestamps from the provided text file
    image_list = []
    timestamps = []
    with open(rgb_txt, 'r') as file:
        for line in file:
            timestamp, path, *extra = line.strip().split(' ')
            image_list.append(os.path.join(sequence_path, path))
            timestamps.append(timestamp)

    device = 'cuda'
    # batch_size = 1
    # schedule = 'cosine'
    # lr = 0.01
    # niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    scene_state, outfile = get_reconstructed_scene(
        outdir = os.path.join(exp_folder, 'tmp'),
        gradio_delete_cache=False,
        model=model,
        retrieval_model=None,
        device=device,
        silent=False,        
        image_size=512, 
        current_scene_state=None,
        filelist=image_list,
        optim_level='refine+depth', #
        lr1 = 0.07, #
        niter1= 300, #
        lr2 = 0.01, #
        niter2= 300, #
        min_conf_thr = 1.5, #
        matching_conf_thr = 0.0, #
        as_pointcloud=True,
        mask_sky =False, #    
        clean_depth = True, #        
        transparent_cams=False, #      
        cam_size=0.2,  #
        scenegraph_type="complete", #      
        winsize=1, #         
        win_cyclic=False, #
        refid=0, #
        TSDF_thresh=0., # 
        shared_intrinsics=False, #
    )

    scene = scene_state.sparse_ga
    # rgbimg = scene.imgs
    # focals = scene.get_focals().cpu()

    poses = scene.get_im_poses().cpu()
    keyFrameTrajectory_txt = os.path.join(exp_folder, exp_id.zfill(5) + '_KeyFrameTrajectory' + '.txt')
    with open(keyFrameTrajectory_txt, 'w') as file:
        for i_pose, pose in enumerate(poses):
            tx, ty, tz = pose[0, 3].item(), pose[1, 3].item(), pose[2, 3].item()
            rotation_matrix = np.array([[pose[0, 0].item(), pose[0, 1].item(), pose[0, 2].item()],
                                        [pose[1, 0].item(), pose[1, 1].item(), pose[1, 2].item()],
                                        [pose[2, 0].item(), pose[2, 1].item(), pose[2, 2].item()]])
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            ts = timestamps[i_pose]
            line = str(ts) + " " + str(tx) + " " + str(ty) + " " + str(tz) + " " + str(qx) + " " + str(
                qy) + " " + str(qz) + " " + str(qw) + "\n"
            file.write(line)
   
    # Visualize reconstruction
    scene.show()

    