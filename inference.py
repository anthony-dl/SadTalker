from glob import glob
import shutil
import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from time import perf_counter


def rename_file_incrementally(file_path):
    base_name, extension = os.path.splitext(file_path)
    if not os.path.exists(file_path):
        return file_path

    index = 1
    while True:
        new_file_path = f"{base_name}_{index}{extension}"
        if not os.path.exists(new_file_path):
            return new_file_path
        index += 1


class SadTalker:
    def __init__(
        self,
        size=256,
        checkpoint_dir="./checkpoints",
        use_cpu=False,
        old_version=False,
        preprocess="crop",
        verbose=False,
    ):
        """
        size: the image size of the facerender
        batch_size: the batch size of facerender
        expression_scale: the batch size of facerender
        pose_style: input pose style from [0, 46)
        preprocess: "crop", "extcrop", "resize", "full", "extfull"
        old_version: use the pth other than safetensor version
        verbose: saving the intermedia output or not
        """
        self.preprocess = preprocess
        self.size = size
        self.verbose = verbose

        if torch.cuda.is_available() and not use_cpu:
            self.device = "cuda"
            # should add info logging here
        else:
            self.device = "cpu"
            print("using CPU for inference")  # add warning logging here

        current_root_path = os.path.split(sys.argv[0])[0]

        sadtalker_paths = init_path(
            checkpoint_dir,
            os.path.join(current_root_path, "src/config"),
            self.size,
            old_version,
            self.preprocess,
        )
        print(f"sadtalker path is {sadtalker_paths}")

        # init model
        self.preprocess_model = CropAndExtract(sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, self.device)

    def __call__(
        self,
        audio_path="./examples/driven_audio/bus_chinese.wav",
        pic_path="./examples/source_image/full_body_1.png",
        batch_size=8,
        enhancer="gfpgan",
        pose_style=0,
        expression_scale=1.0,
        result_dir="./results",
        ref_eyeblink=None,
        ref_pose=None,
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        background_enhancer=None,
        still=False,
    ):
        """
        ref_eyeblink=None: path to reference video providing eye blinking
        ref_pose: path to reference video providing pose
        pose_style: input pose style from [0, 46)
        input_yaw: the input yaw degree of the user
        input_pitch: the input pitch degree of the user
        input_roll: the input roll degree of the user
        enhancer: two options: gfpgan, RestoreFormer
        background_enhancer: background enhancer, [realesrgan]
        face3dvis: generate 3d face and 3d landmarks
        still: can crop back to the original videos for the full body aniamtion
        """

        save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)
        pose_style = pose_style

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        print("3DMM Extraction for source image")
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            pic_path,
            first_frame_dir,
            self.preprocess,
            source_image_flag=True,
            pic_size=self.size,
        )
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[
                0
            ]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing eye blinking")
            ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(
                ref_eyeblink,
                ref_eyeblink_frame_dir,
                self.preprocess,
                source_image_flag=False,
            )
        else:
            ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print("3DMM Extraction for the reference video providing pose")
                ref_pose_coeff_path, _, _ = self.preprocess_model.generate(
                    ref_pose,
                    ref_pose_frame_dir,
                    self.preprocess,
                    source_image_flag=False,
                )
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(
            first_coeff_path,
            audio_path,
            self.device,
            ref_eyeblink_coeff_path,
            still=still,
        )
        coeff_path = self.audio_to_coeff.generate(
            batch, save_dir, pose_style, ref_pose_coeff_path
        )

        # coeff2video
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            input_yaw_list,
            input_pitch_list,
            input_roll_list,
            expression_scale=expression_scale,
            still_mode=still,
            preprocess=self.preprocess,
            size=self.size,
        )

        result = self.animate_from_coeff.generate(
            data,
            save_dir,
            pic_path,
            crop_info,
            enhancer=enhancer,
            background_enhancer=background_enhancer,
            preprocess=self.preprocess,
            img_size=self.size,
        )

        print(f"result is {result}")
        image_name = os.path.splitext(os.path.basename(pic_path))[0]
        print(f"image name is {image_name}")
        # shutil.move(result, save_dir+'.mp4') args.result_dir
        saved_video_path = os.path.join(result_dir, image_name + ".mp4")
        saved_video_path = rename_file_incrementally(saved_video_path)
        shutil.move(result, saved_video_path)
        print("The generated video is named:", saved_video_path)

        if not self.verbose:
            shutil.rmtree(save_dir)


sad_talker = SadTalker()
sad_talker(
    audio_path="./examples/driven_audio/bus_chinese.wav",
    pic_path="./examples/source_image/full_body_1.png",
)
