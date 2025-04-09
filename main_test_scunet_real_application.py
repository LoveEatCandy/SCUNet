import os.path
import logging
import argparse
import math

import numpy as np

import torch

from utils import utils_logger
from utils import utils_image as util


"""
python3 main_test_scunet_real_application.py --model_name scunet_color_real_psnr --testset_name anime
"""


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="scunet_color_real_psnr",
        help="scunet_color_real_psnr, scunet_color_real_gan",
    )
    parser.add_argument(
        "--testset_name", type=str, default="real3", help="test set, bsd68 | set12"
    )
    parser.add_argument("--show_img", type=bool, default=False, help="show the image")
    parser.add_argument(
        "--model_zoo", type=str, default="model_zoo", help="path of model_zoo"
    )
    parser.add_argument(
        "--testsets", type=str, default="testsets", help="path of testing folder"
    )
    parser.add_argument(
        "--results", type=str, default="results", help="path of results"
    )

    args = parser.parse_args()

    n_channels = 3

    result_name = args.testset_name + "_" + args.model_name  # fixed
    model_path = os.path.join(args.model_zoo, args.model_name + ".pth")

    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = os.path.join(
        args.testsets, args.testset_name
    )  # L_path, for Low-quality images
    E_path = os.path.join(args.results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(
        logger_name, log_path=os.path.join(E_path, logger_name + ".log")
    )
    logger = logging.getLogger(logger_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_scunet import SCUNet as net

    model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info("Model path: {:s}".format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info("Params number: {}".format(number_parameters))

    logger.info("model_name:{}".format(args.model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters / 10**6))

    for idx, img in enumerate(L_paths):
        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info("{:->4d}--> {:>10s}".format(idx + 1, img_name + ext))

        img_L = util.imread_uint(img, n_channels=n_channels)

        parts = []
        width, height = img_L.shape[0], img_L.shape[1]
        for i in range(math.ceil(width / 512)):
            for j in range(math.ceil(height / 512)):
                part_of_img_L = img_L[
                    i * 512 : (i + 1) * 512, j * 512 : (j + 1) * 512, :
                ].copy()

                part_of_img_L = util.uint2tensor4(part_of_img_L)
                part_of_img_L = part_of_img_L.to(device)

                # ------------------------------------
                # (2) img_E
                # ------------------------------------

                # img_E = utils_model.test_mode(model, img_L, refield=64, min_size=512, mode=2)

                current_img_E = model(part_of_img_L)
                current_img_E = util.tensor2uint(current_img_E)
                parts.append(current_img_E)

        img_E = np.zeros((width, height, n_channels), dtype=np.uint8)
        for i in range(math.ceil(width / 512)):
            for j in range(math.ceil(height / 512)):
                part_of_img_E = parts[i * int(math.ceil(height / 512)) + j]
                img_E[i * 512 : (i + 1) * 512, j * 512 : (j + 1) * 512, :] = (
                    part_of_img_E
                )

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_E, os.path.join(E_path, img_name + ".png"))


if __name__ == "__main__":
    main()
