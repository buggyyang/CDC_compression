import os
import tarfile
from cv2 import DRAW_MATCHES_FLAGS_DEFAULT
import numpy as np

from .misc_data_util import transforms as trans
from .misc_data_util.url_save import save
from zipfile import ZipFile


def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config["data_path"]  # path to data directory
    if data_path is not None:
        assert os.path.exists(data_path), "Data path {} not found.".format(data_path)

    # the name of the dataset to load
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case
    train = val = None
    if dataset_name == "kth_actions":
        if not os.path.exists(os.path.join(data_path, "kth_actions")):
            os.makedirs(os.path.join(data_path, "kth_actions"))
        if not os.path.exists(os.path.join(data_path, "kth_actions", "train")):
            print("Downloading KTH Actions dataset...")
            actions = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
            for action in actions:
                print("Downloading " + action + "...")
                # if not os.path.exists(os.path.join(data_path, 'kth_actions', action + '.zip')):
                save(
                    "http://www.nada.kth.se/cvap/actions/" + action + ".zip",
                    os.path.join(data_path, "kth_actions", action + ".zip"),
                )
                print("\n")
            print("Done.")

            print("Unzipping KTH Actions dataset...")
            for action in actions:
                print("Unzipping " + action + "...")
                zip_ref = ZipFile(os.path.join(data_path, "kth_actions", action + ".zip"), "r")
                os.makedirs(os.path.join(data_path, "kth_actions", action))
                zip_ref.extractall(os.path.join(data_path, "kth_actions", action))
                zip_ref.close()
                os.remove(os.path.join(data_path, "kth_actions", action + ".zip"))
            print("Done.")

            print("Processing KTH Actions dataset...")
            from .misc_data_util.convert_kth_actions import convert

            convert(os.path.join(data_path, "kth_actions"))
            import shutil

            for action in actions:
                shutil.rmtree(os.path.join(data_path, "kth_actions", action))
            print("Done.")

        from .datasets import KTHActions

        train_transforms = []
        if data_config["img_hz_flip"]:
            train_transforms.append(trans.RandomHorizontalFlip())
        transforms = [
            trans.Resize(data_config["img_size"]),
            trans.RandomSequenceCrop(data_config["sequence_length"]),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        val_transforms = [
            trans.Resize(data_config["img_size"]),
            trans.FixedSequenceCrop(data_config["sequence_length"], 12),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        train_trans = trans.Compose(train_transforms + transforms)
        val_trans = trans.Compose(val_transforms)
        test_trans = trans.Compose(
            [trans.Resize(data_config["img_size"]), trans.ImageToTensor(), trans.ConcatSequence()]
        )
        train = KTHActions(
            os.path.join(data_path, "kth_actions", "train"),
            train_trans,
            add_noise=data_config["add_noise"],
        )
        val = KTHActions(
            os.path.join(data_path, "kth_actions", "val"),
            val_trans,
            add_noise=data_config["add_noise"],
        )
        test = KTHActions(
            os.path.join(data_path, "kth_actions", "test"),
            test_trans,
            add_noise=data_config["add_noise"],
        )

    elif dataset_name == "bair_robot_pushing":
        if not os.path.exists(os.path.join(data_path, "bair_robot_pushing")):
            os.makedirs(os.path.join(data_path, "bair_robot_pushing"))

        if not os.path.exists(os.path.join(data_path, "bair_robot_pushing", "train")):
            print("Downloading BAIR Robot Pushing dataset...")
            save(
                "http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar",
                os.path.join(data_path, "bair_robot_pushing", "bair_robot_pushing_dataset_v0.tar"),
            )
            print("Done.")

            print("Untarring BAIR Robot Pushing dataset...")
            tar = tarfile.open(
                os.path.join(data_path, "bair_robot_pushing", "bair_robot_pushing_dataset_v0.tar")
            )
            tar.extractall(os.path.join(data_path, "bair_robot_pushing"))
            tar.close()
            os.remove(
                os.path.join(data_path, "bair_robot_pushing", "bair_robot_pushing_dataset_v0.tar")
            )
            print("Done.")

            print("Converting TF records...")
            from .misc_data_util.convert_bair import convert

            convert(os.path.join(data_path, "bair_robot_pushing"))
            import shutil

            shutil.rmtree(os.path.join(data_path, "bair_robot_pushing", "softmotion30_44k"))
            print("Done.")

        from .datasets import BAIRRobotPushing

        train_transforms = []
        if data_config["img_hz_flip"]:
            train_transforms.append(trans.RandomHorizontalFlip())
        transforms = [
            trans.Resize(data_config["img_size"]),
            trans.RandomSequenceCrop(data_config["sequence_length"]),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        val_transforms = [
            trans.Resize(data_config["img_size"]),
            trans.FixedSequenceCrop(data_config["sequence_length"]),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        train_trans = trans.Compose(train_transforms + transforms)
        test_trans = trans.Compose(val_transforms)
        train = BAIRRobotPushing(
            os.path.join(data_path, "bair_robot_pushing", "train"),
            train_trans,
            data_config["add_noise"],
        )
        val = BAIRRobotPushing(
            os.path.join(data_path, "bair_robot_pushing", "test"),
            test_trans,
            data_config["add_noise"],
        )
        # import ipdb
        # ipdb.set_trace()

    elif dataset_name == "moving_mnist":
        if not os.path.exists(os.path.join(data_path, "moving_mnist")):
            os.makedirs(os.path.join(data_path, "moving_mnist"))

        if not os.path.exists(os.path.join(data_path, "moving_mnist", "train")):
            print("Downloading Moving MNIST dataset...")
            save(
                "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
                os.path.join(data_path, "moving_mnist", "mnist_test_seq.npy"),
            )
            data = np.load(os.path.join(data_path, "moving_mnist", "mnist_test_seq.npy"))
            train_data = data[:, :9990, ...]
            val_data = data[:, 9990:, ...]
            os.makedirs(os.path.join(data_path, "moving_mnist", "train"))
            np.save(os.path.join(data_path, "moving_mnist", "train", "data.npy"), train_data)
            os.makedirs(os.path.join(data_path, "moving_mnist", "val"))
            np.save(os.path.join(data_path, "moving_mnist", "val", "data.npy"), val_data)
            os.remove(os.path.join(data_path, "moving_mnist", "mnist_test_seq.npy"))

        from .datasets import MovingMNIST

        train_transforms = []
        transforms = [
            trans.RandomSequenceCrop(data_config["sequence_length"]),
            trans.ToTensor(),
            trans.Normalize(0.0, 255.0),
        ]
        val_transforms = [
            trans.FixedSequenceCrop(data_config["sequence_length"]),
            trans.ToTensor(),
            trans.Normalize(0.0, 255.0),
        ]
        train_trans = trans.Compose(train_transforms + transforms)
        test_trans = trans.Compose(val_transforms)
        train = MovingMNIST(
            os.path.join(data_path, "moving_mnist", "train", "data.npy"),
            train_trans,
            data_config["add_noise"],
        )
        val = MovingMNIST(
            os.path.join(data_path, "moving_mnist", "val", "data.npy"),
            test_trans,
            data_config["add_noise"],
        )

    elif dataset_name == "audi":
        from .datasets import AUDI

        transforms = [
            trans.CentercropList(data_config["img_size"]),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = AUDI(
            os.path.join(data_path, "audi"), data_config["sequence_length"], True, transforms
        )
        val = AUDI(
            os.path.join(data_path, "audi"), data_config["sequence_length"], False, transforms
        )

    elif dataset_name == "city":
        from .datasets import CITY

        transforms = [
            trans.VFResize(data_config["img_size"]),
            trans.CentercropList(data_config["img_size"]),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = CITY(
            os.path.join(data_path, "city"), data_config["sequence_length"], True, transforms
        )
        val = CITY(
            os.path.join(data_path, "city"), data_config["sequence_length"], False, transforms
        )
    elif dataset_name == "simu":
        from .datasets import Simulation

        train = Simulation(
            os.path.join(data_path, "simulation", "vortex.npy"),
            data_config["sequence_length"],
            True,
            data_config["img_size"],
        )
        val = Simulation(
            os.path.join(data_path, "simulation", "vortex.npy"),
            data_config["sequence_length"],
            False,
            data_config["img_size"],
        )
    elif dataset_name == "vimeo":
        from .datasets import VIMEO

        transforms = [
            trans.RandomCrop(data_config["img_size"], False),
            trans.RandomSequenceCrop(data_config["sequence_length"]),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
            #   torchvision.transforms.RandomCrop(data_config['img_size'])
        ]
        transforms = trans.Compose(transforms)
        train = VIMEO(os.path.join(data_path, "vimeo_septuplet"), True, transforms)
        val = VIMEO(os.path.join(data_path, "vimeo_septuplet"), False, transforms)
    elif dataset_name == "kodak":
        from .datasets import IMG
        transforms = [
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "Kodak_rh"), transforms)
        val = IMG(os.path.join(data_path, "Kodak_rh"), transforms)
    elif dataset_name == "tecnick":
        from .datasets import IMG
        transforms = [
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "Tecnick"), transforms)
        val = IMG(os.path.join(data_path, "Tecnick"), transforms)
    elif dataset_name == "div2k":
        from .datasets import IMG
        transforms = [
            trans.VFResize(768),
            trans.CentercropList(768),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "DIV2K_valid_HR"), transforms)
        val = IMG(os.path.join(data_path, "DIV2K_valid_HR"), transforms)
    elif dataset_name == "coco":
        from .datasets import IMG
        transforms = [
            trans.RandomCrop(data_config["img_size"], False),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "coco2017/proc/train2017"), transforms)
        val = IMG(os.path.join(data_path, "coco2017/proc/val2017"), transforms)
    elif dataset_name == "cocotest":
        from .datasets import IMG
        transforms = [
            trans.VFResize(384),
            trans.CentercropList(384),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "coco2017/raw/test2017_larger_than_512x512"), transforms)
        val = IMG(os.path.join(data_path, "coco2017/raw/test2017_larger_than_512x512"), transforms)
    elif dataset_name == "anime":
        from .datasets import IMG
        transforms = [
            trans.VFResize(256),
            trans.CentercropList(256),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "anime_faces"), transforms)
        val = IMG(os.path.join(data_path, "anime_faces"), transforms)
    elif dataset_name == "surrealism":
        from .datasets import IMG
        transforms = [
            trans.VFResize(256),
            trans.CentercropList(256),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "artbench/surrealism_lmdb/output"), transforms)
        val = IMG(os.path.join(data_path, "artbench/surrealism_lmdb/output"), transforms)
    elif dataset_name == "expressionism":
        from .datasets import IMG
        transforms = [
            trans.VFResize(256),
            trans.CentercropList(256),
            trans.ImageToTensor(),
            trans.ConcatSequence(),
        ]
        transforms = trans.Compose(transforms)
        train = IMG(os.path.join(data_path, "artbench/expressionism_lmdb/output"), transforms)
        val = IMG(os.path.join(data_path, "artbench/expressionism_lmdb/output"), transforms)
    else:
        raise Exception("Dataset name not found.")

    return train, val
