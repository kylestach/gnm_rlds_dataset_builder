from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle
import os
from PIL import Image
import tqdm

IMAGE_SIZE = (128, 128)

class GNMDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.4")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.1": "Changed file structure to make separate TFDS datasets for each original dataset.",
        "1.0.2": "Split datasets by builder config",
        "1.0.3": "Increase resolution",
        "1.0.4": "Use JPEG",
    }

    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="recon"),
        tfds.core.BuilderConfig(name="go_stanford"),
        tfds.core.BuilderConfig(name="scand"),
        tfds.core.BuilderConfig(name="cory_hall"),
        tfds.core.BuilderConfig(name="sacson"),
        tfds.core.BuilderConfig(name="seattle"),
        tfds.core.BuilderConfig(name="tartan_drive"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        # compute Kona language embedding
        self.language_instruction = "Navigate to the goal."
        self.language_embedding = self._embed([self.language_instruction])[0].numpy()

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(3,),
                                        dtype=np.float64,
                                        doc="Robot state, consists of [2x position, 1x yaw]",
                                    ),
                                    "position": tfds.features.Tensor(
                                        shape=(2,),
                                        dtype=np.float64,
                                        doc="Robot position",
                                    ),
                                    "yaw": tfds.features.Tensor(
                                        shape=(1,),
                                        dtype=np.float64,
                                        doc="Robot yaw",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(2,),
                                dtype=np.float64,
                                doc="Robot action, consists of 2x position",
                            ),
                            "action_angle": tfds.features.Tensor(
                                shape=(3,),
                                dtype=np.float64,
                                doc="Robot action, consists of 2x position, 1x yaw",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float64,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float64,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        gnm_root_dir = "/home/kstachowicz/data/gnm_dataset"
        dataset_name = self.builder_config.name

        return {
            "train": self._generate_examples(
                dataset_path=os.path.join(gnm_root_dir, dataset_name),
                dataset_name=dataset_name,
            )
        }

    def _generate_examples(
        self, dataset_path, dataset_name
    ) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _yaw_rotmat(yaw: float) -> np.ndarray:
            return np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )

        def _to_local_coords(
            positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
        ) -> np.ndarray:
            """
            Convert positions to local coordinates

            Args:
                positions (np.ndarray): positions to convert
                curr_pos (np.ndarray): current position
                curr_yaw (float): current yaw
            Returns:
                np.ndarray: positions in local coordinates
            """
            rotmat = _yaw_rotmat(curr_yaw)
            if positions.shape[-1] == 2:
                rotmat = rotmat[:2, :2]
            elif positions.shape[-1] == 3:
                pass
            else:
                raise ValueError

            return (positions - curr_pos).dot(rotmat)

        def _process_image(path, mode="stretch"):
            img = Image.open(path)
            if mode == "stretch":
                img = img.resize(IMAGE_SIZE)
            elif mode == "crop":
                width = int(85/64*IMAGE_SIZE[0] + 0.5)
                height = IMAGE_SIZE[1]
                img = img.resize((width, height))

                top = 0
                bottom = height
                left = (width - height) // 2
                right = (width + height) // 2
                img = img.crop((left, top, right, bottom))

            return np.asarray(img, dtype="uint8")

        def _compute_actions(
            traj_data,
            curr_time,
            goal_time,
            len_traj_pred=1,
            waypoint_spacing=1,
            learn_angle=True,
            normalize=False,
        ):
            start_index = curr_time
            end_index = curr_time + len_traj_pred * waypoint_spacing + 1
            yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing].astype(
                np.float64
            )
            positions = traj_data["position"][
                start_index:end_index:waypoint_spacing
            ].astype(np.float64)
            goal_pos = traj_data["position"][
                min(goal_time, len(traj_data["position"]) - 1)
            ].astype(np.float64)

            if len(yaw.shape) == 2:
                yaw = yaw.squeeze(1)

            if yaw.shape != (len_traj_pred + 1,):
                const_len = len_traj_pred + 1 - yaw.shape[0]
                yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
                positions = np.concatenate(
                    [positions, np.repeat(positions[-1][None], const_len, axis=0)],
                    axis=0,
                )

            assert yaw.shape == (
                len_traj_pred + 1,
            ), f"{yaw.shape} and {(len_traj_pred + 1,)} should be equal"
            assert positions.shape == (
                len_traj_pred + 1,
                2,
            ), f"{positions.shape} and {(len_traj_pred + 1, 2)} should be equal"

            waypoints = _to_local_coords(positions, positions[0], yaw[0])
            goal_pos = _to_local_coords(goal_pos, positions[0], yaw[0])

            assert waypoints.shape == (
                len_traj_pred + 1,
                2,
            ), f"{waypoints.shape} and {(len_traj_pred + 1, 2)} should be equal"

            if learn_angle:
                yaw = yaw[1:] - yaw[0]
                actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
            else:
                actions = waypoints[1:]

            if normalize:
                actions[:, :2] /= (
                    self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )
                goal_pos /= (
                    self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                )

            return actions, goal_pos

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data_path = os.path.join(episode_path, "traj_data.pkl")
            data = np.load(
                data_path, allow_pickle=True
            )  # this is a list of dicts in our case
            data = {k: v.astype(np.float64) for k, v in data.items()}

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data["position"])):
                # Get image observation
                image_path = f"{i}.jpg"
                img = _process_image(
                    os.path.join(episode_path, image_path), mode="stretch"
                )

                # Get state observation
                position = data["position"][i]
                yaw = data["yaw"][i].reshape(-1)
                state = np.concatenate((position, yaw))

                # Recover action(s)
                action, goal_pos = _compute_actions(
                    data,
                    i,
                    i + 1,
                    len_traj_pred=1,
                    waypoint_spacing=1,
                    learn_angle=False,
                    normalize=False,
                )
                action_angle, goal_pos = _compute_actions(
                    data,
                    i,
                    i + 1,
                    len_traj_pred=1,
                    waypoint_spacing=1,
                    learn_angle=True,
                    normalize=False,
                )
                action = action[0]
                action_angle = action_angle[0]

                episode.append(
                    {
                        "observation": {
                            "image": img,
                            "state": state,
                            "position": position,
                            "yaw": yaw,
                        },
                        "action": action,
                        "action_angle": action_angle,
                        "discount": 1.0,
                        "reward": float(i == (len(data["position"]) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(data["position"]) - 1),
                        "is_terminal": i == (len(data["position"]) - 1),
                        "language_instruction": self.language_instruction,
                        "language_embedding": self.language_embedding,
                    }
                )

            # create output data sample
            dataset_key = os.path.normpath(episode_path)
            dataset_key = os.path.join(*dataset_key.split(os.sep)[-2:])
            sample = {"steps": episode, "episode_metadata": {"file_path": dataset_key}}

            success = len(data["position"]) >= 1

            # if you want to skip an example for whatever reason, simply return None
            return dataset_key, sample, success

        def _get_folder_names(data_dir):
            folder_names = [
                f
                for f in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, f))
                and "traj_data.pkl" in os.listdir(os.path.join(data_dir, f))
            ]
            return folder_names

        episode_paths = _get_folder_names(dataset_path)
        episode_paths = [os.path.join(dataset_path, sample) for sample in episode_paths]

        def _process(path):
            episode_path, sample, success = _parse_example(path)
            if not success:
                return None
            return episode_path, sample

        # for smallish datasets, use single-thread parsing
        use_beam = False
        if use_beam:
            # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
            import apache_beam as beam

            return beam.Create(episode_paths) | beam.Map(_process)
        else:
            for path in episode_paths:
                result = _process(path)
                if result is not None:
                    yield result
