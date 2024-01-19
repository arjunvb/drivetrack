import argparse
import enum
import functools
import glob
import os
from typing import Any, Callable, Iterable, Tuple, Optional, Dict, List
import warnings
import multiprocessing as mp

import time

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

from tqdm import tqdm
import dask.dataframe as dd
from dask.distributed import Client
import gcsfs
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # or any {'0', '1', '2'}
import tensorflow as tf
from scipy.interpolate import NearestNDInterpolator

from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.camera_image import CameraName
from waymo_open_dataset.v2.perception.lidar import LaserName
from waymo_open_dataset.v2.perception.box import BoxType
from waymo_open_dataset.utils import box_utils, transform_utils

import torch
import torch.multiprocessing as torchmp
import torchvision.transforms.functional as TF

# Poison pill for PyTorch model processes
POISON_PILL = "STOP"


class DepthMethod(enum.Enum):
    NEAREST = "nearest"
    NLSPN = "nlspn"
    COMPLETIONFORMER = "former"


def get_reader(
    context_name: str,
    split: str,
    fs: Optional[gcsfs.GCSFileSystem] = None,
    local_dataset_path: str = None,
) -> Callable[[str, str], dd.DataFrame]:
    if fs is None:  # Read from local file
        assert (
            local_dataset_path is not None
        ), "Must provide local path if reading from local filesystem"

        def read_table(table: str) -> dd.DataFrame:
            paths = os.path.join(
                local_dataset_path,
                split,
                table,
                f"{context_name}.parquet",
            )
            return dd.read_parquet(paths)

    else:  # Read from remote filesystem

        def read_table(table: str) -> dd.DataFrame:
            paths = os.path.join(
                "waymo_open_dataset_v_2_0_0", split, table, f"{context_name}.parquet"
            )
            return dd.read_parquet(paths, filesystem=fs)

    return read_table


def project_points(
    points: npt.NDArray[np.float32],
    camera_calibration: v2.CameraCalibrationComponent,
    width: int,
    height: int,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Project points from the vehicle frame to the image plane.

    Args:
        points (np.ndarray): points with shape [..., N, 3]
        camera_calibration (CameraCalibrationComponent): camera parameters
        width (int): image width
        height (int): image height

    Returns:
        A tuple of 2-d points with shape [..., N, 2] projected to the image plane,
        and a boolean mask indicating points inside the image frame
    """
    # vehicle frame to camera sensor frame.
    extrinsic = camera_calibration.extrinsic.transform.reshape(4, 4)
    vehicle_to_sensor = np.linalg.inv(extrinsic)
    # Points [B, N, 3]
    homo_points = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    point_camera_frame = np.einsum("ij,...nj->...ni", vehicle_to_sensor, homo_points)
    mask = np.full(point_camera_frame.shape[:-1], fill_value=True, dtype=bool)
    mask &= point_camera_frame[..., 0] >= 0
    u_d = -point_camera_frame[..., 1] / point_camera_frame[..., 0]
    v_d = -point_camera_frame[..., 2] / point_camera_frame[..., 0]

    # add distortion model here if you'd like.
    f_u = camera_calibration.intrinsic.f_u
    f_v = camera_calibration.intrinsic.f_v
    c_u = camera_calibration.intrinsic.c_u
    c_v = camera_calibration.intrinsic.c_v
    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    mask &= u_d >= 0
    mask &= u_d < width
    mask &= v_d >= 0
    mask &= v_d < height

    return np.stack([u_d, v_d], axis=-1), mask


def process_annotation(
    data: Tuple[int, pd.Series],
    scene_cache: Dict[str, npt.NDArray[np.float32]],
    output_dir: str,
    version: str = "dev",
    split: str = "training",
    occlusion_factor: float = 0.95,
    distance_filter: float = 50.0,
    video_length_filter: int = 24,
):
    pid = mp.current_process().pid
    tf.config.experimental.set_visible_devices([], "GPU")
    _, row = data

    print(f"Process {pid} processing annotation")

    # Metadata
    segment_context_name = row["key.segment_context_name"]
    camera_name = row["key.camera_name"]
    laser_object_id = row["key.laser_object_id"]
    current_timestamps = np.array(row["key.frame_timestamp_micros"])

    # Calibration components
    camera_calibration = v2.CameraCalibrationComponent.from_dict(row)

    # Get video
    cached_data = scene_cache[(segment_context_name, camera_name)]
    video = cached_data["video"].copy()
    timestamps = cached_data["timestamps"].copy()
    predicted_depths = cached_data["depths"].copy()
    all_points = cached_data["points"]

    object_frames = np.argwhere(
        ((timestamps[:, None] - current_timestamps[None, :]) == 0).any(axis=-1)
    ).flatten()
    video = video[object_frames]
    predicted_depths = predicted_depths[object_frames]
    all_points = [
        points.copy() for i, points in enumerate(all_points) if i in object_frames
    ]
    num_frames, height, width, _ = video.shape

    # Box data
    all_lidar_boxes = v2.LiDARCameraSyncedBoxComponent.from_dict(row)
    all_boxes = v2.LiDARBoxComponent.from_dict(row)
    # TODO: Use speed and acceleration
    speed = all_boxes.speed.numpy.T
    acceleration = all_boxes.acceleration.numpy.T
    object_boxes = all_lidar_boxes.camera_synced_box.numpy(np.float32).T

    ###########
    # FILTERS #
    ###########

    # Short-circuit if no LiDAR box projects to the camera
    box_corners = box_utils.get_upright_3d_box_corners(object_boxes).numpy()
    _, box_mask = project_points(box_corners, camera_calibration, width, height)
    valid_frames: npt.NDArray[np.bool_] = box_mask.any(axis=-1)  # [N]
    is_contiguous = np.all(np.diff(np.argwhere(valid_frames).flatten()) == 1)
    if not valid_frames.any() or not is_contiguous:
        return

    # Short-circuit if the video is too short
    if valid_frames.sum() < video_length_filter:
        return

    # Filter out objects that are far away from the camera
    corners_distance_from_camera = np.linalg.norm(box_corners, axis=-1)
    if not (corners_distance_from_camera <= distance_filter).any():
        return

    ##############
    # TRANSFORMS #
    ##############

    # Pose data
    all_vehicle_pose = v2.VehiclePoseComponent.from_dict(row)
    world_from_vehicle = np.stack(
        [t.reshape(4, 4) for t in all_vehicle_pose.world_from_vehicle.transform]
    )
    vehicle_from_world = np.linalg.inv(world_from_vehicle)
    id_reference = np.eye(4)[None, ...].repeat(num_frames, axis=0)

    # Vehicle bounding box -> world bounding box
    world_boxes = box_utils.transform_box(
        object_boxes[:, None, ...],
        world_from_vehicle,
        id_reference,
    )

    # Bounding box transformations (origin -> world)
    world_box_rotations = transform_utils.get_yaw_rotation(world_boxes[..., 0, -1])
    world_box_transforms = transform_utils.get_transform(
        world_box_rotations, world_boxes[..., 0, :3]
    ).numpy()

    # Bounding boxes at the origin
    origin_boxes = box_utils.transform_box(
        world_boxes,
        id_reference,
        world_box_transforms,
    )

    # Sizing transforms at origin
    # TODO: Maybe don't do this?
    origin_sizings = np.stack(
        [
            np.diag(np.r_[origin_boxes[i, 0, 3:6].numpy(), np.ones(1)])
            for i in range(num_frames)
        ]
    )

    # Compute point cloud correspondences across all frames
    # Steps:
    # 1. Transform points from vehicle frame to world frame
    # 2. Transform points from world frame to origin (0, 0, 0)
    # 3. Transform points from origin to bounding boxes in all world frames, then all vehicle frames
    # 4. Project points to image plane
    all_tracks = []
    all_visibles = []
    all_correspondence_points = []
    # Shrink the bounding boxes by a 100cm so that we get less points from the ground
    shrink_factor = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0])
    for frame, points in enumerate(all_points):
        inside_box = (
            box_utils.is_within_box_3d(
                tf.constant(points, dtype=tf.float32),
                tf.constant(
                    object_boxes[frame, None, :] - shrink_factor,
                    dtype=tf.float32,
                ),
            )
            .numpy()
            .flatten()
        )

        box_points = points[inside_box, ...]

        ###########################
        # Correspondence Tracking #
        ###########################

        # Vehicle frame -> world frame
        world_box_points = (
            world_from_vehicle[frame]
            @ np.r_[box_points.T, np.ones((1, box_points.shape[0]))]
        )

        # TODO: Determine if using box sizes is actually helpful or results in more drift
        # World frame -> origin
        # origin_points = np.linalg.inv(origin_sizings[0]) @ np.linalg.inv(world_box_transforms[0]) @ world_box_points
        origin_points = np.linalg.inv(world_box_transforms[frame]) @ world_box_points

        # Origin -> All vehicle frames
        # correspondence_points = (vehicle_from_world @ world_box_transforms @ origin_sizings @ origin_points).transpose(0, 2, 1)[..., :3]
        correspondence_points = (
            vehicle_from_world @ world_box_transforms @ origin_points
        ).transpose(0, 2, 1)[
            ..., :3
        ]  # [F, N, 3]
        all_correspondence_points.append(correspondence_points.transpose(1, 0, 2))

        # Distance to the camera
        gt_distance = np.linalg.norm(correspondence_points, axis=-1).transpose(1, 0)

        tracks, visibles = project_points(
            correspondence_points, camera_calibration, width, height
        )
        tracks = tracks.transpose(1, 0, 2)
        visibles = visibles.transpose(1, 0)

        ########################
        # Occlusion estimation #
        ########################

        interpolated_distances = []
        for inner_frame in range(num_frames):
            x = tracks[:, inner_frame, 0]  # [N]
            y = tracks[:, inner_frame, 1]
            x_low = np.clip(np.floor(x).astype(np.int32), 0, width - 1)
            x_high = np.clip(np.ceil(x).astype(np.int32), 0, width - 1)
            y_low = np.clip(np.floor(y).astype(np.int32), 0, height - 1)
            y_high = np.clip(np.ceil(y).astype(np.int32), 0, height - 1)

            # Get the 4 closest points to our interpolated point
            z1 = predicted_depths[inner_frame, y_low, x_low]
            z2 = predicted_depths[inner_frame, y_low, x_high]
            z3 = predicted_depths[inner_frame, y_high, x_low]
            z4 = predicted_depths[inner_frame, y_high, x_high]
            z = np.stack((z1, z2, z3, z4), axis=-1)  # [N, 4]
            z_max = np.max(z, axis=-1)
            interpolated_distances.append(z_max)

        interpolated_distances = np.stack(interpolated_distances, axis=-1)  # [N, F]

        # Estimate occlusions
        # If our 3D point is behind the interpolated depth map by a certain amount
        # then we know something is in front of the point and we estimate occlusion
        occluded = interpolated_distances < occlusion_factor * gt_distance  # [N, F]
        visibles &= ~occluded

        # TODO: Consistency Checks HERE!

        all_tracks.append(tracks)
        all_visibles.append(visibles)

    # Build final correspondences for all points
    tracks = np.concatenate(all_tracks, axis=0)
    visibles = np.concatenate(all_visibles, axis=0)
    correspondences = np.concatenate(all_correspondence_points, axis=0)

    # Filter out frames that don't have any LiDAR boxes
    valid_video = video[valid_frames]
    valid_depths = predicted_depths[valid_frames]
    valid_points = visibles[:, valid_frames].any(axis=-1)
    valid_tracks = tracks[:, valid_frames, :][valid_points, ...]
    valid_visibles = visibles[:, valid_frames][valid_points, ...]

    print(f"Process {pid} finished processing annotation")

    data = {
        "video": valid_video,
        "depths": valid_depths,
        "tracks": valid_tracks,
        "visibles": valid_visibles,
        "points3d": correspondences,
    }

    filename = f"waymo_{segment_context_name}_{camera_name}_{laser_object_id}"
    np.savez(
        f"/{output_dir}/{version}/{split}/{filename}.npz",
        **data,
    )

    return


def complete_depth(
    rgb: npt.NDArray[np.uint8],
    sparse_points: npt.NDArray[np.float32],
    queue: torchmp.Queue,
    depth_method: DepthMethod = DepthMethod.NEAREST,
    width: int = 1920,
    height: int = 1280,
):
    if depth_method is DepthMethod.NEAREST:
        interpolator = NearestNDInterpolator(sparse_points[:, :2], sparse_points[:, 2])
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        interpolated_depths = interpolator(X, Y)
        return interpolated_depths

    elif (
        depth_method is DepthMethod.NLSPN
        or depth_method is DepthMethod.COMPLETIONFORMER
    ):
        sparse_depth = np.zeros((height, width), dtype=np.float32)
        sparse_depth[
            sparse_points[:, 1].astype(np.int32),
            sparse_points[:, 0].astype(np.int32),
        ] = sparse_points[:, 2]

        # Automatically pass tensors through PyTorch shared memory
        rgb = TF.to_tensor(rgb)
        depth = TF.to_tensor(sparse_depth)
        result = torch.zeros((height, width), dtype=torch.float32)
        is_complete = torch.scalar_tensor(False, dtype=torch.bool)

        queue.put((rgb, depth, result, is_complete))
        # Wait for model to finish result
        process_complete = False
        start_time = time.time()
        while not process_complete:
            process_complete = is_complete.item()

            if time.time() - start_time > 60:
                raise RuntimeError("Process took too long!")

        # Result stored in buffer - return it
        return result.numpy()
    else:
        raise ValueError(f"Unknown depth method {depth_method.value}")


def process_video(
    data: Tuple[int, pd.Series],
    column_names: List[str],
    depth_method: DepthMethod = DepthMethod.NEAREST,
):
    tf.config.experimental.set_visible_devices([], "GPU")
    _, row = data

    # Metadata
    segment_context_name = row["key.segment_context_name"]
    camera_name = row["key.camera_name"]
    timestamps = np.array(row["key.frame_timestamp_micros"])

    # Calibration components
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)

    # Get video
    images = v2.CameraImageComponent.from_dict(row)
    video = np.stack([tf.image.decode_jpeg(image).numpy() for image in images.image])
    num_frames, height, width, _ = video.shape

    # Explode grouped rows into array of dicts to be read by Waymo components
    col_names = list(
        filter(
            lambda x: any(
                col in x
                for col in [
                    "VehiclePoseComponent",
                    "LiDARPoseComponent",
                    "LiDARComponent",
                    "LiDARCameraProjectionComponent",
                ]
            ),
            column_names,
        )
    )
    key_names = list(filter(lambda x: "key" in x, column_names))
    col_groups = [
        {col_name: row[col_name][i] for col_name in col_names}
        | {key_name: row[key_name] for key_name in key_names}
        for i in range(num_frames)
    ]

    ############################
    # Point Cloud Construction #
    ############################

    # Iterate through LiDAR data for each frame and build point cloud
    all_points: list[npt.NDArray[np.float32]] = []
    depth_maps: list[npt.NDArray[np.float32]] = []
    for frame, inner_row in enumerate(col_groups):
        # TODO:
        # 1. Use second range image?
        # 2. Merge Top LiDAR data with another short-range LiDAR sensor?

        vehicle_pose = v2.VehiclePoseComponent.from_dict(inner_row)
        lidar_pose = v2.LiDARPoseComponent.from_dict(inner_row)
        lidar = v2.LiDARComponent.from_dict(inner_row)
        projection = v2.LiDARCameraProjectionComponent.from_dict(inner_row)

        # Get 3D point cloud
        points = v2.convert_range_image_to_point_cloud(
            lidar.range_image_return1,
            lidar_calibration,
            lidar_pose.range_image_return1,
            vehicle_pose,
        ).numpy()

        # Get 3D -> 2D projection map
        projection_map = v2._lidar_utils.extract_pointwise_camera_projection(
            lidar.range_image_return1,
            projection.range_image_return1,
        ).numpy()

        # Get projections for current camera and build mapping
        # Each point can be projected to two different cameras, so we need to check both
        projection_points1 = projection_map[..., 0] == camera_name
        projection_points2 = projection_map[..., 3] == camera_name
        distances = np.linalg.norm(points, axis=-1)
        image_points = np.concatenate(
            (
                projection_map[projection_points1, 1:3],
                projection_map[projection_points2, 4:6],
            ),
            axis=0,
        )
        depths = np.concatenate(
            (
                distances[projection_points1],
                distances[projection_points2],
            ),
            axis=0,
        )

        depth_points = np.concatenate(
            [image_points, depths[..., None]], axis=-1
        )  # [N, 3]

        all_points.append(points)
        depth_maps.append(depth_points)

    ####################
    # Depth Completion #
    ####################

    depth_fn = functools.partial(
        complete_depth,
        queue=queue,
        depth_method=depth_method,
        width=width,
        height=height,
    )

    start_time = time.time()

    with mp.Pool(32) as pool:
        all_predicted_depths = pool.starmap(depth_fn, zip(video, depth_maps))

    if receivers is not None:
        for receiver in receivers:
            if receiver.poll():
                error = receiver.recv()
                if error:
                    raise RuntimeError("Process had error!")

    print(f"Depth completion took {time.time() - start_time} seconds")

    predicted_depths = np.stack(all_predicted_depths, axis=0)  # [F, H, W]

    return (
        segment_context_name,
        camera_name,
        {
            "video": video,
            "timestamps": timestamps,
            "depths": predicted_depths,
            "points": all_points,
        },
    )


def process_scene(
    context_name: str,
    output_dir: str,
    fs: Optional[gcsfs.GCSFileSystem] = None,
    local_dataset_path: str = None,
    version: str = "dev",
    split: str = "training",
    occlusion_factor: float = 0.95,
    distance_filter: float = 50.0,
    video_length_filter: int = 24,
    depth_method: DepthMethod = DepthMethod.NEAREST,
    num_samples: Optional[int] = None,
) -> Iterable[Tuple[str, str, str, str, Dict[str, npt.NDArray[np.float32]]]]:
    read = get_reader(context_name, split, fs, local_dataset_path)

    # Camera images and parameters
    cam_image_df: dd.DataFrame = read("camera_image")
    camera_calibration_df: dd.DataFrame = read("camera_calibration")

    # LiDAR Data and Parameters
    lidar_df: dd.DataFrame = read("lidar")
    lidar_calibration_df: dd.DataFrame = read("lidar_calibration")
    lidar_pose_df: dd.DataFrame = read("lidar_pose")
    lidar_camera_projection_df: dd.DataFrame = read("lidar_camera_projection")

    # Vehicle pose data
    vehicle_pose_df: dd.DataFrame = read("vehicle_pose")

    # Object box data
    lidar_box_df: dd.DataFrame = v2.merge(
        read("lidar_box"),
        read("lidar_camera_synced_box"),
    )
    # Filter out pedestrians
    # NOTE: Cyclists are considered vehicles, but we probably don't want to include them?
    lidar_box_df = lidar_box_df[
        lidar_box_df["[LiDARBoxComponent].type"] == BoxType.TYPE_VEHICLE.value
    ]

    ################
    # Build Tables #
    ################

    # Merge LiDAR data
    # NOTE: Using top LiDAR data only, which is a mid-range LiDAR with a range of 75m
    merged_lidar_data = functools.reduce(
        lambda left, right: v2.merge(left, right),
        [
            lidar_df,
            lidar_pose_df,
            lidar_camera_projection_df,
        ],
    )
    merged_lidar_data = merged_lidar_data[
        merged_lidar_data["key.laser_name"] == LaserName.TOP.value
    ]
    image_to_lidar_df = functools.reduce(
        lambda left, right: v2.merge(left, right),
        [
            cam_image_df,
            vehicle_pose_df,
            merged_lidar_data,
        ],
    )

    scene_df = (
        image_to_lidar_df.sort_values("key.frame_timestamp_micros")
        .groupby(
            [
                "key.segment_context_name",
                "key.camera_name",
                "key.laser_name",
            ]
        )
        .agg(list)
        .reset_index()
    )
    scene_df = v2.merge(scene_df, lidar_calibration_df)

    scene_df = scene_df[
        (scene_df["key.camera_name"] != CameraName.SIDE_LEFT.value)
        & (scene_df["key.camera_name"] != CameraName.SIDE_RIGHT.value)
    ]

    # Merge camera data
    lidar_box_df["key.camera_name"] = lidar_box_df[
        "[LiDARCameraSyncedBoxComponent].most_visible_camera_name"
    ]
    box_to_merged_data_df = v2.merge(lidar_box_df, vehicle_pose_df)
    annotation_df = (
        box_to_merged_data_df.sort_values("key.frame_timestamp_micros")
        .groupby(
            [
                "key.segment_context_name",
                "key.camera_name",
                "key.laser_object_id",
            ]
        )
        .agg(list)
        .reset_index()
    )
    annotation_df = v2.merge(annotation_df, camera_calibration_df)

    annotation_df = annotation_df[
        (annotation_df["key.camera_name"] != CameraName.SIDE_LEFT.value)
        & (annotation_df["key.camera_name"] != CameraName.SIDE_RIGHT.value)
    ]

    annotation_pdf: pd.DataFrame = annotation_df.compute()
    if num_samples is not None:
        samples = min(num_samples, len(annotation_pdf))
        annotation_pdf = annotation_pdf.sample(n=samples)
    else:
        samples = len(annotation_pdf)

    #################
    # Process Scene #
    #################

    scene_cache = {}

    f = functools.partial(
        process_video,
        column_names=image_to_lidar_df.columns,
        depth_method=depth_method,
    )

    print("Processing video")

    for row in scene_df.iterrows():
        context_scene, camera_name, data = f(row)
        scene_cache[(context_scene, camera_name)] = data

    # TODO: Filtering
    # 1. Filter out frames where the LiDAR box doesn't project to the camera
    # 2. Filter out objects that are far away from the camera

    print("Processing annotations")

    # Iterate over each (video, object_id) pair
    with mp.Manager() as manager:
        scene_cache = manager.dict(scene_cache)

        f = functools.partial(
            process_annotation,
            scene_cache=scene_cache,
            output_dir=output_dir,
            version=version,
            split=split,
            occlusion_factor=occlusion_factor,
            distance_filter=distance_filter,
            video_length_filter=video_length_filter,
        )

        with mp.Pool(samples) as pool:
            # Just trigger computation
            for _ in pool.imap_unordered(f, annotation_pdf.iterrows()):
                pass

    # Cleanup worker
    del scene_df
    del annotation_df
    del annotation_pdf
    del image_to_lidar_df
    del lidar_box_df
    del vehicle_pose_df
    del lidar_camera_projection_df
    del lidar_pose_df
    del lidar_df
    del camera_calibration_df
    del lidar_calibration_df
    del cam_image_df
    del merged_lidar_data
    del scene_cache
    gc.collect()


def main(
    client: Client,
    output_dir: str,
    fs: Optional[gcsfs.GCSFileSystem] = None,
    local_dataset_path: str = None,
    version: str = "dev",
    split: str = "training",
    occlusion_factor: float = 0.95,
    distance_filter: float = 50.0,
    video_length_filter: int = 24,
    depth_method: DepthMethod = DepthMethod.NEAREST,
    num_samples: int = 16,
    gpus: str = "0,1,2,3,4,5",
):
    global queue
    global receivers
    gpu_list = gpus.split(",")
    num_gpus = len(gpu_list)
    if (
        depth_method is DepthMethod.NLSPN
        or depth_method is DepthMethod.COMPLETIONFORMER
    ):
        if depth_method is DepthMethod.NLSPN:
            from nlspn.src.worker import worker
        elif depth_method is DepthMethod.COMPLETIONFORMER:
            from completionformer.src.worker import worker

        # Use a shared queue across depth completion processes
        manager = torchmp.Manager()
        queue = manager.Queue()
        processes: List[torchmp.Process] = []
        receivers = []
        for i in range(num_gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_list[i]
            receiver, sender = torchmp.Pipe(duplex=False)
            p = torchmp.Process(target=worker, args=(queue, sender, env))
            p.start()
            processes.append(p)
            receivers.append(receiver)
    else:
        queue = None
        receivers = None

    try:
        if local_dataset_path is not None:
            filenames = glob.glob(
                os.path.join(local_dataset_path, split, "camera_image", "*.parquet")
            )
        elif fs is not None:
            filenames = fs.glob(
                os.path.join(
                    "waymo_open_dataset_v_2_0_0", split, "camera_image", "*.parquet"
                )
            )
        else:
            raise ValueError("Must provide local path or GCS filesystem")

        context_names = list(
            map(lambda filename: filename.split("/")[-1].split(".")[0], filenames)
        )

        with open(f"{output_dir}/{version}/{split}/_LOG.txt", "r") as f:
            processed_context_names = set(f.read().splitlines())

        context_names = list(set(context_names) - processed_context_names)

        for context_name in tqdm(context_names):
            process_scene(
                context_name=context_name,
                fs=fs,
                local_dataset_path=local_dataset_path,
                output_dir=output_dir,
                version=version,
                split=split,
                occlusion_factor=occlusion_factor,
                distance_filter=distance_filter,
                video_length_filter=video_length_filter,
                depth_method=depth_method,
                num_samples=num_samples,
            )

            with open(f"{output_dir}/{version}/{split}/_LOG.txt", "a") as f:
                f.write(f"{context_name}\n")

            client.restart()
    except Exception as e:
        print(e)
        raise e
    finally:
        if (
            depth_method is DepthMethod.NLSPN
            or depth_method is DepthMethod.COMPLETIONFORMER
        ):
            # Cleanup
            for i in range(num_gpus):
                queue.put(POISON_PILL)

            for receiver in receivers:
                receiver.close()

            for p in processes:
                p.join()
                p.close()


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--local-dataset-path", type=str, default=None)
    parser.add_argument("--use-gcsfs", action="store_true")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--occlusion-factor", type=float, default=0.95)
    parser.add_argument("--distance-filter", type=float, default=50.0)
    parser.add_argument("--video-length-filter", type=int, default=24)
    parser.add_argument("--version", type=str, default="dev")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=None, required=False)
    parser.add_argument(
        "--depth-method",
        type=str,
        default="nearest",
        choices=["nearest", "nlspn", "former"],
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5")
    args = parser.parse_args()

    client = Client(
        threads_per_worker=args.threads_per_worker, n_workers=args.num_workers
    )

    if args.depth_method == "nearest":
        depth_method = DepthMethod.NEAREST
    elif args.depth_method == "nlspn":
        depth_method = DepthMethod.NLSPN
    elif args.depth_method == "former":
        depth_method = DepthMethod.COMPLETIONFORMER
    else:
        raise ValueError(f"Unknown depth method {args.depth_method}")

    fs = gcsfs.GCSFileSystem(token="google_default") if args.use_gcsfs else None

    os.makedirs(f"{args.output_dir}/{args.version}/{args.split}", exist_ok=True)
    # Emulate touch
    with open(f"{args.output_dir}/{args.version}/{args.split}/_LOG.txt", "a") as f:
        os.utime(f"{args.output_dir}/{args.version}/{args.split}/_LOG.txt", None)

    main(
        client,
        fs=fs,
        output_dir=args.output_dir,
        local_dataset_path=args.local_dataset_path,
        version=args.version,
        split=args.split,
        occlusion_factor=args.occlusion_factor,
        distance_filter=args.distance_filter,
        video_length_filter=args.video_length_filter,
        depth_method=depth_method,
        num_samples=args.num_samples,
        gpus=args.gpus,
    )
