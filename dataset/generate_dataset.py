import numpy as np
import pandas as pd
from pqdm.threads import pqdm
from rich.console import Console
from tqdm import tqdm


def aumgented_vectors(fall: bool, frames_each_video: int, pose_vector_dim: int) -> None:
    """
    this function will generate dummy pose data in form of 128-D vector against each frame
    of some arbitrary video

    Parameters
    ----------
    fall
    frames_each_video
    pose_vector_dim

    Returns
    -------

    """
    per_video_vector = np.random.randn(frames_each_video, pose_vector_dim)
    if fall == True:
        return 10.0 + per_video_vector
    if fall == False:
        return -10.0 + per_video_vector


def sample_dataset(n_videos: int, frames_each_video: int, pose_vector_dim: int):
    """
    creates a sample dataset of n_videos with frames_each_video and pose_vector_dim
    Parameters
    ----------
    n_videos
    frames_each_video
    pose_vector_dim

    Returns
    -------

    """
    split = int(n_videos / 2)
    fall_videos = [[True, frames_each_video, pose_vector_dim]] * split
    non_fall_videos = [[False, frames_each_video, pose_vector_dim]] * split
    Console().print(f"🔴 [fall videos] [red]{len(fall_videos)}")
    Console().print(f"🔴 [non-fall videos] [red]{len(non_fall_videos)}")
    Console().print(fall_videos)
    fall = pqdm(
        fall_videos,
        aumgented_vectors,
        n_jobs=8,
        desc=f"Generating dataset [fall]",
        argument_type="args",
    )

    not_fall = pqdm(
        non_fall_videos,
        aumgented_vectors,
        n_jobs=8,
        desc=f"Generating dataset [not-fall]",
        argument_type="args",
    )

    fall = np.stack(fall)
    not_fall = np.stack(not_fall)
    columns = ["video", "frame_no"] + ["pose_vector_idx_" + str(i) for i in range(pose_vector_dim)] + ["Label_fall"]
    stacked_data = np.vstack([fall, not_fall])
    Console().print(stacked_data.shape)
    all_data = []
    for k in tqdm(range(stacked_data.shape[0])):
        current_video_frames = stacked_data[k]
        videos_col = np.repeat(k, current_video_frames.shape[0]).reshape(-1, 1)
        frame_col = np.arange(current_video_frames.shape[0]).reshape(-1, 1)

        this_video = np.hstack([videos_col, frame_col, current_video_frames])
        all_data.append(this_video)
    Console().print(f"🔥 all_data [red]{np.array(all_data).shape}")
    all_data = np.vstack(all_data)
    labels = np.vstack([np.repeat(np.ones(fall.shape[0]), frames_each_video), np.repeat(np.zeros(not_fall.shape[0]), frames_each_video)]).reshape(-1, 1)
    Console().print(f"all data {all_data.shape}\t\t labels {labels.shape}")
    pd.DataFrame(np.hstack([all_data, labels]), columns=columns).to_csv("dataset.csv", index=False)
