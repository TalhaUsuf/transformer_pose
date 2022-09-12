from dataset import sample_dataset


def test_sample_dataset():
    sample_dataset(n_videos=100, frames_each_video=50, pose_vector_dim=128)
