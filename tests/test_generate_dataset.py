from shutil import rmtree

import numpy as np
import pandas as pd

from dataset import sample_dataset


def test_sample_dataset():
    # generate a dummy dataset
    sample_dataset(n_videos=100, frames_each_video=50, pose_vector_dim=128)
    # read the generated csv and check if columns exist

    df = pd.read_csv("dataset.csv", skip_blank_lines=True, skipinitialspace=True)

    assert len(df.columns.tolist()) == 131, f"dataset should have 131 columns, but found {len(df.columns.tolist())}"

    dtypes = []
    for k in df.dtypes.to_list():
        if k == np.float64:
            dtypes.append(True)
        else:
            dtypes.append(False)
    assert all(dtypes), f"all columns should be of type float64, but found {df.dtypes.to_list()}"
    for c in ["video", "frame_no", "Label_fall"]:
        assert c in df.columns.tolist(), f"column {c} should be present in dataset, but found {df.columns.tolist()}"

    assert df.shape == (5000, 131), f"dataset should have 5000 rows and 131 columns, but found {df.shape}"
    # delete dataset.csv file after test is complete
    rmtree("dataset.csv", ignore_errors=True)
