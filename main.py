from ezvitsdataset.EZVitsDataset import EZVitsDataset, EZVitsDatasetParams

dataset = EZVitsDataset(
    EZVitsDatasetParams(
        device="cuda",
        language="en",
    )
)

dataset.main(
    mode="download",
    path_or_url="youtube video or playlist url",
)
