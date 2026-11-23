"""Where the SignSuisse script writes the ELAN files of the 2023 model.

Run from `segment-any-sign/`:
    pytest tests/test_signsuisse_outputs.py

`datasets/signsuisse/run_2023_segmentation.py` writes one ELAN file per example,
and `datasets/signsuisse/explore.py` reads them from
`example_segmentation/v2023_e4s_60_50` in the SignSuisse folder. The script must
name that folder when it is run with its default arguments.

The test imports the script, which imports torch, pose_format and pympi. It
calls one function: `main()` does not run, and no file is read or written. It is
marked with `needs` (see test_locations.py) and skipped, with the missing
packages named, where they are not installed.
"""

from pathlib import Path

from tests.test_locations import needs

# what the script imports
SCRIPT_NEEDS = ("numpy", "pympi", "torch", "pose_format")


@needs(*SCRIPT_NEEDS)
def test_default_out_dir_is_a_folder_of_example_segmentation_beside_the_poses():
    from datasets.signsuisse.run_2023_segmentation import default_out_dir

    root = Path("Signsuisse")
    # the default arguments, model_E4s-1.pth with --b 60 --o 50, as argparse gives
    # them: the folder that the notebook reads
    assert default_out_dir(root / "example_mediapipe", "e4s", 60.0, 50.0) \
        == root / "example_segmentation" / "v2023_e4s_60_50"
    # another model and other thresholds get a folder of their own beside it
    assert default_out_dir(root / "example_mediapipe", "e1s", 40.0, 60.0) \
        == root / "example_segmentation" / "v2023_e1s_40_60"
    # beside the poses folder, wherever --poses points
    assert default_out_dir(Path("elsewhere") / "poses", "e4s", 60.0, 50.0) \
        == Path("elsewhere") / "example_segmentation" / "v2023_e4s_60_50"
