"""
testui.py
---------
A simpler way to run the synthetic-face demo: change the variables below,
then just run this file. No command-line flags to remember.

This does exactly the same thing as run_test.py / demo_run.py underneath
-- it's a different way to choose the settings, not different logic.
If you're comfortable with --flags, run_test.py works exactly as before
and this file changes nothing about it.

Usage
-----
    1. Edit the variables in the "SETTINGS" section below.
    2. Run:  python demo/synthetic_test/testui.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo.synthetic_test.generate_synthetic_face import main as generate_video, OUTPUT_PATH
from demo.demo_run import (
    run_demo,
    CROP_MODE_CHOICES, CROP_MODE_DESCRIPTIONS,
    CHANNEL_CHOICES, CHANNEL_DESCRIPTIONS,
    HR_METHOD_CHOICES, HR_METHOD_DESCRIPTIONS,
)


# =============================================================================
# SETTINGS -- change these, then run this file.
# =============================================================================

# Which part of the face to crop to before sampling colour.
#   "face_track"     - tracks the whole face region
#   "bbox_forehead"  - crops to the forehead using the face bounding box (good default)
#   "mesh_forehead"  - crops to the forehead using a precise 478-point face mesh
#   "poly"           - uses forehead + both cheeks for maximum skin coverage
# On the synthetic face specifically, "mesh_forehead" is the most reliable
# (see demo/synthetic_test/README.md for why).
CROP_MODE = "mesh_forehead"

# Which signal-channel category to extract the heart rate from.
#   "R" / "G" / "B"  - a single colour channel ("G" is the standard rPPG default)
#   "GREY_W"         - weighted greyscale
#   "GREY_A"         - simple average greyscale
#   "PCA"            - Principal Component Analysis, up to 3 signals
#   "ZCA"            - ZCA whitening, up to 3 signals
#   "ICA"            - Independent Component Analysis, up to 3 signals
#   "CHROM"          - Chrominance-based method
#   "POS"            - Plane-Orthogonal-to-Skin method
CHANNEL = "G"

# Which HR estimation method(s) to run. Put one or both in this list --
# each one produces its own separate plot.
#   "welch"          - Welch periodogram, generally the most robust
#   "hilbert"        - Hilbert-transform instantaneous HR, windowed to match Welch
HR_METHODS = ["welch", "hilbert"]

# =============================================================================
# End of settings -- nothing below this line needs changing.
# =============================================================================


def _validate(value, choices, name):
    if value not in choices:
        raise ValueError(
            f"{name} = '{value}' is not valid. Choose one of: {choices}"
        )


def main():
    _validate(CROP_MODE, CROP_MODE_CHOICES, "CROP_MODE")
    _validate(CHANNEL, CHANNEL_CHOICES, "CHANNEL")
    if not HR_METHODS:
        raise ValueError("HR_METHODS is empty -- pick at least one of: "
                          f"{HR_METHOD_CHOICES}")
    for method in HR_METHODS:
        _validate(method, HR_METHOD_CHOICES, "HR_METHODS entry")

    print("Running with:")
    print(f"  CROP_MODE  = {CROP_MODE!r}  ({CROP_MODE_DESCRIPTIONS[CROP_MODE]})")
    print(f"  CHANNEL    = {CHANNEL!r}  ({CHANNEL_DESCRIPTIONS[CHANNEL]})")
    for m in HR_METHODS:
        print(f"  HR_METHODS includes {m!r}  ({HR_METHOD_DESCRIPTIONS[m]})")
    print()

    if not os.path.isfile(OUTPUT_PATH):
        print("No synthetic test video found yet -- generating one now...\n")
        generate_video()
        print()

    try:
        run_demo(OUTPUT_PATH, CROP_MODE, CHANNEL, HR_METHODS,
                  output_dir=os.path.join(os.path.dirname(__file__), "output"))
    except ValueError as e:
        if "valid samples" in str(e).lower():
            print("\nNo face was detected in the synthetic video at all, "
                  "so there was nothing to estimate a heart rate from.")
            print(f"This can happen with CROP_MODE='{CROP_MODE}' -- "
                  f"try 'mesh_forehead', which tends to be the most reliable "
                  f"on a synthetic (non-photographic) face.")
        else:
            raise


if __name__ == "__main__":
    main()
