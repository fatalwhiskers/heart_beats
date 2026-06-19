"""
run_test.py
-----------
Generates the synthetic face video (if it doesn't already exist) and runs
it through the real demo pipeline -- the same code path a real video would
take. This is the fastest way to confirm the pipeline works end-to-end
without needing any real video file or internet access for a real face.

Usage
-----
    python demo/synthetic_test/run_test.py
    python demo/synthetic_test/run_test.py --crop_mode mesh_forehead
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo.synthetic_test.generate_synthetic_face import main as generate_video, OUTPUT_PATH
from demo.demo_run import run_demo, CROP_MODE_CHOICES


def main():
    parser = argparse.ArgumentParser(
        description="Run the rPPG demo against a generated synthetic face (no real person involved)."
    )
    parser.add_argument("--crop_mode", choices=CROP_MODE_CHOICES, default="mesh_forehead",
                         help="Crop mode to test (default: mesh_forehead, the most reliable on synthetic faces)")
    parser.add_argument("--channel", default="G", help="Signal channel category to test (default: G)")
    parser.add_argument("--hr_method", action="append", default=None,
                         help="HR method(s) to test. Repeat to test more than one. Default: welch")
    args = parser.parse_args()

    if not os.path.isfile(OUTPUT_PATH):
        print("No synthetic test video found yet -- generating one now...\n")
        generate_video()
        print()

    hr_methods = args.hr_method or ["welch"]

    try:
        run_demo(OUTPUT_PATH, args.crop_mode, args.channel, hr_methods,
                  output_dir=os.path.join(os.path.dirname(__file__), "output"))
    except ValueError as e:
        if "valid samples" in str(e).lower():
            print("\nNo face was detected in the synthetic video at all, "
                  "so there was nothing to estimate a heart rate from.")
            print(f"This can happen with crop_mode='{args.crop_mode}' -- "
                  f"try 'mesh_forehead', which tends to be the most reliable "
                  f"on a synthetic (non-photographic) face.")
        else:
            raise


if __name__ == "__main__":
    main()
