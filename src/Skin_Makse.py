import cv2
import numpy as np
# \cite{Kakumanu2007} https://doi.org/10.1016/j.patcog.2006.06.010
# De Haan, G., & Jeanne, V. (2013). "Robust pulse rate from chrominance-based rPPG." IEEE Trans. Biomedical Engineering, 60(10), 2878–2886.
# McDuff, D., Estepp, J. R., Piasecki, A. M., & Blackford, E. B. (2015). "A survey of remote optical photoplethysmographic imaging methods." Proc. 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC).

def get_skin_means(frame, cfg):
    """
    Try to compute mean R,G,B only over skin pixels (HSV mask).
    Returns (R,G,B) if mask is valid, else None.

    Parameters
    ----------
    frame : np.ndarray (H,W,3) BGR image
    cfg   : config.Signal with thresholds (ROI_MIN_FRAC, ROI_MAX_FRAC, etc.)
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Two hue ranges because skin hue wraps around red (0 and 180 in OpenCV HSV)
    lower1 = np.array([0,   30,  40], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30,  40], dtype=np.uint8)
    upper2 = np.array([179, 180, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Coverage check
    num_skin = int(np.count_nonzero(mask))
    frac = num_skin / float(frame.shape[0] * frame.shape[1])
    if frac < cfg.ROI_MIN_FRAC or frac > cfg.ROI_MAX_FRAC:
        return None

    # Compute means over mask
    B = cv2.mean(frame[:, :, 0], mask=mask)[0]
    G = cv2.mean(frame[:, :, 1], mask=mask)[0]
    R = cv2.mean(frame[:, :, 2], mask=mask)[0]
    return R, G, B