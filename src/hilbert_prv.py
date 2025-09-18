import numpy as np
import neurokit2 as nk
from scipy.signal import hilbert
from scipy.ndimage import median_filter, uniform_filter1d

from src.config import Signal, PRV


def resample_uniform(t_in, x_in, fs_target):

    t_in = np.asarray(t_in, float)
    x_in = np.asarray(x_in, float)

    dt = np.median(np.diff(t_in))
    fs_infer = 1.0 / dt

    x_uniform = nk.signal_resample(
        signal=x_in,
        sampling_rate=fs_infer,
        desired_sampling_rate=fs_target,
        method="pchip",
    )
    t_uniform = np.linspace(t_in[0], t_in[-1], num=len(x_uniform))
    return t_uniform, x_uniform


def estimate_prv_hilbert_simple(t_in, x_in, fs_target=None):

    if fs_target is None:
        fs_target = float(PRV.FPS_RESAMPLE_RATE)

    t_uniform, x_uniform = resample_uniform(t_in, x_in, fs_target)

    x_bp = nk.signal_filter(
        x_uniform,
        sampling_rate=fs_target,
        lowcut=float(Signal.HR_LOW),
        highcut=float(Signal.HR_HIGH),
        method="butterworth",
        order=4,
    )

    x_analytic = hilbert(x_bp - np.mean(x_bp))
    phi = np.unwrap(np.angle(x_analytic))

    cycle_idx = np.floor((phi - phi[0]) / (2 * np.pi)).astype(int)
    wrap_idx = np.flatnonzero(np.diff(cycle_idx) > 0)
    t_beats = t_uniform[wrap_idx]

    if t_beats.size < 2:
        return (np.array([]),) * 6

    pp = np.diff(t_beats)
    t_mid = 0.5 * (t_beats[1:] + t_beats[:-1])

    med_pp = median_filter(pp, size=int(PRV.KUBIOS_L), mode="reflect")
    artifact_mask = np.abs(pp - med_pp) > float(PRV.KUBIOS_THRESHOLD)
    pp_clean = np.where(artifact_mask, med_pp, pp)

    hr_raw = 60.0 / pp
    hr_clean = 60.0 / pp_clean
    hr_smooth = uniform_filter1d(hr_clean, size=5)

    return pp_clean, hr_smooth, hr_raw, t_mid, t_beats, artifact_mask
