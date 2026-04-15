#!/usr/bin/env python3
"""
Compute RGB channel gains to shift a display white point
from blackbody temperature T to T_prime.

Assumptions:
- White points are approximated from correlated color temperature using
  standard blackbody-locus approximations.
- Display primaries are linear sRGB.
- Gains are applied in linear RGB, not gamma-encoded RGB.

Usage example:
    python whitepoint_gains.py --T 6500 --Tp 5000
"""

from __future__ import annotations

import argparse
import numpy as np


# Standard linear sRGB -> XYZ matrix (D65)
M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

M_XYZ_TO_RGB = np.linalg.inv(M_RGB_TO_XYZ)


def cct_to_xy_blackbody(T: float) -> tuple[float, float]:
    """
    Approximate blackbody chromaticity (x, y) from temperature T in Kelvin.

    Valid roughly for 1667 K to 25000 K.

    Source:
    Common approximation published by Hernández-Andrés et al. / widely reused
    in colour-science references.
    """
    if not (1667 <= T <= 25000):
        raise ValueError("T must be in the range [1667, 25000] Kelvin")

    # Compute x(T)
    if 1667 <= T <= 4000:
        x = (
            -0.2661239e9 / T**3
            - 0.2343580e6 / T**2
            + 0.8776956e3 / T
            + 0.179910
        )
    else:
        x = (
            -3.0258469e9 / T**3
            + 2.1070379e6 / T**2
            + 0.2226347e3 / T
            + 0.240390
        )

    # Compute y(x)
    if 1667 <= T <= 2222:
        y = (
            -1.1063814 * x**3
            - 1.34811020 * x**2
            + 2.18555832 * x
            - 0.20219683
        )
    elif 2222 < T <= 4000:
        y = (
            -0.9549476 * x**3
            - 1.37418593 * x**2
            + 2.09137015 * x
            - 0.16748867
        )
    else:
        y = (
            3.0817580 * x**3
            - 5.87338670 * x**2
            + 3.75112997 * x
            - 0.37001483
        )

    return x, y


def xy_to_xyz(x: float, y: float, Y: float = 1.0) -> np.ndarray:
    """
    Convert xy chromaticity to XYZ tristimulus with chosen luminance Y.
    """
    if y == 0:
        raise ValueError("y must be nonzero")
    X = x * Y / y
    Z = (1.0 - x - y) * Y / y
    return np.array([X, Y, Z], dtype=np.float64)


def temperature_to_white_xyz(T: float) -> np.ndarray:
    """
    Convert blackbody temperature to normalized XYZ white point with Y = 1.
    """
    x, y = cct_to_xy_blackbody(T)
    return xy_to_xyz(x, y, Y=1.0)


def xyz_to_linear_srgb(XYZ: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to linear sRGB.
    """
    return M_XYZ_TO_RGB @ XYZ


def compute_channel_gains(
    T_source: float,
    T_target: float,
    normalization: str = "green",
) -> dict[str, np.ndarray | tuple[float, float] | float]:
    """
    Compute linear RGB channel gains to shift white from T_source to T_target.

    Returns a dict containing intermediate values and final gains.

    normalization options:
    - "none": raw componentwise ratio
    - "green": normalize so G gain = 1
    - "max": normalize so max gain = 1
    """
    xyz_source = temperature_to_white_xyz(T_source)
    xyz_target = temperature_to_white_xyz(T_target)

    rgb_source = xyz_to_linear_srgb(xyz_source)
    rgb_target = xyz_to_linear_srgb(xyz_target)

    # Componentwise ratio
    gains = rgb_target / rgb_source

    if normalization == "green":
        gains = gains / gains[1]
    elif normalization == "max":
        gains = gains / np.max(gains)
    elif normalization == "none":
        pass
    else:
        raise ValueError("normalization must be one of: none, green, max")

    xy_source = cct_to_xy_blackbody(T_source)
    xy_target = cct_to_xy_blackbody(T_target)

    return {
        "T_source": T_source,
        "T_target": T_target,
        "xy_source": xy_source,
        "xy_target": xy_target,
        "xyz_source": xyz_source,
        "xyz_target": xyz_target,
        "rgb_source": rgb_source,
        "rgb_target": rgb_target,
        "gains": gains,
    }


def format_vec(name: str, v: np.ndarray) -> str:
    return f"{name} = [{v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("T", type=float,
                        help="source white temperature in Kelvin")
    parser.add_argument("Tp", type=float,
                        help="target white temperature in Kelvin")
    parser.add_argument(
        "--norm",
        type=str,
        default="green",
        choices=["none", "green", "max"],
        help="gain normalization mode"
    )
    args = parser.parse_args()

    result = compute_channel_gains(args.T, args.Tp, normalization=args.norm)

    print(f"Source temperature T  = {result['T_source']:.2f} K")
    print(f"Target temperature T' = {result['T_target']:.2f} K")
    print()

    xs, ys = result["xy_source"]
    xt, yt = result["xy_target"]
    print(f"Source xy = ({xs:.8f}, {ys:.8f})")
    print(f"Target xy = ({xt:.8f}, {yt:.8f})")
    print()

    print(format_vec("Source XYZ", result["xyz_source"]))
    print(format_vec("Target XYZ", result["xyz_target"]))
    print()

    print(format_vec("Source linear sRGB white", result["rgb_source"]))
    print(format_vec("Target linear sRGB white", result["rgb_target"]))
    print()

    g = result["gains"]
    print(format_vec("Channel gains [R, G, B]", g))
    print()
    print("Apply these gains to LINEAR RGB:")
    print("R_out = gain_R * R_in")
    print("G_out = gain_G * G_in")
    print("B_out = gain_B * B_in")


if __name__ == "__main__":
    main()
