"""
Tests for core statistics functions.
"""

import numpy as np
import pytest
from meteor.core.stats import (
    compute_basic_stats,
    compute_additional_stats,
    compute_entropy,
    compute_volume,
    dice_coefficient
)

def test_compute_basic_stats():
    arr = np.array([1, 2, 3, 4, 5], dtype=float)
    stats = compute_basic_stats(arr)
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["mean"] == 3
    assert stats["median"] == 3
    assert abs(stats["std"] - 1.4142) < 1e-3

def test_compute_additional_stats():
    arr = np.random.randn(1000).astype(float)
    stats = compute_additional_stats(arr)
    assert "skew" in stats
    assert "kurtosis" in stats
    assert "p25" in stats
    assert "p75" in stats

def test_entropy():
    arr = np.random.randint(0, 256, size=(1000,), dtype=np.uint8)
    e = compute_entropy(arr, nbins=32)
    assert e >= 0

def test_volume():
    mask = np.zeros((10,10,10), dtype=bool)
    mask[2:5,3:7,4:9] = True
    sp = (1,1,1)
    vol = compute_volume(mask, sp)
    assert vol == 60  # (3 in z * 4 in y * 5 in x)

def test_dice_coefficient():
    m1 = np.zeros((5,5,5), dtype=bool)
    m2 = np.zeros((5,5,5), dtype=bool)
    m1[1:3,1:3,1:3] = True
    m2[2:4,2:4,2:4] = True
    dice = dice_coefficient(m1, m2)
    assert abs(dice - 0.125) < 1e-9
