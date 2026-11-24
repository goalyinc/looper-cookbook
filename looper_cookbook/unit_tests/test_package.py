"""The cookbook as installed: its version. looper-isolated also runs this file
against the built wheel, at the lowest versions its ranges allow."""

from __future__ import annotations

import importlib.metadata

from looper_cookbook._version import __version__


def test_the_distribution_carries_the_package_version() -> None:
    assert importlib.metadata.version("goaly-looper-cookbook") == __version__
