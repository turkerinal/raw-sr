"""Test that the package structure is importable."""
import importlib


def test_raw_sr_package_importable():
    importlib.import_module('raw_sr')


def test_core_subpackage_importable():
    importlib.import_module('raw_sr.core')


def test_ui_subpackage_importable():
    importlib.import_module('raw_sr.ui')


def test_main_module_importable():
    importlib.import_module('raw_sr.main')