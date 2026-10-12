import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run dataset-backed tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs the cached PurelandFold dataset / node; use --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
