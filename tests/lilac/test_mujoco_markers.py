import pytest


@pytest.mark.mujoco
def test_mujoco_integration_placeholder_requires_server_environment():
    pytest.skip("MuJoCo integration is intentionally server-only for LILAC validation.")
