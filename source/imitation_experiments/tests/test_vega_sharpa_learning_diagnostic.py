import numpy as np

from imitation_experiments.audit.vega_sharpa_learning import contact_force_from_balance


def test_gravity_compensated_free_object_needs_no_contact_force():
    result = contact_force_from_balance(
        0.3, [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 2.943]], [0, 0, -9.81], 0.05
    )
    np.testing.assert_allclose(result, 0, atol=1e-12)


def test_downward_virtual_force_requires_upward_contact_at_equilibrium():
    result = contact_force_from_balance(
        0.3, [[0, 0, 0]], [[0, 0, 0]], [[0, 0, -2.3]], [0, 0, -9.81], 0.05
    )
    np.testing.assert_allclose(result, [[0, 0, 5.243]])
