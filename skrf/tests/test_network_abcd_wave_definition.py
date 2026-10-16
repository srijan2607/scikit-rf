"""ABCD parameters describe port voltages/currents, not a wave basis."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skrf import Network


def scattering_from_port_states(abcd, z0, s_def):
    """Excite independent output voltage/current states and solve b = S a."""
    result = []
    for chain, impedance in zip(abcd, z0):
        # [V1, I1] = chain @ [V2, -I2]; columns are independent states.
        voltage = np.array([chain[0], [1, 0]], dtype=complex)
        current = np.array([chain[1], [0, -1]], dtype=complex)
        impedance = impedance[:, None]
        if s_def == 'power':
            scale = 1 / (2 * np.sqrt(impedance.real))
            reflected = scale * (voltage - impedance.conj() * current)
        else:
            if s_def == 'pseudo':
                scale = np.sqrt(impedance.real) / (2 * np.abs(impedance))
            else:
                scale = 1 / (2 * np.sqrt(impedance))
            reflected = scale * (voltage - impedance * current)
        incident = scale * (voltage + impedance * current)
        result.append(np.linalg.solve(incident.T, reflected.T).T)
    return np.array(result)


CHAINS = [
    [[1, 0], [0, 1]],  # ideal through: no finite Z representation
    [[1, 25], [0, 1]],  # series impedance
    [[1, 0], [0.02, 1]],  # shunt admittance
    [[2, 0], [0, 0.5]],  # ideal transformer
    [[3, 200], [0.05, 3.5]],  # nonreciprocal, determinant 0.5
    [[1 + 0.2j, 15 - 5j], [0.01 + 0.02j, 0.8 - 0.1j]],
]
REFERENCES = [
    [[50, 50], [75, 25]],
    [[50 + 30j, 50 + 30j], [40 - 20j, 80 + 10j]],
    [[20 + 90j, 100 - 50j], [75 + 2j, 15 - 30j]],
]


@pytest.mark.parametrize('s_def', ['power', 'pseudo', 'traveling'])
@pytest.mark.parametrize('z0', REFERENCES)
@pytest.mark.parametrize('chain', CHAINS)
def test_abcd_getter_uses_physical_port_states(s_def, z0, chain):
    abcd = np.array([chain, chain], dtype=complex)
    z0 = np.asarray(z0)
    scattering = scattering_from_port_states(abcd, z0, s_def)
    network = Network(f=[1, 2], s=scattering, z0=z0, s_def=s_def)
    original = network.s.copy()

    assert_allclose(network.a, abcd, rtol=1e-12, atol=1e-12)
    assert_array_equal(network.s, original)
    assert_array_equal(network.z0, z0)
    assert network.s_def == s_def


@pytest.mark.parametrize('s_def', ['power', 'pseudo', 'traveling'])
@pytest.mark.parametrize('z0', REFERENCES)
@pytest.mark.parametrize('chain', CHAINS)
def test_abcd_setter_uses_declared_wave_definition(s_def, z0, chain):
    abcd = np.array([chain, chain], dtype=complex)
    z0 = np.asarray(z0)
    expected = scattering_from_port_states(abcd, z0, s_def)
    network = Network(f=[1, 2], s=np.zeros((2, 2, 2)), z0=z0, s_def=s_def)
    network.a = abcd

    assert_allclose(network.s, expected, rtol=1e-12, atol=1e-12)
    assert_array_equal(network.z0, z0)
    assert network.s_def == s_def


@pytest.mark.parametrize('s_def', ['power', 'pseudo', 'traveling'])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64, np.complex64, np.complex128])
def test_abcd_constructor_input_dtype(s_def, dtype):
    abcd = np.array([[[1, 25], [0, 1]]], dtype=dtype)
    z0 = np.array([[50 + 30j, 75 - 20j]])
    expected = scattering_from_port_states(abcd, z0, s_def)
    network = Network(f=[1], a=abcd, z0=z0, s_def=s_def)
    assert_allclose(network.s, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(network.a, abcd, rtol=1e-12, atol=1e-12)
    assert network.s_def == s_def
