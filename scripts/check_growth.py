"""Analytic and numerical regression checks for the verification notebook."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from brainsmodels.growth_verification import audit, diffuse, evolve, production_audit
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-checkout", type=Path)
    args = parser.parse_args()
    result = audit()
    assert result["logistic_max_abs_error"] < 1e-8, result
    assert result["relative_mass_error"] < 1e-12, result
    assert result["diffusion_minimum"] >= 0, result
    assert result["zero_uptake_coupled_abs_error"] < 1e-8, result
    assert result["zero_uptake_necrosis"] == 0, result
    for key in ("spatial_rms_errors", "temporal_l2_errors"):
        errors = result[key]
        assert all(b < .65*a for a, b in zip(errors, errors[1:])), result
    # A blocked face transmits nothing; reflecting edges never wrap around.
    initial = np.array([1., 0., 0., 0.])
    step = diffuse(initial, np.array([1., 1., 0., 1.]), 1., .1)
    np.testing.assert_allclose(step, [.9, .1, 0., 0.])
    np.testing.assert_array_equal(evolve(np.ones(5), rho=0.), np.ones(5))
    try:
        diffuse(initial, 1., 1., 1.)
    except ValueError:
        pass
    else:
        raise AssertionError("Unstable explicit step was accepted")
    if args.reference_checkout:
        # Keep bytecode and JIT caches out of the read-only reference checkout.
        sys.dont_write_bytecode = True
        os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".build/numba"))
        result["external_3d"] = production_audit(args.reference_checkout)
    print(json.dumps(result, indent=2))
    print("PASS: analytic logistic solution; no-flux conservation and barriers; "
          "stability rejection; spatial/time refinement; coupled limiting case.")


if __name__ == "__main__":
    main()
