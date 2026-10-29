"""Exercise real widget callbacks, cached views and withheld-data controls."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from brainsmodels.brainweb_data import BUNDLED, FILES, verify_files
from brainsmodels.growth_lessons import fit_recovery_problem
from brainsmodels.growth_widgets import planar_explorer, brainweb_explorer, recovery_explorer


def run(panel):
    panel.run_button.click()
    assert panel.error is None, panel.error
    assert panel.last_result is not None and not panel.run_button.disabled
    return panel.last_result


def main():
    verify_files(BUNDLED)
    # Source anatomy and all simulation code must be usable with network blocked.
    with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network access")), \
            redirect_stdout(io.StringIO()):
        a = planar_explorer()
        first = run(a)
        area_high = first[2]["Tissue-dependent D"].threshold_measure(.3)[-1]
        a.controls["threshold"].value = .05
        a.controls["day"].value = 30
        assert a.last_result is first, "View changes reran the simulation"
        assert first[2]["Tissue-dependent D"].threshold_measure(.05)[-1] >= area_high
        a.controls["d_white"].value = .1
        assert a.pending and a.last_result is first
        uniform = run(a)
        np.testing.assert_allclose(uniform[2]["Homogeneous D"].density,
                                   uniform[2]["Tissue-dependent D"].density, atol=1e-13)
        a.controls["d_white"].value = 1.
        a.controls["dt"].value = 1.
        a.run_button.click()
        assert a.error and "no larger" in a.error and a.last_result is None
        assert not a.run_button.disabled

        b = brainweb_explorer()
        anatomy = run(b)
        assert anatomy[0].labels.shape == (46, 55, 46)
        b.controls["threshold"].value = .2
        b.controls["x"].value = -40.
        assert b.last_result is anatomy, "Slice/threshold controls reran the simulation"
        b.controls["spacing"].value = 6
        assert b.pending
        coarser = run(b)
        assert coarser[0].labels.shape == (31, 37, 31)
        assert coarser[0].spacing_mm == (6, 6, 6)

        c = recovery_explorer()
        baseline = run(c)
        assert baseline["summary"]["fit_days"] == [20., 40., 60.]
        c.controls["fit_indices"].value = (0,)
        c.controls["grid_size"].value = (5, 5)
        c.controls["seed"].value = 8
        c.controls["true_scale"].value = 1.2
        c.controls["true_rho"].value = .05
        assert c.pending
        single = run(c)
        s = single["summary"]
        assert s["fit_days"] == [20.] and s["withheld_day"] == 90.
        assert s["known_scale"] == 1.2 and s["known_rho_per_day"] == .05
        assert single["fit"]["losses"].shape == (5, 5)
        roi = single["roi"]
        expected = np.sqrt(np.mean((single["observations"][0, roi]
                                   - single["observations"][3, roi])**2))
        np.testing.assert_allclose(s["persistence_withheld_rmse"], expected)
        assert not np.array_equal(single["observations"], baseline["observations"])
        try:
            fit_recovery_problem(single, fit_indices=(0, 3))
        except ValueError:
            pass
        else:
            raise AssertionError("Day 90 was allowed into parameter fitting")
        # Corrupt the withheld array: estimates must stay unchanged.
        original = fit_recovery_problem(single, fit_indices=(0,),
                                         scales=[.8, 1.2], rhos=[.04, .05])
        single["observations"][3] = 100.
        protected = fit_recovery_problem(single, fit_indices=(0,),
                                          scales=[.8, 1.2], rhos=[.04, .05])
        np.testing.assert_array_equal(original["losses"], protected["losses"])

    print(f"PASS: {len(FILES)} bundled BrainWeb files; network-blocked widget calculations; "
          "button callbacks; cached views; pending controls; stability rejection; grid changes; "
          "generating parameters; selected-time persistence; protected withheld data.")


if __name__ == "__main__":
    main()
