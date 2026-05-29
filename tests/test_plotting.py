"""Regression tests for ``plot_continuous_tissue_3d`` dtype handling (issue #116).

The previous categorical-detection branch used ``isinstance(vals.dtype, object)``
which is *always* True (every dtype instance is an ``object``).  As a result,
numeric ``obs`` columns were silently converted to integer category codes
before plotting, corrupting any continuous coloring (e.g. depth, density,
score).  The fix narrows the test to ``pd.CategoricalDtype`` plus the explicit
non-numeric ``dtype.kind`` set.
"""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd


def test_numeric_obs_coloring_preserved(monkeypatch):
    """A numeric ``obs`` column must reach the plotter as numeric values.

    Before the fix, ``isinstance(vals.dtype, object)`` was always True so the
    categorical branch fired for numeric columns and ``vals`` became
    ``pd.Categorical(vals).codes`` (rank integers).  This test asserts the
    real numeric values reach the Plotly scatter trace.
    """
    import aether_3d.visualization.plot_3d as plot_mod

    captured: dict = {}

    class _FakeScatter3d:
        def __init__(self, **kwargs):
            captured["scatter_kwargs"] = kwargs

    class _FakeMesh3d:
        def __init__(self, **kwargs):
            captured["mesh_kwargs"] = kwargs

    class _FakeFigure:
        def __init__(self, data=None):
            self.data = data

        def update_layout(self, **kwargs):
            pass

        def write_html(self, path):
            captured["html_path"] = path

        def show(self):
            pass

    fake_go = types.SimpleNamespace(
        Scatter3d=_FakeScatter3d, Mesh3d=_FakeMesh3d, Figure=_FakeFigure
    )
    monkeypatch.setattr(plot_mod, "_HAS_PYVISTA", False)
    monkeypatch.setattr(plot_mod, "_HAS_PLOTLY", True)
    monkeypatch.setattr(plot_mod, "go", fake_go)

    n = 5
    rng = np.random.default_rng(0)
    X = rng.random((n, 3)).astype(np.float32)
    coords3d = rng.random((n, 3)).astype(np.float32)
    depth_values = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    obs = pd.DataFrame({"depth": depth_values})
    adata = ad.AnnData(X=X, obs=obs, obsm={"spatial_3d": coords3d})

    plot_mod.plot_continuous_tissue_3d(
        adata,
        color_by="depth",
        layer=None,
        use_pyvista=False,
        output_html="/tmp/aether_test_plot_116.html",
    )

    color = np.asarray(captured["scatter_kwargs"]["marker"]["color"])
    np.testing.assert_array_equal(
        color,
        depth_values,
        err_msg="numeric obs column was corrupted (likely category codes)",
    )
    # Sanity: ensure we did NOT receive category codes [0..n-1].
    assert not np.array_equal(color, np.arange(n)), (
        "marker.color collapsed to rank codes — categorical branch wrongly fired"
    )


def test_categorical_obs_coloring_still_uses_codes(monkeypatch):
    """The fix must not break the categorical branch.

    A string / categorical obs column should still be mapped to integer codes
    so existing plotting paths keep working.
    """
    import aether_3d.visualization.plot_3d as plot_mod

    captured: dict = {}

    class _FakeScatter3d:
        def __init__(self, **kwargs):
            captured["scatter_kwargs"] = kwargs

    class _FakeFigure:
        def __init__(self, data=None):
            self.data = data

        def update_layout(self, **kwargs):
            pass

        def write_html(self, path):
            pass

        def show(self):
            pass

    fake_go = types.SimpleNamespace(
        Scatter3d=_FakeScatter3d,
        Mesh3d=_FakeScatter3d,
        Figure=_FakeFigure,
    )
    monkeypatch.setattr(plot_mod, "_HAS_PYVISTA", False)
    monkeypatch.setattr(plot_mod, "_HAS_PLOTLY", True)
    monkeypatch.setattr(plot_mod, "go", fake_go)

    n = 4
    rng = np.random.default_rng(0)
    X = rng.random((n, 3)).astype(np.float32)
    coords3d = rng.random((n, 3)).astype(np.float32)
    labels = np.array(["A", "B", "A", "C"], dtype=object)
    obs = pd.DataFrame({"label": labels})
    adata = ad.AnnData(X=X, obs=obs, obsm={"spatial_3d": coords3d})

    plot_mod.plot_continuous_tissue_3d(
        adata,
        color_by="label",
        layer=None,
        use_pyvista=False,
        output_html="/tmp/aether_test_plot_116_cat.html",
    )

    color = np.asarray(captured["scatter_kwargs"]["marker"]["color"])
    # Categorical codes are integers in [0, n_unique).
    assert color.dtype.kind in "iu", f"expected integer codes, got dtype {color.dtype}"
    assert set(color.tolist()) <= {0, 1, 2}, (
        f"expected codes in {{0,1,2}} for 3 unique labels, got {sorted(set(color.tolist()))}"
    )
