"""Public-API behavioral coverage for previously untested surface (issue #89).

Covers:
- ``velocity_to_score`` / ``velocity_to_noise`` against analytic LinearPath values.
- Boundary finiteness of those conversions for GVP and VP paths at t in {0, 1}.
- An external-reconstruction adapter's ``_reconstruct`` path, exercised end-to-end
  through the audited contract via an injected stub module.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import torch

from aether_3d.flow.path import GVPPath, LinearPath, VPPath


def test_velocity_to_score_linear_matches_analytic() -> None:
    path = LinearPath()
    t = torch.tensor([0.5])
    x = torch.tensor([[2.0, -1.0]])
    v = torch.tensor([[1.0, 4.0]])

    # LinearPath: alpha=t, dalpha=1, sigma=1-t, dsigma=-1.
    #   ratio = t,  var = (1-t)  =>  score = (t*v - x) / (1-t)  = v - 2x at t=0.5
    score = path.velocity_to_score(v, x, t)
    expected = v - 2.0 * x
    assert torch.allclose(score, expected, atol=1e-5)


def test_velocity_to_noise_linear_matches_analytic() -> None:
    path = LinearPath()
    t = torch.tensor([0.5])
    x = torch.tensor([[2.0, -1.0]])
    v = torch.tensor([[1.0, 4.0]])

    # var = ratio*dsigma - sigma = -1  =>  noise = x - t*v = x - 0.5*v at t=0.5
    noise = path.velocity_to_noise(v, x, t)
    expected = x - 0.5 * v
    assert torch.allclose(noise, expected, atol=1e-5)


def test_velocity_conversions_finite_at_boundaries() -> None:
    x = torch.randn(4, 3)
    v = torch.randn(4, 3)
    for path in (LinearPath(), GVPPath(), VPPath()):
        for t_val in (0.0, 1.0):
            t = torch.full((4,), t_val)
            score = path.velocity_to_score(v, x, t)
            noise = path.velocity_to_noise(v, x, t)
            assert torch.isfinite(score).all(), f"{path} score not finite at t={t_val}"
            assert torch.isfinite(noise).all(), f"{path} noise not finite at t={t_val}"


def test_gvp_dalpha_nonnegative_at_t1() -> None:
    # GVP dalpha is clamped to its analytic lower bound 0 so the boundary sign of
    # ratio = alpha/dalpha is not flipped by float32 noise (review of #135).
    _, da = GVPPath().alpha(torch.tensor([1.0]))
    assert (da >= 0).all()


class _StubReconstructor:
    """Minimal stand-in for an external 3D reconstruction package."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._template: ad.AnnData | None = None

    def fit(self, visible: list[ad.AnnData]) -> None:
        self._template = visible[0]

    def predict(self, virtual_z: list[float]) -> ad.AnnData:
        assert self._template is not None
        n_genes = self._template.n_vars
        rng = np.random.default_rng(0)
        out = []
        for z in virtual_z:
            n = 5
            v = ad.AnnData(X=rng.normal(size=(n, n_genes)).astype(np.float32))
            v.var_names = self._template.var_names
            xy = rng.normal(size=(n, 2)).astype(np.float32)
            v.obsm["spatial"] = xy
            v.obsm["spatial_3d"] = np.hstack([xy, np.full((n, 1), float(z), dtype=np.float32)])
            v.obs["z"] = float(z)
            out.append(v)
        import scanpy as sc

        return sc.concat(out, axis=0, join="outer")


def test_external_adapter_reconstruct_via_contract(
    make_aether_slices, stub_adapter_module
) -> None:
    from aether_3d.benchmarks.adapters import SpatialZAdapter
    from aether_3d.benchmarks.runner import run_holdout

    stub_adapter_module("spatialz", _StubReconstructor)

    slices = make_aether_slices(n_slices=3, n_cells=10, n_genes=6)
    for s in slices:
        # The contract default z_key is "z"; mirror the per-slice z there.
        s.obs["z"] = float(s.obs["z_coord"].iloc[0])

    adapter = SpatialZAdapter()
    available, _ = adapter.is_available()
    assert available, "stub spatialz module should make the adapter available"

    results = run_holdout(
        [adapter], slices, held_out_indices=[1], z_key="z", label_key="cell_class"
    )
    assert results[0].status == "ok", results[0].status
    assert results[0].metrics_json["n_virtual_cells"] > 0
