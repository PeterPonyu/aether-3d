"""
Proper pytest tests for Aether3D (shared flow + Aether-specific components).
"""

import pytest
import torch
import numpy as np
import anndata as ad
import pytorch_lightning as pl

from aether_3d.flow import create_flow_transport, FlowSampler, LinearPath
from aether_3d.flow.integrators import ode as ode_integrator
from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset
from aether_3d.models.aether_velocity_field import MultiModalVelocityField
from aether_3d.coupling.uot import compute_hybrid_cost, compute_uot_coupling


def test_linear_path():
    path = LinearPath()
    t = torch.tensor([0.2, 0.8])
    x0 = torch.randn(2, 2)
    x1 = torch.randn(2, 2)
    _, _, ut = path.plan(t, x0, x1)
    assert ut.shape == x0.shape


def test_aether_multi_modal_field():
    # Note: current model expects gene input already patched to patch_size (default 8)
    model = MultiModalVelocityField(
        spatial_dim=2,
        gene_dim=32,
        num_classes=4,
        hidden_size=24,
        depth=2,
        num_heads=2,
        patch_size=8,
    )
    state = {
        "x": torch.randn(3, 2),
        "g": torch.randn(3, 32),  # raw gene features
        "c": torch.randn(3, 4),
    }
    t = torch.rand(3)

    vel = model(state, t, torch.zeros(3, dtype=torch.long))
    assert "vx" in vel and vel["vx"].shape == (3, 2)
    assert "vg" in vel and vel["vg"].shape[0] == 3


def test_uot_coupling_runs():
    rng = np.random.default_rng(0)
    n0, n1 = 50, 40
    cost = rng.random((n0, n1)).astype(np.float32)
    src, tgt, w = compute_uot_coupling(cost, n_samples=100)
    assert len(src) == 100
    assert len(tgt) == 100


def test_aether_config():
    cfg = Aether3DConfig(hidden_size=32, lambda_g=0.2)
    assert cfg.lambda_g == 0.2
    assert cfg.hidden_size == 32


def test_pytorch_uot_and_cost_parity():
    rng = np.random.default_rng(42)
    n0, n1 = 30, 25
    x0, x1 = (
        rng.random((n0, 2)).astype(np.float32),
        rng.random((n1, 2)).astype(np.float32),
    )
    g0, g1 = (
        rng.random((n0, 10)).astype(np.float32),
        rng.random((n1, 10)).astype(np.float32),
    )

    # one-hot cell labels
    c0 = np.zeros((n0, 3), dtype=np.float32)
    c0[np.arange(n0), rng.integers(0, 3, n0)] = 1.0
    c1 = np.zeros((n1, 3), dtype=np.float32)
    c1[np.arange(n1), rng.integers(0, 3, n1)] = 1.0

    # CPU hybrid cost
    c_cpu = compute_hybrid_cost(x0, g0, c0, x1, g1, c1, alpha_spatial=0.6)

    # PyTorch CPU hybrid cost
    x0_t, x1_t = torch.tensor(x0), torch.tensor(x1)
    g0_t, g1_t = torch.tensor(g0), torch.tensor(g1)
    c0_t, c1_t = torch.tensor(c0), torch.tensor(c1)

    c_pt = compute_hybrid_cost(x0_t, g0_t, c0_t, x1_t, g1_t, c1_t, alpha_spatial=0.6)

    assert isinstance(c_pt, torch.Tensor)
    assert np.allclose(c_cpu, c_pt.numpy(), atol=1e-5)

    # UOT coupling parity
    src_cpu, tgt_cpu, w_cpu = compute_uot_coupling(
        c_cpu, reg=0.5, tau=0.1, n_samples=100
    )
    src_pt, tgt_pt, w_pt = compute_uot_coupling(c_pt, reg=0.5, tau=0.1, n_samples=100)

    assert isinstance(src_pt, torch.Tensor)
    assert isinstance(tgt_pt, torch.Tensor)
    assert isinstance(w_pt, torch.Tensor)
    assert len(src_pt) == 100
    assert w_pt.sum() > 0.0


def test_aether_reconstructor_fit_runs_one_training_batch(tmp_path):
    rng = np.random.default_rng(9)
    adata_list = []
    for z in (0.0, 1.0):
        adata = ad.AnnData(
            X=rng.normal(size=(6, 8)).astype(np.float32),
            obs={
                "cell_class": ["T", "B", "T", "B", "T", "B"],
                "z_coord": [z] * 6,
            },
        )
        adata.obsm["spatial"] = rng.normal(size=(6, 2)).astype(np.float32)
        adata_list.append(adata)

    cfg = Aether3DConfig(
        n_samples_base=6,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
        output_dir=tmp_path,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)
    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    returned = recon.fit(trainer=trainer)

    assert returned is trainer
    assert trainer.global_step == 1
    assert recon.module is not None


# ---------------------------------------------------------------------------
# New edge-case tests for UOT coupling, volume reconstruction, and ODE integration
# ---------------------------------------------------------------------------


def test_uot_hybrid_cost_shapes():
    """Verify compute_hybrid_cost produces a cost matrix of shape (n0, n1)."""
    rng = np.random.default_rng(42)
    n0, n1 = 50, 40
    x0 = rng.random((n0, 2)).astype(np.float32)
    g0 = rng.random((n0, 32)).astype(np.float32)
    c0 = np.eye(4)[rng.integers(0, 4, n0)].astype(np.float32)
    x1 = rng.random((n1, 2)).astype(np.float32)
    g1 = rng.random((n1, 32)).astype(np.float32)
    c1 = np.eye(4)[rng.integers(0, 4, n1)].astype(np.float32)

    cost = compute_hybrid_cost(x0, g0, c0, x1, g1, c1, alpha_spatial=0.5)
    assert cost.shape == (n0, n1)


def test_uot_coupling_valid_indices():
    """Verify UOT coupling returns src/tgt indices within valid bounds."""
    rng = np.random.default_rng(7)
    n0, n1 = 30, 20
    cost = rng.random((n0, n1)).astype(np.float32)
    src, tgt, w = compute_uot_coupling(cost, n_samples=200)

    assert len(src) == 200
    assert len(tgt) == 200
    assert np.all(src >= 0) and np.all(src < n0)
    assert np.all(tgt >= 0) and np.all(tgt < n1)


def test_uot_numpy_fallback():
    """Verify the NumPy (non-PyTorch) code path returns NumPy arrays."""
    rng = np.random.default_rng(13)
    n0, n1 = 20, 15
    cost = rng.random((n0, n1)).astype(np.float32)
    src, tgt, w = compute_uot_coupling(cost, reg=0.5, tau=0.1, n_samples=50)

    assert isinstance(src, np.ndarray)
    assert isinstance(tgt, np.ndarray)
    assert isinstance(w, np.ndarray)


def test_uot_empty_input():
    """Verify graceful handling of an empty source slice (0 cells)."""
    # Construct empty cost matrices directly (compute_hybrid_cost may fail on
    # fully-empty input due to .max() on zero-size arrays)
    cost_empty_src = np.zeros((0, 10), dtype=np.float32)
    cost_empty_tgt = np.zeros((10, 0), dtype=np.float32)

    # UOT coupling on empty-cost matrices should raise a clear error
    with pytest.raises(ValueError):
        compute_uot_coupling(cost_empty_src, n_samples=10)

    with pytest.raises(ValueError):
        compute_uot_coupling(cost_empty_tgt, n_samples=10)


def test_reconstructor_setup_data():
    """Verify setup_data() correctly infers spatial_dim, gene_dim, num_classes."""
    rng = np.random.default_rng(17)
    adata_list = []
    for z in (0.0, 1.0, 2.0):
        n_cells = 50
        adata = ad.AnnData(
            X=rng.normal(size=(n_cells, 32)).astype(np.float32),
            obs={
                "cell_type": np.random.choice(
                    ["T", "B", "Myeloid", "Epithelial"], n_cells
                ),
                "z_coord": [z] * n_cells,
            },
        )
        adata.obsm["spatial"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
        adata_list.append(adata)

    cfg = Aether3DConfig(
        label_key="cell_type",
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)

    assert recon.model is not None
    assert recon.spatial_dim == 2
    assert recon.gene_dim == 32
    assert recon.num_classes == 4


def test_reconstructor_reconstruct_shapes():
    """Verify reconstructed volume AnnData has spatial_3d, z_3d, source_slice."""
    from unittest.mock import patch
    from aether_3d.flow.integrators import ode as _real_ode

    # Work around t0==t1==0 (first depth) which torchdiffeq rejects.
    # The real fix belongs in the ODE integrator; this patch is test-only.
    def _safe_ode(
        drift,
        *,
        t0=0.0,
        t1=1.0,
        num_steps=None,
        solver_type="dopri5",
        atol=1e-5,
        rtol=1e-5,
        device=None,
    ):
        if abs(t1 - t0) < 1e-8:

            def sample(x):
                return x

            return sample
        return _real_ode(
            drift,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            solver_type=solver_type,
            atol=atol,
            rtol=rtol,
            device=device,
        )

    rng = np.random.default_rng(23)
    n_cells = 50
    adata_list = []
    for z in (0.0, 1.0):
        adata = ad.AnnData(
            X=rng.normal(size=(n_cells, 32)).astype(np.float32),
            obs={
                "cell_type": np.random.choice(
                    ["T", "B", "Myeloid", "Epithelial"], n_cells
                ),
                "z_coord": [z] * n_cells,
            },
        )
        adata.obsm["spatial"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
        adata_list.append(adata)

    cfg = Aether3DConfig(
        label_key="cell_type",
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_volume=100,
        thickness=5.0,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)

    with patch("aether_3d.core.aether_reconstructor.ode", _safe_ode):
        volume = recon.reconstruct_continuous_volume(
            adata_list, n_samples=100, num_depths=3
        )

    assert "spatial_3d" in volume.obsm
    assert "z_3d" in volume.obs
    assert "source_slice" in volume.obs
    assert volume.n_obs > 0


def test_ode_integrator_smoke():
    """Verify ODE integrator produces correct output shape for simple linear drift."""

    def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    x_start = torch.randn(10, 4)
    integrator = ode_integrator(
        drift, t0=0.0, t1=1.0, num_steps=10, solver_type="euler"
    )
    x_end = integrator(x_start)

    assert x_end.shape == x_start.shape
    # dx/dt = -x  →  x(t) = x0 * exp(-t)  →  x(1) ≈ 0.368 * x0
    expected = x_start * np.exp(-1.0)
    assert torch.allclose(x_end, expected, atol=0.15)


def test_flow_sampler_ode_shape():
    """Verify FlowSampler.sample_ode produces output of the requested shape."""

    class TrivialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)  # ensures model has parameters

        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    transport = create_flow_transport(path="linear", prediction="velocity")
    model = TrivialModel()
    sampler = FlowSampler(transport, model)

    shape = (4, 8)
    out = sampler.sample_ode(shape=shape, num_steps=5, solver="euler")

    assert out.shape == shape


def _make_aether_slices(n_slices=3, n_cells=12, n_genes=8, seed=101):
    rng = np.random.default_rng(seed)
    adata_list = []
    for z in range(n_slices):
        adata = ad.AnnData(
            X=rng.normal(size=(n_cells, n_genes)).astype(np.float32),
            obs={
                "cell_class": ["T", "B"] * (n_cells // 2),
                "z_coord": [float(z)] * n_cells,
            },
        )
        adata.obsm["spatial"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
        adata_list.append(adata)
    return adata_list


def test_ode_integrator_zero_interval_returns_identity():
    x_start = torch.randn(4, 3)

    def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise AssertionError("zero-interval integration must not call drift")

    integrator = ode_integrator(drift, t0=0.0, t1=0.0, solver_type="dopri5")

    assert integrator(x_start) is x_start


def test_serial_slice_dataset_requires_at_least_two_slices():
    adata = _make_aether_slices(n_slices=1)[0]
    cfg = Aether3DConfig(hidden_size=8, depth=1, num_heads=2, patch_size=4)

    with pytest.raises(ValueError, match=">=2 slices"):
        SerialSliceTrajectoryDataset([adata], cfg)


def test_serial_slice_dataset_uses_config_seed_for_uot_pairs():
    cfg = Aether3DConfig(
        seed=42,
        hidden_size=8,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_base=20,
    )

    np.random.seed(0)
    pairs_a = list(SerialSliceTrajectoryDataset(_make_aether_slices(), cfg).pairs)
    np.random.seed(999)
    pairs_b = list(SerialSliceTrajectoryDataset(_make_aether_slices(), cfg).pairs)

    assert pairs_a == pairs_b


def test_reconstruct_continuous_volume_does_not_duplicate_interior_z_planes():
    adata_list = _make_aether_slices(n_slices=3, n_cells=12, n_genes=8)
    cfg = Aether3DConfig(
        seed=42,
        hidden_size=8,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_base=12,
        n_samples_volume=12,
        thickness=10.0,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)

    volume = recon.reconstruct_continuous_volume(
        adata_list, thickness=10.0, n_samples=12, num_depths=3
    )

    z_by_source = {
        int(source): set(group["z_3d"].astype(float))
        for source, group in volume.obs.groupby("source_slice")
    }
    assert z_by_source[0] == {0.0, 5.0, 10.0}
    assert z_by_source[1] == {15.0, 20.0}
    assert z_by_source[0].isdisjoint(z_by_source[1])


def test_reconstruct_continuous_volume_uses_config_seed_per_call():
    adata_list = _make_aether_slices(n_slices=2, n_cells=12, n_genes=8)
    cfg = Aether3DConfig(
        seed=42,
        hidden_size=8,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_base=12,
        n_samples_volume=12,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)

    vol_a = recon.reconstruct_continuous_volume(adata_list, n_samples=12, num_depths=2)
    vol_b = recon.reconstruct_continuous_volume(adata_list, n_samples=12, num_depths=2)

    assert np.allclose(vol_a.obsm["spatial_3d"], vol_b.obsm["spatial_3d"])
    assert np.allclose(np.asarray(vol_a.X), np.asarray(vol_b.X))


def test_uot_pytorch_warns_and_stays_finite_on_underflow_risk():
    cost = torch.tensor([[100.0, 0.0], [50.0, 200.0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(123)

    with pytest.warns(RuntimeWarning, match="underflow risk"):
        src, tgt, weights = compute_uot_coupling(
            cost, reg=0.8, tau=0.05, n_samples=16, torch_generator=generator
        )

    assert len(src) == len(tgt) == len(weights) == 16
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)


def test_verify_aether_pipeline_data_root_is_repo_local():
    from pathlib import Path
    import scripts.e2e.verify_aether_pipeline as verify

    data_root = (
        Path(verify.__file__).resolve().parents[2]
        / "data"
        / "baselines"
        / "deepspatial"
        / "merfish_mouse_hypothalamus"
    )

    assert data_root.parts[-5] == "aether-3d"


def test_pyproject_pins_pytorch_lightning_and_pytest_pythonpath():
    """Regression: keep pytorch-lightning in runtime deps (#17) and the
    pytest pythonpath block intact (#20) so editable installs + bare
    pytest keep working."""
    import sys
    from pathlib import Path

    if sys.version_info >= (3, 11):
        import tomllib  # type: ignore[attr-defined]
    else:  # pragma: no cover - python 3.10 path
        import tomli as tomllib  # type: ignore[no-redef]

    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)

    runtime = cfg["project"]["dependencies"]
    assert any(
        spec.replace("_", "-").startswith(("pytorch-lightning", "lightning"))
        for spec in runtime
    ), f"pytorch-lightning must be a runtime dep (issue #17); got {runtime}"

    pytest_block = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert pytest_block.get("pythonpath") == ["src", "."], (
        f"pytest ini_options.pythonpath drifted (issue #20); got {pytest_block}"
    )
    assert pytest_block.get("testpaths") == ["tests"], (
        f"pytest ini_options.testpaths drifted (issue #20); got {pytest_block}"
    )
