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
from aether_3d.flow.integrators import sde as sde_integrator
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


def test_training_step_passes_real_class_conditioning():
    from aether_3d.modules.aether_flow_module import AetherFlowModule

    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))
            self.seen_class_condition = None

        def forward(self, state, t, class_condition):
            self.seen_class_condition = class_condition.detach().clone()
            return {
                "vx": torch.zeros_like(state["x"]) + self.dummy,
                "vg": torch.zeros_like(state["g"]) + self.dummy,
                "vc": torch.zeros_like(state["c"]) + self.dummy,
            }

    cfg = Aether3DConfig(hidden_size=8, depth=1, num_heads=2, patch_size=4)
    model = CaptureModel()
    module = AetherFlowModule(cfg, model)  # type: ignore[arg-type]
    batch = {
        "x0": torch.zeros(2, 2),
        "g0": torch.zeros(2, 4),
        "c0": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "x1": torch.ones(2, 2),
        "g1": torch.ones(2, 4),
        "c1": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "z0": torch.zeros(2, 1),
        "z1": torch.ones(2, 1),
        "delta_z": torch.ones(2, 1),
    }

    module.training_step(batch, 0)

    assert model.seen_class_condition is not None
    assert torch.equal(model.seen_class_condition, batch["c0"])


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

    # The real ode() integrator handles t0==t1 as identity (see #15);
    # no test-only monkey-patch needed.
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


def test_ode_fixed_step_num_steps_counts_intervals():
    calls = []

    def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        calls.append(float(t[0].item()))
        return torch.ones_like(x)

    x_start = torch.zeros(2, 1)
    integrator = ode_integrator(
        drift, t0=0.0, t1=1.0, num_steps=4, solver_type="euler"
    )
    x_end = integrator(x_start)

    assert torch.allclose(x_end, torch.ones_like(x_start))
    assert len(calls) == 4


def test_sde_sampler_type_changes_heun_drift_path_with_zero_diffusion():
    def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    def diffusion(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    x_start = torch.ones(3, 1)
    euler = sde_integrator(
        drift, diffusion, t0=0.0, t1=1.0, num_steps=1, sampler_type="Euler"
    )
    heun = sde_integrator(
        drift, diffusion, t0=0.0, t1=1.0, num_steps=1, sampler_type="Heun"
    )

    assert torch.allclose(euler(x_start), torch.full_like(x_start, 2.0))
    assert torch.allclose(heun(x_start), torch.full_like(x_start, 2.5))


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


def test_serial_slice_dataset_rejects_empty_input():
    # An empty list collapses to zero pairs the same way 1-slice input does,
    # so the >=2 guard must also fire here rather than producing len==0 silently.
    cfg = Aether3DConfig(hidden_size=8, depth=1, num_heads=2, patch_size=4)

    with pytest.raises(ValueError, match=">=2 slices"):
        SerialSliceTrajectoryDataset([], cfg)


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


def test_reconstruct_continuous_volume_no_duplicate_z_scales_to_n_slices():
    # Generalised regression for issue #14: every interior z-plane must be
    # owned by exactly one source pair, regardless of how many slices are
    # stacked. With n_slices=5 and num_depths=4 we should see 1 + 3 + 3 + 3
    # = a sorted union without repeats across (i, i+1) intervals.
    adata_list = _make_aether_slices(n_slices=5, n_cells=10, n_genes=8, seed=7)
    cfg = Aether3DConfig(
        seed=11,
        hidden_size=8,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_base=10,
        n_samples_volume=10,
        thickness=4.0,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)

    volume = recon.reconstruct_continuous_volume(
        adata_list, thickness=4.0, n_samples=10, num_depths=4
    )

    z_by_source = {
        int(source): set(group["z_3d"].astype(float))
        for source, group in volume.obs.groupby("source_slice")
    }
    # All-pairs disjointness: no z is owned by more than one source pair.
    for i in z_by_source:
        for j in z_by_source:
            if i < j:
                assert z_by_source[i].isdisjoint(z_by_source[j]), (
                    f"interior z double-write between pair {i} and pair {j}"
                )


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


def test_aether_reconstructor_fit_is_reproducible_across_runs(tmp_path):
    """End-to-end reproducibility regression for #24 + #26: two fresh
    fit() runs against identical data with the same cfg.seed must
    produce identical first-step train loss values."""

    def _build_adatas(seed: int = 9):
        rng = np.random.default_rng(seed)
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
        return adata_list

    def _train_once(out_dir):
        cfg = Aether3DConfig(
            seed=42,
            n_samples_base=6,
            batch_size=2,
            max_epochs=1,
            num_workers=0,
            hidden_size=16,
            depth=1,
            num_heads=2,
            patch_size=4,
            output_dir=out_dir,
        )
        recon = AetherReconstructor(cfg)
        recon.setup_data(_build_adatas())
        trainer = pl.Trainer(
            fast_dev_run=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        recon.fit(trainer=trainer)
        return float(trainer.callback_metrics["train_loss"].item())

    loss_a = _train_once(tmp_path / "run_a")
    loss_b = _train_once(tmp_path / "run_b")
    assert loss_a == pytest.approx(loss_b, abs=1e-6), (
        f"fit() with cfg.seed=42 must be reproducible; got {loss_a} vs {loss_b}"
    )


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


def test_pyproject_declares_scikit_learn_and_pot_runtime_deps():
    """Regression (issue #119): scikit-learn is a hard, top-level import in
    aether_3d.data.trajectory_dataset (sklearn.preprocessing.LabelEncoder), so
    it must be a *declared* runtime dependency rather than relying on scanpy's
    transitive edge. Also assert pot is declared as a runtime dep with no
    'optional later' contradiction, so the packaging intent matches the code."""
    import sys
    from pathlib import Path

    if sys.version_info >= (3, 11):
        import tomllib  # type: ignore[attr-defined]
    else:  # pragma: no cover - python 3.10 path
        import tomli as tomllib  # type: ignore[no-redef]

    repo_root = Path(__file__).resolve().parents[1]
    raw = (repo_root / "pyproject.toml").read_text()
    with (repo_root / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)

    runtime = cfg["project"]["dependencies"]

    def _declared(name: str) -> bool:
        norm = name.replace("_", "-")
        return any(spec.replace("_", "-").startswith(norm) for spec in runtime)

    assert _declared("scikit-learn"), (
        "scikit-learn must be a declared runtime dependency (issue #119); "
        f"it is a hard top-level import in trajectory_dataset. Got {runtime}"
    )
    assert _declared("pot"), (
        f"pot must remain a declared runtime dependency (issue #119); got {runtime}"
    )
    assert "# optional later" not in raw, (
        "the contradictory 'pot # optional later' comment must be removed "
        "(issue #119): pot is a required runtime dependency."
    )
