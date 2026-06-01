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


def test_uot_pytorch_stays_finite_on_underflow_risk_without_warning():
    # Formerly asserted the #23 fp64-promotion "underflow risk" warning; the
    # log-domain solver (#134) handles this regime silently, so we now assert
    # the coupling stays finite and no underflow warning is emitted.
    import warnings

    cost = torch.tensor([[100.0, 0.0], [50.0, 200.0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(123)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src, tgt, weights = compute_uot_coupling(
            cost, reg=0.8, tau=0.05, n_samples=16, torch_generator=generator
        )

    messages = " ".join(str(w.message).lower() for w in caught)
    assert "underflow" not in messages, f"unexpected underflow warning: {messages!r}"
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
        / "serial3d_ref"
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


# ---------------------------------------------------------------------------
# Issue #140 — unified, shared training-time sampler
# ---------------------------------------------------------------------------


def test_sample_time_is_deterministic_and_respects_train_eps():
    """The shared sampler is reproducible for a fixed seed and honours
    train_eps (which the module's old hardcoded [0.01, 0.99] range bypassed)."""
    transport = create_flow_transport(
        path="linear", prediction="score", train_eps=0.1
    )
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    t1 = transport.sample_time(64, torch.device("cpu"), generator=g1)
    t2 = transport.sample_time(64, torch.device("cpu"), generator=g2)

    assert torch.equal(t1, t2)
    assert torch.all(t1 >= 0.1) and torch.all(t1 <= 1.0)


def test_module_and_transport_share_one_time_sampler():
    """Both training paths (FlowTransport.training_losses and
    AetherFlowModule.training_step) route time sampling through the *same*
    transport.sample_time, so for a fixed seed they draw identical t and the
    historical range inconsistency ([0.01,0.99] vs [train_eps,1]) is gone."""
    from aether_3d.modules.aether_flow_module import AetherFlowModule

    class TimeCaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))
            self.seen_t = None

        def forward(self, state, t, class_condition):
            self.seen_t = t.detach().clone()
            return {
                "vx": torch.zeros_like(state["x"]) + self.dummy,
                "vg": torch.zeros_like(state["g"]) + self.dummy,
                "vc": torch.zeros_like(state["c"]) + self.dummy,
            }

    cfg = Aether3DConfig(hidden_size=8, depth=1, num_heads=2, patch_size=4)
    model = TimeCaptureModel()
    module = AetherFlowModule(cfg, model)  # type: ignore[arg-type]
    n = 5
    batch = {
        "x0": torch.zeros(n, 2), "g0": torch.zeros(n, 4),
        "c0": torch.eye(2)[torch.zeros(n, dtype=torch.long)],
        "x1": torch.ones(n, 2), "g1": torch.ones(n, 4),
        "c1": torch.eye(2)[torch.ones(n, dtype=torch.long)],
        "z0": torch.zeros(n, 1), "z1": torch.ones(n, 1),
        "delta_z": torch.ones(n, 1),
    }

    # The first stochastic op in training_step is the time sample, so seeding
    # the global RNG then replaying transport.sample_time must reproduce it.
    torch.manual_seed(123)
    module.training_step(batch, 0)
    t_module = model.seen_t

    torch.manual_seed(123)
    t_replay = module.transport.sample_time(n, torch.device("cpu"))
    assert torch.equal(t_module, t_replay)

    # And FlowTransport.training_losses uses the same sampler: training_losses
    # draws x0 = randn_like(x1) first, then t = sample_time(...).
    class VelModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def forward(self, xt, t, **kwargs):
            return torch.zeros_like(xt) + self.dummy

    transport = module.transport
    x1 = torch.randn(n, 3)
    torch.manual_seed(7)
    out = transport.training_losses(VelModel(), x1)
    torch.manual_seed(7)
    _ = torch.randn_like(x1)
    t_expected = transport.sample_time(n, x1.device)
    assert torch.equal(out["t"], t_expected)


# ---------------------------------------------------------------------------
# Issue #88 — POT <-> PyTorch UOT "mathematical parity" (claimed, now tested)
# ---------------------------------------------------------------------------


def _reference_uot_plan(cost_np, reg, tau, max_iter=1000, tol=1e-6):
    """Normalized unbalanced-Sinkhorn plan computed with the documented solver
    math (the recursion compute_uot_coupling_pytorch implements), in float64.

    This is a fixed reference for the parity claim; it does not call into POT
    (no vendoring of POT internals) and is compared against the installed
    ``ot.sinkhorn_unbalanced`` API below.
    """
    cost = torch.as_tensor(cost_np, dtype=torch.float64)
    n0, n1 = cost.shape
    a_t = torch.ones(n0, dtype=torch.float64) / n0
    b_t = torch.ones(n1, dtype=torch.float64) / n1
    K = torch.exp(-cost / reg) * (a_t.unsqueeze(1) * b_t.unsqueeze(0))
    u = torch.ones(n0, dtype=torch.float64)
    v = torch.ones(n1, dtype=torch.float64)
    fi = tau / (tau + reg)
    for _ in range(max_iter):
        up, vp = u.clone(), v.clone()
        u = (a_t / torch.clamp(K @ v, min=1e-12)) ** fi
        v = (b_t / torch.clamp(K.T @ u, min=1e-12)) ** fi
        eu = torch.max(torch.abs(u - up)) / torch.clamp(
            torch.maximum(torch.max(torch.abs(u)), torch.max(torch.abs(up))), min=1.0
        )
        ev = torch.max(torch.abs(v - vp)) / torch.clamp(
            torch.maximum(torch.max(torch.abs(v)), torch.max(torch.abs(vp))), min=1.0
        )
        if 0.5 * (eu + ev) < tol:
            break
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return (P / P.sum()).numpy()


def test_uot_pytorch_matches_pot_numerically():
    """Pin the docstring claim that compute_uot_coupling_pytorch has
    'mathematical parity with POT's sinkhorn_unbalanced': the normalized
    transport plans agree to numerical tolerance on a fixed cost matrix."""
    ot = pytest.importorskip("ot")

    rng = np.random.default_rng(0)
    cost = rng.random((30, 25))
    reg, tau = 0.8, 0.05
    a, b = np.ones(30) / 30, np.ones(25) / 25

    p_pot = ot.sinkhorn_unbalanced(a, b, cost, reg, tau)
    p_pot = p_pot / p_pot.sum()
    p_torch = _reference_uot_plan(cost, reg, tau)

    max_abs = float(np.max(np.abs(p_pot - p_torch)))
    frob_rel = float(np.linalg.norm(p_pot - p_torch) / np.linalg.norm(p_pot))
    assert max_abs < 1e-9, f"max|P_pot - P_torch| = {max_abs} exceeds 1e-9"
    assert frob_rel < 1e-9, f"Frobenius rel error = {frob_rel} exceeds 1e-9"


def test_uot_pytorch_sampler_reflects_pot_plan_end_to_end():
    """The actual public sampler (compute_uot_coupling_pytorch) draws pairs
    whose empirical distribution matches POT's normalized plan — parity of the
    code path users call, not just the replicated math."""
    ot = pytest.importorskip("ot")

    rng = np.random.default_rng(0)
    cost = rng.random((30, 25))
    reg, tau = 0.8, 0.05
    a, b = np.ones(30) / 30, np.ones(25) / 25

    p_pot = ot.sinkhorn_unbalanced(a, b, cost, reg, tau)
    p_pot = p_pot / p_pot.sum()

    src, tgt, _ = compute_uot_coupling(
        torch.as_tensor(cost, dtype=torch.float32),
        reg=reg, tau=tau, n_samples=200_000,
        torch_generator=torch.Generator().manual_seed(0),
    )
    emp = np.zeros((30, 25))
    np.add.at(emp, (src.numpy(), tgt.numpy()), 1.0)
    emp = emp / emp.sum()

    corr = float(np.corrcoef(emp.ravel(), p_pot.ravel())[0, 1])
    max_abs = float(np.max(np.abs(emp - p_pot)))
    assert corr > 0.95, f"empirical/POT plan correlation {corr} too low"
    assert max_abs < 5e-3, f"empirical vs POT max abs diff {max_abs} too large"


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


# ---------------------------------------------------------------------------
# Issue #134 — log-domain stabilized unbalanced Sinkhorn UOT
# ---------------------------------------------------------------------------


def test_uot_plan_matches_pot_in_benign_regime():
    """The log-domain PyTorch solver reproduces POT's normalized plan to
    float32 tolerance in a benign regime (parity claim is now executable)."""
    ot = pytest.importorskip("ot")
    from aether_3d.coupling.uot import compute_uot_plan_pytorch

    rng = np.random.default_rng(0)
    cost = rng.random((30, 25))
    reg, tau = 0.8, 0.05
    a, b = np.ones(30) / 30, np.ones(25) / 25

    p_pot = ot.sinkhorn_unbalanced(a, b, cost, reg, tau)
    p_pot = p_pot / p_pot.sum()

    p_torch = compute_uot_plan_pytorch(
        torch.as_tensor(cost, dtype=torch.float32), reg=reg, tau=tau
    ).numpy()

    assert np.isfinite(p_torch).all()
    assert np.isclose(p_torch.sum(), 1.0, atol=1e-5)
    assert np.max(np.abs(p_pot - p_torch)) < 1e-6


def test_uot_logdomain_stays_finite_where_expdomain_collapses():
    """A large-cost / small-reg regime underflows the exp-domain kernel to a
    zero-sum (collapsed) plan, but the log-domain solver returns a finite,
    non-degenerate coupling whose mass tracks the structured optimum."""
    from aether_3d.coupling.uot import compute_uot_plan_pytorch

    n = 20
    # Banded cost (prefers the diagonal) + a large constant offset that mimics
    # the lambda_class penalty added on top of a normalized cost.
    struct = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32)
    cost_np = 100.0 + struct
    cost = torch.as_tensor(cost_np, dtype=torch.float32)
    reg, tau = 0.05, 0.05

    # Replicate the OLD exp-domain kernel in float32: it underflows to all-zero.
    a_t = torch.ones(n, dtype=torch.float32) / n
    b_t = torch.ones(n, dtype=torch.float32) / n
    K = torch.exp(-cost / reg) * (a_t.unsqueeze(1) * b_t.unsqueeze(0))
    assert float(K.sum()) == 0.0, "expected exp-domain kernel to underflow here"

    # Log-domain solver: finite, normalized, and non-degenerate (not uniform).
    p = compute_uot_plan_pytorch(cost, reg=reg, tau=tau)
    assert torch.isfinite(p).all()
    assert np.isclose(float(p.sum()), 1.0, atol=1e-5)

    # Mass concentrates on the diagonal: each row's argmax tracks its index,
    # which a collapsed/uniform plan could not achieve.
    row_argmax = p.argmax(dim=1)
    assert torch.equal(row_argmax, torch.arange(n))
    uniform = 1.0 / (n * n)
    assert float(p.max()) > 5 * uniform
