import ast
import inspect

from scripts.e2e import verify_aether_pipeline


def test_bounded_smoke_reconstruction_sample_count_is_explicit_and_small() -> None:
    assert verify_aether_pipeline.SMOKE_RECONSTRUCTION_SAMPLES == 100
    assert verify_aether_pipeline.SMOKE_RECONSTRUCTION_SAMPLES < 10_000
    assert verify_aether_pipeline.SMOKE_CELLS_PER_SLICE == 16

    source = inspect.getsource(verify_aether_pipeline.main)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reconstruct_continuous_volume"
    ]

    assert len(calls) == 1
    n_samples_keywords = [kw for kw in calls[0].keywords if kw.arg == "n_samples"]
    assert len(n_samples_keywords) == 1
    assert isinstance(n_samples_keywords[0].value, ast.Name)
    assert n_samples_keywords[0].value.id == "SMOKE_RECONSTRUCTION_SAMPLES"
