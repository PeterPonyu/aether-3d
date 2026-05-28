#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import numpy as np

# Add src and project root to pythonpath
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
from aether_3d.visualization.plot_3d import plot_continuous_tissue_3d

def main(args):
    # 1. Generate synthetic serial slices
    print("[INFO] Generating synthetic serial slices for dashboard demo...")
    from scripts.data_flow.generate_serial_slices import generate_synthetic_serial_slices
    slices, class_names = generate_synthetic_serial_slices(
        n_slices=3,
        cells_per_slice=300,
        n_genes=16,
        n_classes=3,
        seed=42,
        slice_spacing=10.0
    )
    
    print(f"  Generated {len(slices)} slices with classes: {class_names}")

    # 2. Setup AetherReconstructor
    cfg = Aether3DConfig(
        hidden_size=32,
        depth=2,
        num_heads=2,
        batch_size=64,
        max_epochs=1,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(slices)
    # Using initialized model directly for visualization demo
    
    # 3. Perform 3D reconstruction
    print("[INFO] Reconstructing 3D continuous volume...")
    volume = recon.reconstruct_continuous_volume(slices, thickness=20.0, num_depths=5)
    
    # Add dummy cell types to volume based on predicted cell_class_vel probabilities
    if "cell_class_vel" in volume.obsm:
        predicted_idx = np.argmax(volume.obsm["cell_class_vel"], axis=1)
        volume.obs["cell_class"] = [class_names[idx % len(class_names)] for idx in predicted_idx]
    else:
        volume.obs["cell_class"] = "Unknown"

    # Ensure results directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Export continuous tissue HTML (point cloud mode)
    html_pts = output_dir / "reconstructed_tissue_points.html"
    print(f"[INFO] Exporting reconstructed tissue point cloud to {html_pts}...")
    plot_continuous_tissue_3d(
        volume,
        color_by="cell_class",
        use_pyvista=False,  # Fall back to Plotly for highly compatible web rendering
        output_html=str(html_pts),
        title="Continuous Reconstructed 3D Tissue (Point Cloud)"
    )
    
    # 5. Export continuous tissue HTML (Delaunay surface mesh mode)
    html_mesh = output_dir / "reconstructed_tissue_mesh.html"
    print(f"[INFO] Exporting reconstructed tissue Delaunay surface mesh to {html_mesh}...")
    plot_continuous_tissue_3d(
        volume,
        color_by="cell_class",
        use_pyvista=False,
        as_mesh=True,
        output_html=str(html_mesh),
        title="Continuous Reconstructed 3D Tissue (Delaunay Surface Mesh)"
    )

    # 6. Create a comparison Plotly dashboard showing Raw Slices vs. Aligned Continuous Volume
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("[INFO] Generating side-by-side comparison dashboard...")
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Raw Input Slices (Z=0, Z=10, Z=20)", "Aligned Reconstructed Volume (Continuous)")
    )
    
    # Color map for classes
    color_map = {name: f"rgb({(i*80)%256}, {(i*120)%256}, {(255 - i*70)%256})" for i, name in enumerate(class_names)}
    color_map["Unknown"] = "rgb(128, 128, 128)"

    # Add raw slices to subplot 1
    for i, sl in enumerate(slices):
        coords = sl.obsm["spatial"]
        z_val = sl.obs["z_coord"].iloc[0]
        c_types = sl.obs["cell_class"].values
        colors = [color_map.get(ct, "rgb(128, 128, 128)") for ct in c_types]
        
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=np.full(len(coords), z_val),
                mode='markers',
                marker=dict(size=4, color=colors, opacity=0.8),
                name=f"Raw Slice {i} (Z={z_val})"
            ),
            row=1, col=1
        )
        
    # Add reconstructed volume to subplot 2
    recon_coords = volume.obsm["spatial_3d"]
    recon_ctypes = volume.obs["cell_class"].values
    recon_colors = [color_map.get(ct, "rgb(128, 128, 128)") for ct in recon_ctypes]
    
    fig.add_trace(
        go.Scatter3d(
            x=recon_coords[:, 0],
            y=recon_coords[:, 1],
            z=recon_coords[:, 2],
            mode='markers',
            marker=dict(size=3, color=recon_colors, opacity=0.7),
            name="Aligned Volume (Continuous)"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Aether3D: Raw Spatial Slices vs. 3D Reconstructed Tissue Volume",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        scene2=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    dashboard_html = output_dir / "multi_slice_dashboard.html"
    fig.write_html(str(dashboard_html))
    print(f"[SUCCESS] Multi-slice comparison dashboard written to {dashboard_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results", help="Directory to save output HTML dashboards")
    args = parser.parse_args()
    main(args)
