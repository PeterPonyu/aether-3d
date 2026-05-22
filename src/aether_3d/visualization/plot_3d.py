"""
Unified Multi-modal 3D Tissue Visualization (PyVista and Plotly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False
    pv = None

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    go = None


def plot_continuous_tissue_3d(
    adata: ad.AnnData,
    color_by: str,
    layer: str | None = "imputed",
    use_pyvista: bool = True,
    point_size: float = 5.0,
    cmap: str = "viridis",
    title: str | None = None,
) -> None:
    """
    Plot continuous spatial slices in 3D coordinate space.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing 3D spatial coordinates in adata.obsm['spatial']
        or in adata.obs columns ('x', 'y', 'z').
    color_by : str
        Gene name or observation metadata column to color by.
    layer : str or None, default 'imputed'
        Gene expression layer to retrieve expression values from (if color_by is a gene).
    use_pyvista : bool, default True
        If True, renders interactively with PyVista. If False or if PyVista is missing,
        uses Plotly.
    point_size : float, default 5.0
        Size of the dots in PyVista.
    cmap : str, default 'viridis'
        Colormap name.
    title : str or None, default None
        Plot title.
    """
    # 1. Extract coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        if coords.shape[1] < 3:
            raise ValueError(f"adata.obsm['spatial'] must have at least 3 dimensions for 3D plot, found shape {coords.shape}")
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
    else:
        # Check obs columns
        keys = adata.obs.columns
        z_col = [col for col in keys if col.lower() in ("z", "z_level", "z_coord", "slice_z")]
        x_col = [col for col in keys if col.lower() in ("x", "x_coord")]
        y_col = [col for col in keys if col.lower() in ("y", "y_coord")]
        if not z_col or not x_col or not y_col:
            raise ValueError("Could not find spatial coordinates in adata.obsm['spatial'] or (x, y, z) obs columns.")
        x = adata.obs[x_col[0]].values
        y = adata.obs[y_col[0]].values
        z = adata.obs[z_col[0]].values

    # 2. Extract coloring values
    is_gene = color_by in adata.var_names
    if is_gene:
        if layer is not None and layer in adata.layers:
            vals = adata[:, color_by].layers[layer]
        else:
            vals = adata[:, color_by].X
        if hasattr(vals, "toarray"):
            vals = vals.toarray()
        vals = vals.ravel()
    else:
        if color_by not in adata.obs:
            raise ValueError(f"'{color_by}' not found in adata.var_names or adata.obs.columns.")
        vals = adata.obs[color_by].values
        # If it's a category/object, map it to indices
        if isinstance(vals.dtype, (pd.CategoricalDtype, object)) or vals.dtype.kind in "OSU":
            categories = pd.Categorical(vals)
            vals = categories.codes

    # 3. Plotting
    plot_title = title or f"3D Tissue Reconstruction (colored by {color_by})"

    if use_pyvista and _HAS_PYVISTA:
        points = np.column_stack((x, y, z))
        point_cloud = pv.PolyData(points)
        point_cloud[color_by] = vals

        plotter = pv.Plotter(title=plot_title)
        plotter.add_mesh(
            point_cloud,
            scalars=color_by,
            cmap=cmap,
            point_size=point_size,
            render_points_as_spheres=True,
        )
        plotter.show_grid()
        plotter.show()
    elif _HAS_PLOTLY:
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=point_size,
                color=vals,
                colorscale=cmap,
                opacity=0.8,
                colorbar=dict(title=color_by)
            )
        )])
        fig.update_layout(
            title=plot_title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig.show()
    else:
        raise ImportError("Neither pyvista nor plotly is installed. Install one to render 3D plots.")
