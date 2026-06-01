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
    output_html: str | None = None,
    as_mesh: bool = False,
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
    output_html : str or None, default None
        File path to save the interactive HTML export.
    as_mesh : bool, default False
        If True, reconstructs a 3D Delaunay surface mesh of the tissue slices.
    """
    # 1. Extract coordinates
    coords_key = "spatial_3d" if "spatial_3d" in adata.obsm else "spatial"
    if coords_key in adata.obsm:
        coords = adata.obsm[coords_key]
        if coords.shape[1] < 3:
            z_col = [col for col in adata.obs.columns if col.lower() in ("z", "z_level", "z_coord", "z_3d", "slice_z")]
            if z_col:
                coords = np.column_stack((coords, adata.obs[z_col[0]].values))
            else:
                raise ValueError(f"adata.obsm['{coords_key}'] must have at least 3 dimensions for 3D plot, found shape {coords.shape} and no Z column in obs.")
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
    else:
        # Check obs columns
        keys = adata.obs.columns
        z_col = [col for col in keys if col.lower() in ("z", "z_level", "z_coord", "z_3d", "slice_z")]
        x_col = [col for col in keys if col.lower() in ("x", "x_coord")]
        y_col = [col for col in keys if col.lower() in ("y", "y_coord")]
        if not z_col or not x_col or not y_col:
            raise ValueError("Could not find spatial coordinates in adata.obsm['spatial'], adata.obsm['spatial_3d'], or (x, y, z) obs columns.")
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
        # If it's a category/string/bool, map it to integer codes.  Previously
        # the branch tested ``isinstance(vals.dtype, object)`` which is *always*
        # True (every dtype instance is an ``object``), so numeric ``obs``
        # columns were silently collapsed to category codes.  See issue #116.
        is_categorical = (
            isinstance(vals.dtype, pd.CategoricalDtype)
            or vals.dtype.kind in "OSUb"
        )
        if is_categorical:
            categories = pd.Categorical(vals)
            vals = categories.codes

    # 3. Plotting
    plot_title = title or f"3D Tissue Reconstruction (colored by {color_by})"

    if use_pyvista and _HAS_PYVISTA:
        points = np.column_stack((x, y, z))
        point_cloud = pv.PolyData(points)
        point_cloud[color_by] = vals

        plotter = pv.Plotter(title=plot_title, off_screen=(output_html is not None))
        
        if as_mesh:
            try:
                # Reconstruct Delaunay 3D mesh
                mesh = point_cloud.delaunay_3d(alpha=5.0)
                plotter.add_mesh(
                    mesh,
                    scalars=color_by,
                    cmap=cmap,
                    opacity=0.8,
                    show_edges=True,
                )
            except Exception as e:
                print(f"[WARNING] Delaunay 3D failed: {e}. Falling back to point cloud.")
                plotter.add_mesh(
                    point_cloud,
                    scalars=color_by,
                    cmap=cmap,
                    point_size=point_size,
                    render_points_as_spheres=True,
                )
        else:
            plotter.add_mesh(
                point_cloud,
                scalars=color_by,
                cmap=cmap,
                point_size=point_size,
                render_points_as_spheres=True,
            )
        
        plotter.show_grid()
        if output_html is not None:
            try:
                plotter.export_html(output_html)
                print(f"Successfully exported PyVista 3D tissue HTML to {output_html}")
            except Exception as e:
                print(f"[WARNING] PyVista HTML export failed: {e}. Falling back to Plotly export.")
                use_pyvista = False
            finally:
                plotter.close()
        
        if output_html is None or not use_pyvista:
            if output_html is None:
                plotter.show()

    # Note: we might fall back to Plotly if export_html fails or use_pyvista is False
    if not use_pyvista or not _HAS_PYVISTA:
        if _HAS_PLOTLY:
            traces = []
            if as_mesh:
                try:
                    traces.append(go.Mesh3d(
                        x=x, y=y, z=z,
                        intensity=vals,
                        colorscale=cmap,
                        opacity=0.6,
                        alphahull=5.0,
                        showscale=True,
                        colorbar=dict(title=color_by)
                    ))
                except Exception as e:
                    print(f"[WARNING] Plotly Mesh3D failed: {e}. Falling back to Scatter3D.")
            
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=vals,
                    colorscale=cmap,
                    opacity=0.8,
                    colorbar=dict(title=color_by) if not as_mesh else None
                )
            ))
            
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=plot_title,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )
            
            if output_html is not None:
                fig.write_html(output_html)
                print(f"Successfully exported Plotly 3D tissue HTML to {output_html}")
            else:
                fig.show()
        else:
            raise ImportError("Neither pyvista nor plotly is installed. Install one to render 3D plots.")
