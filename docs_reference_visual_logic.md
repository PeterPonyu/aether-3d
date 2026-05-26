# Aether3D reference visualization logic (clean-room)

Source read: `../2026.04.28.721395v2.full.pdf`, rendered locally under `../results/reference_paper_visual_audit/2026.04.28.721395v2.full/`.

Observed panel logic from the DEEPSPATIAL paper:

- Fig. 1 style: serial 2D slices and omics metadata flowing into continuous 3D virtual tissue and downstream multi-planar analysis.
- Fig. 2 style: ground-truth/reconstruction/baseline 3D blocks, density maps, cell-type distribution bars, and marker/correlation metrics.
- Fig. 3 style: multi-slice IMC workflow, real-vs-generated virtual slice grids, cell-type proportions, marker heatmaps, neighborhood matrix, and 3D cell-type distribution.
- Fig. 4 style: z-axis cell-type distribution, canonical marker expression across sections, density around target cell neighborhoods.
- Fig. 5 style: scalable atlas dashboard with z-depth region/cell-type composition, UMAP comparison, proportion correlations, and marker consistency.
- Fig. 6 style: multi-planar virtual slicing and 3D visualization of selected anatomical regions.

Clean-room adaptation rules used here:

- Keep only the visualization grammar: reconstruction workflow, 3D cell architecture, virtual z slices, z-depth composition, marker continuity, metric dashboard, neighborhood matrix.
- Use Aether3D-specific labels and repo claim-gating terms.
- Do not copy source paper artwork, logos, captions, exact panel ordering, or unsupported scale claims.
- Demo values are labelled as planning/demo values until replaced with local benchmark JSON.

Implemented script:

```bash
python scripts/visualize/fig_reference_visual_story.py
```

Primary output:

- `results/figures/aether_reference_visual_story.png`

## Composed main-claim figure

Implemented script:

```bash
python scripts/visualize/fig_composed_main_claim.py
```

Primary output:

- `results/figures/aether_composed_main_claim.png`

This figure is intentionally different from the earlier `aether_reference_visual_story.png`: the earlier file checks that individual panel types align with the DEEPSPATIAL visual grammar, while the composed main-claim figure arranges those panel types into a single argument chain:

1. serial 2D slices and the z-gap reconstruction problem;
2. Aether3D UOT/flow reconstruction workflow;
3. coherent 3D cellular architecture;
4. virtual z-slice validation;
5. cell-type continuity along z;
6. metric and neighborhood gates;
7. explicit evidence-tier and no-scale-claim limitation.

The figure remains `demo/planning` unless local virtual-slice holdout artifacts and real validation metrics are present. It must not be used for paper-ready fidelity or 39M-cell-style scale claims until those gates pass.
