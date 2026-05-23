"""
import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
import nibabel as nib
import pyvista as pv
from nilearn.datasets import load_mni152_template

print("Loading MNI152 template...")
img = load_mni152_template(resolution=1)
data = img.get_fdata()

# Normalize
data = (data - data.min()) / (data.max() - data.min())

# Convert to pyvista
grid = pv.ImageData()
grid.dimensions = np.array(data.shape) + 1
grid.spacing = img.header.get_zooms()
grid.cell_data["values"] = data.flatten(order="F")

brain = grid.threshold(0.2)

zooms = img.header.get_zooms()
patch_size_xy = 20

# (x, y, z, depth)
patch_positions = [
    (55,  60, 90, 90),    # moved inward on x
    (110, 95, 85, 60),
    (55,  85, 45, 80),
    (105, 60, 50, 80),
    (75,  70, 95, 80),
    (80,  100, 40, 80),
]

patches = []
for (x, y, z, depth) in patch_positions:
    box = pv.Box(bounds=(
        x * zooms[0], (x + patch_size_xy) * zooms[0],
        y * zooms[1], (y + depth) * zooms[1],
        z * zooms[2], (z + patch_size_xy) * zooms[2],
    ))
    patches.append(box)

print("Rendering...")
plotter = pv.Plotter(off_screen=True)

plotter.add_mesh(brain, scalars="values", opacity=0.15,
                 cmap="gray", show_scalar_bar=False)

for patch in patches:
    plotter.add_mesh(patch, color="#888888", opacity=0.8,
                     show_edges=True, edge_color="white", line_width=1.0)

plotter.camera_position = [(600, -500, 300), (90, 108, 90), (0, 0, 1)]

print("Saving...")
plotter.screenshot("brain_masked.png", window_size=[2000, 2000], transparent_background=True)
print("Done!")

"""

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
import nibabel as nib
import pyvista as pv
from nilearn.datasets import load_mni152_template

print("Loading MNI152 template...")
img = load_mni152_template(resolution=1)
data = img.get_fdata()

# Normalize
data = (data - data.min()) / (data.max() - data.min())

# Convert to pyvista
grid = pv.ImageData()
grid.dimensions = np.array(data.shape) + 1
grid.spacing = img.header.get_zooms()
grid.cell_data["values"] = data.flatten(order="F")

brain = grid.threshold(0.2)

print("Rendering...")
plotter = pv.Plotter(off_screen=True)

plotter.add_mesh(brain, scalars="values", opacity=0.15,
                 cmap="gray", show_scalar_bar=False)

# Exact same camera position
plotter.camera_position = [(600, -500, 300), (90, 108, 90), (0, 0, 1)]

print("Saving...")
plotter.screenshot("brain_unmasked.png", window_size=[2000, 2000], transparent_background=True)
print("Done!")