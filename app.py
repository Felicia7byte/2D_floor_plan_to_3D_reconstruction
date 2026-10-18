import streamlit as st
import numpy as np
import pyvista as pv

from PIL import Image
from stpyvista import stpyvista

from inference import load_model, predict_mask
from reconstruction import create_floorplan_3d


st.set_page_config(
    page_title="2D Floorplan to 3D",
    layout="wide"
)

st.title("2D Floorplan to 3D Room Reconstruction")

CHECKPOINT = "best_unet_floorplan(1).pth"


@st.cache_resource
def get_model():
    return load_model(CHECKPOINT)


uploaded_file = st.file_uploader(
    "Upload 2D floor plan",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.header("Building Dimensions")

wall_height = st.sidebar.slider(
    "Height of wall (m)",
    2.0, 5.0, 3.0, 0.1
)

wall_thickness = st.sidebar.slider(
    "Wall thickness (m)",
    0.05, 0.40, 0.15, 0.01
)

floor_thickness = st.sidebar.slider(
    "floor thickness (m)",
    0.05, 0.30, 0.12, 0.01
)

pixel_size = st.sidebar.number_input(
    "Scale (meter/pixel)",
    min_value=0.001,
    max_value=0.2,
    value=0.02,
    step=0.001,
    format="%.3f"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Denah")
    st.image(image, use_container_width=True)

    if st.button("3D Reconstruction", type="primary"):
        try:
            model = get_model()
            mask = predict_mask(model, image)

            scene = create_floorplan_3d(
                mask=mask,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                pixel_size=pixel_size,
                floor_thickness=floor_thickness,
            )

            st.session_state["scene"] = scene

        except Exception as e:
            st.error(f"Proses gagal: {e}")

    if "scene" in st.session_state:
        st.subheader("3D Model")

        # Visualisasi sementara dari geometri trimesh.
        scene = st.session_state["scene"]
        plotter = pv.Plotter(
            window_size=[900, 650],
            notebook=False
        )

        for name, mesh in scene.geometry.items():
            vertices = mesh.vertices
            faces = np.hstack([
                np.full((len(mesh.faces), 1), 3),
                mesh.faces
            ]).astype(np.int64)

            poly = pv.PolyData(vertices, faces)

            plotter.add_mesh(
                poly,
                color="lightgray",
                show_edges=False
            )

        plotter.set_background("white")
        plotter.add_axes()
        plotter.view_isometric()

        stpyvista(plotter, key="floorplan_3d")
