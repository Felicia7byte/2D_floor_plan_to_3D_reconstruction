import cv2
import numpy as np
import trimesh


def create_floorplan_3d(
    mask,
    wall_height=3.0,
    wall_thickness=0.15,
    pixel_size=0.02,
    floor_thickness=0.12,
):
    """
    Class ID:
        0 = background
        1 = floor
        2 = wall
        3 = door
        4 = window

    Membuat floor plan 3D tanpa ceiling.
    Koordinat Z adalah sumbu vertikal.
    """

    scene = trimesh.Scene()

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("mask harus berupa array 2D berisi class ID.")

    h, w = mask.shape

    floor_id = 1
    wall_id = 2
    door_id = 3
    window_id = 4

    def add_box(name, size, center, color):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        mesh.visual.face_colors = color
        scene.add_geometry(mesh, node_name=name)

    # --------------------------------------------------
    # 1. FLOOR: slab dasar, tanpa penutup bagian atas
    # --------------------------------------------------
    add_box(
        name="floor_slab",
        size=(
            w * pixel_size,
            h * pixel_size,
            floor_thickness,
        ),
        center=(
            w * pixel_size / 2,
            h * pixel_size / 2,
            -floor_thickness / 2,
        ),
        color=[190, 190, 190, 255],
    )

    # --------------------------------------------------
    # 2. WALL: gabungkan piksel wall dalam setiap baris
    #    menjadi segmen horizontal, bukan bounding box
    #    besar yang menutupi interior.
    # --------------------------------------------------
    wall_mask = (mask == wall_id).astype(np.uint8)

    for y in range(h):
        row = wall_mask[y]
        padded = np.pad(row, (1, 1), constant_values=0)
        changes = np.diff(padded.astype(np.int8))

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for j, (x0, x1) in enumerate(zip(starts, ends)):
            length = (x1 - x0) * pixel_size

            if length <= 0:
                continue

            add_box(
                name=f"wall_row{y}_seg{j}",
                size=(
                    length,
                    pixel_size,
                    wall_height,
                ),
                center=(
                    (x0 + x1) * pixel_size / 2,
                    (y + 0.5) * pixel_size,
                    wall_height / 2,
                ),
                color=[225, 225, 220, 255],
            )

    # --------------------------------------------------
    # 3. DOOR / WINDOW
    # Belum dibuat menjadi objek tersendiri.
    # Piksel door dan window tidak diekstrusi sebagai wall,
    # sehingga dapat terlihat sebagai area terbuka.
    # --------------------------------------------------

    # Tidak ada geometri ceiling yang dibuat.

    return scene