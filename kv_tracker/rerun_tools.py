import numpy as np
import rerun as rr


def rr_viz_cam(cam_name, T_wc, fx=400, fy=400, w=600, h=400, scale=15.0, color=[200, 200, 200]):
    # rr.log(cam_name, rr.Transform3D(translation=np.round(T_wc[:3, 3], decimals=2), mat3x3=T_wc[:3, :3], axis_length=0))
    rr.log(cam_name, rr.Transform3D(translation=np.round(T_wc[:3, 3], decimals=2), mat3x3=T_wc[:3, :3]))

    # rr.log(cam_name, rr.ViewCoordinates.RDF)
    rr.log(
        cam_name,
        rr.Pinhole(
            focal_length=[fx, fy], width=w, height=h, image_plane_distance=scale, color=color
        ),
    )


def rr_viz_mesh(name, mesh):
    rr.log(
        name,
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_normals=mesh.vertex_normals,
            triangle_indices=mesh.faces,
        )
    )


def rr_viz_pts3d(name, pts3d, pts_colors=None):
    """
    Visualize 3D points in rerun.
    """
    rr.log(name, rr.Points3D(pts3d, colors=pts_colors, radii=0.05))


def rr_viz_origin():
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

def rr_viz_pose(name, T_wc):
    rr.log(
        name,
        rr.Transform3D(
            translation=T_wc[:3, 3],
            mat3x3=T_wc[:3, :3],
        ),
    )
    arrow_len = 0.05
    rr.log(name, rr.ViewCoordinates.RDF)
    rr.log(
        name,
        rr.Arrows3D(
            vectors=[[arrow_len, 0, 0], [0, arrow_len, 0], [0, 0, arrow_len]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )
