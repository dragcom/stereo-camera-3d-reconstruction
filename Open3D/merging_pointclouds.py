import copy
import open3d as o3d
from pathlib import Path

folder = Path("outputs_2")

ply_files = sorted(
    p for p in folder.glob("*.ply")
    if "merged" not in p.name.lower()
)

if not ply_files:
    raise RuntimeError("No .ply files found")

voxel_size = 0.008
icp_distance = 0.05

clouds = []

for path in ply_files:
    pcd = o3d.io.read_point_cloud(str(path))
    pcd = pcd.voxel_down_sample(voxel_size)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=40,
        )
    )

    clouds.append(pcd)
    print(f"Loaded {path.name}: {len(pcd.points)} points")

merged = copy.deepcopy(clouds[0])

for i, source in enumerate(clouds[1:], start=1):
    print(f"\nAligning {ply_files[i].name}...")

    source = copy.deepcopy(source)

    merged.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=40,
        )
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        merged,
        max_correspondence_distance=icp_distance,
        init=[[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]],
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    print("Fitness:", result.fitness)
    print("RMSE:", result.inlier_rmse)

    if result.fitness < 0.35 or result.inlier_rmse > 0.035:
        print("Skipping: poor alignment")
        continue

    source.transform(result.transformation)

    merged += source
    merged = merged.voxel_down_sample(voxel_size)

    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )

output_path = folder / "merged_pointcloud.ply"
o3d.io.write_point_cloud(str(output_path), merged)

print("Saved:", output_path)
o3d.visualization.draw_geometries([merged])