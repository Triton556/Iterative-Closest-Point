# Iterative Closest Point for Unity

**An interactive experiment in aligning 3D geometry.** A C# point-cloud registration prototype developed for a human–machine interface: compare scanned geometry with a reference mesh and explore the transform needed to align them.

`Unity` · `C#` · `Math.NET Numerics` · `Supercluster.KDTree`

## Inside the experiment

- Uses vertices from two Unity `MeshFilter` components as point clouds.
- Finds nearest-neighbor correspondences with a KD-tree.
- Computes centroids and a quaternion-based alignment estimate with matrix operations.
- Applies the estimated position and rotation to the moving object.
- Exposes `startICP` in the Inspector for manually triggered alignment steps.

The core implementation is [`icpalg.cs`](ICP/Assets/Scripts/icpalg.cs). A demonstration scene lives at [`ICP/Assets/Scenes/SampleScene.unity`](ICP/Assets/Scenes/SampleScene.unity).

## How the code is organized

| Method | Role |
| --- | --- |
| `ParceFromVector3` | Collect mesh vertices with the object's positional offset |
| `KD_tree` | Build correspondences using nearest-neighbor queries |
| `FindCentroid` | Calculate the center of a point set |
| `ICP_run` | Estimate the transform and a summed squared correspondence error |
| `Update` | Run a requested step and apply the result to `PGameObject` |

## Try it in Unity

The repository contains assets and a historical Unity cache, but no committed `ProjectSettings/ProjectVersion.txt` or complete project configuration. Importing the relevant assets into a separate Unity project is the clearest starting point.

1. Create a Unity project and import `ICP/Assets/Scripts/icpalg.cs`.
2. Add compatible **Math.NET Numerics** and **Supercluster.KDTree** assemblies. Historical copies are present in `ICP/Assets/Plugins`; avoid importing duplicates if your project already supplies them.
3. Add the `icpalg` component to a manager object.
4. Assign the reference mesh to **M Game Object** and the moving mesh to **P Game Object**. Both require a `MeshFilter`.
5. Enter Play mode and enable **Start ICP** in the Inspector to request an update. Inspect the result and repeat as needed.

Start with simple, similarly oriented meshes with identity scale. Keep original copies of the objects while experimenting.

## Example results

The original project documents demonstrations with cubes, capsules, cylinders and a more complex tree mesh.

### Cubes

![Cubes](https://sun9-46.userapi.com/s/v1/if2/LLsv5_LFFbyqOSZd8DEgOhJGJdxTSaIbCeje6lycFvSHupRNIcyuKeuZxoDVFulGWcbuWSP3BQ2Pqggv2MNvCVhv.jpg?size=763x526&quality=96&type=album)

### Capsules

![Capsules](https://sun9-43.userapi.com/s/v1/if2/_eFutWb_v7CNqR6dWzqBD53X7nUM2-PdEBPzdD3EHNNgE-dpmXZ9D4DlY0Z1MPccq6ulb75bAIDqIVD1LY3908hp.jpg?size=831x557&quality=96&type=album)

### Cylinders

![Cylinders](https://sun9-79.userapi.com/s/v1/if2/xrvuB3qikhuyMZMkGpV2PBjxGlALb_O0TN7xHm9v36MCFFxX8hTh_wZ4qr_iagi812Kef2HmeaPOJ-MbrVq16cuK.jpg?size=851x546&quality=96&type=album)

### Tree mesh

![Tree mesh](https://sun9-49.userapi.com/s/v1/if2/zHEetqbeMc90_5f594uZcLGFLnUGiUF-vcs6lhy57XjnpaIQipgiArSVff_DUIWXLLCFQoEik1yqXHQhkn4iN8eD.jpg?size=843x535&quality=96&type=album)

## Important limitations

This is an educational prototype, not a validated registration library. The current source has known implementation limitations:

- The covariance terms currently repeat the same X-axis product; general rotation recovery needs correction and numerical tests.
- Mesh conversion adds object position but does not apply the full rotation/scale transform.
- There is no robust outlier rejection, convergence controller or benchmark suite.

The screenshots preserve the original demonstrations; they are not a claim of newly reproduced results. The code has not been changed as part of this documentation refresh.

Bundled libraries and third-party assets retain their own licensing terms. No new blanket license is assigned by this README.
