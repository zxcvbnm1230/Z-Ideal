//! Vulkan resource resolution for renderer-neutral runtime scene snapshots.
use std::collections::{BTreeMap, BTreeSet};

use ash::vk;
use z_ideal_assets::{
    AssetCache, AssetHandle, AssetId, AssetKind, AssetUri, GltfImport, ImportedMaterial,
};
use z_ideal_renderer::{
    BufferBarrier, CommandBufferRecording, MeshDrawError, MeshGpu, MeshUploadError, VulkanDevice,
    VulkanError, VulkanPipelineLayout,
};

use crate::RenderSceneSnapshot;

type MeshKey = (AssetUri, usize);
type PrimitiveKey = (AssetUri, usize, usize);

/// Owns uploaded GPU primitives and pins their imported CPU assets.
///
/// The resources borrow the Vulkan device and must be dropped before that device.
/// This adapter does not create a pipeline or render pass; callers bind their
/// compatible pipeline and own viewport, scissor, and submission policy.
pub struct VulkanSceneMeshes<'device> {
    assets: BTreeMap<AssetUri, AssetHandle<GltfImport>>,
    meshes: BTreeMap<PrimitiveKey, MeshGpu<'device>>,
    primitive_counts: BTreeMap<MeshKey, usize>,
}

impl<'device> VulkanSceneMeshes<'device> {
    /// Resolves every distinct mesh referenced by the snapshot and uploads its primitives.
    ///
    /// Asset IDs use the `Mesh` kind and the URI stored in each ECS mesh instance.
    /// Construction is transactional: if resolution or any primitive upload fails,
    /// already-created temporary GPU resources are dropped and no partial adapter is returned.
    pub fn upload(
        device: &'device VulkanDevice,
        cache: &AssetCache<GltfImport>,
        snapshot: &RenderSceneSnapshot,
    ) -> Result<Self, VulkanSceneMeshesError> {
        let required_meshes: BTreeSet<MeshKey> = snapshot
            .meshes
            .iter()
            .map(|instance| {
                (
                    instance.source.source_uri.clone(),
                    instance.source.mesh_index,
                )
            })
            .collect();

        let mut assets = BTreeMap::new();
        let mut meshes = BTreeMap::new();
        let mut primitive_counts = BTreeMap::new();

        for (uri, mesh_index) in required_meshes {
            let asset_id = AssetId::from_uri(AssetKind::Mesh, &uri);
            let handle = cache
                .get(asset_id)
                .ok_or_else(|| VulkanSceneMeshesError::MissingAsset { uri: uri.clone() })?;
            let matching_meshes: Vec<_> = handle
                .get()
                .meshes
                .iter()
                .filter(|mesh| mesh.source_index == mesh_index)
                .collect();
            let imported_mesh = match matching_meshes.as_slice() {
                [] => {
                    return Err(VulkanSceneMeshesError::MissingMesh { uri, mesh_index });
                }
                [mesh] => *mesh,
                _ => {
                    return Err(VulkanSceneMeshesError::AmbiguousMesh { uri, mesh_index });
                }
            };
            if imported_mesh.primitives.is_empty() {
                return Err(VulkanSceneMeshesError::EmptyMesh { uri, mesh_index });
            }

            let mut uploaded = Vec::new();
            uploaded
                .try_reserve_exact(imported_mesh.primitives.len())
                .map_err(|error| VulkanSceneMeshesError::HostAllocation(error.to_string()))?;
            for (primitive_index, primitive) in imported_mesh.primitives.iter().enumerate() {
                let gpu_mesh = MeshGpu::upload(device, primitive).map_err(|source| {
                    VulkanSceneMeshesError::PrimitiveUpload {
                        uri: uri.clone(),
                        mesh_index,
                        primitive_index,
                        source,
                    }
                })?;
                uploaded.push((primitive_index, gpu_mesh));
            }

            let primitive_count = uploaded.len();
            for (primitive_index, gpu_mesh) in uploaded {
                meshes.insert((uri.clone(), mesh_index, primitive_index), gpu_mesh);
            }
            primitive_counts.insert((uri.clone(), mesh_index), primitive_count);
            assets.insert(uri, handle);
        }

        Ok(Self {
            assets,
            meshes,
            primitive_counts,
        })
    }

    /// Records host-write visibility for all uploaded vertex and index buffers.
    ///
    /// Call this before beginning the render pass that will consume these meshes.
    /// Vulkan does not permit this ordinary pipeline barrier inside a render pass.
    /// It is safe to call once per command buffer before the first use of these resources.
    pub fn record_upload_barriers<'recording, 'pool>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
    ) -> Result<(), VulkanSceneMeshesError> {
        let mut barriers = Vec::new();
        for mesh in self.meshes.values() {
            barriers.push(BufferBarrier {
                buffer: mesh.vertex_buffer(),
                source_access: vk::AccessFlags::HOST_WRITE,
                destination_access: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            });
            barriers.push(BufferBarrier {
                buffer: mesh.index_buffer(),
                source_access: vk::AccessFlags::HOST_WRITE,
                destination_access: vk::AccessFlags::INDEX_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            });
        }
        if !barriers.is_empty() {
            recording.pipeline_barrier(
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
            )?;
        }
        Ok(())
    }

    /// Records every snapshot mesh primitive as an indexed draw.
    ///
    /// The caller must record `record_upload_barriers` before the consuming render
    /// pass, then bind a graphics pipeline compatible with `MeshGpu::vertex_binding`,
    /// `MeshGpu::vertex_attributes`, and the 64-byte vertex push-constant range
    /// before calling this method.
    ///
    /// Returns the number of indexed primitive draws recorded.
    pub fn record_draws<'recording, 'pool>(
        &'recording self,
        snapshot: &RenderSceneSnapshot,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
        layout: &'recording VulkanPipelineLayout<'_, 'device>,
    ) -> Result<usize, VulkanSceneMeshesError> {
        let view_projection = snapshot.projection * snapshot.view;
        if !view_projection.is_finite() {
            return Err(VulkanSceneMeshesError::NonFiniteViewProjection);
        }

        if snapshot.meshes.is_empty() {
            return Ok(0);
        }

        let mut draw_count = 0;
        for instance in &snapshot.meshes {
            let mesh_key = (
                instance.source.source_uri.clone(),
                instance.source.mesh_index,
            );
            let primitive_count =
                self.primitive_counts
                    .get(&mesh_key)
                    .copied()
                    .ok_or_else(|| VulkanSceneMeshesError::MeshNotUploaded {
                        uri: instance.source.source_uri.clone(),
                        mesh_index: instance.source.mesh_index,
                    })?;
            for primitive_index in 0..primitive_count {
                let key = (
                    instance.source.source_uri.clone(),
                    instance.source.mesh_index,
                    primitive_index,
                );
                let mesh = self.meshes.get(&key).ok_or_else(|| {
                    VulkanSceneMeshesError::PrimitiveNotUploaded {
                        uri: key.0.clone(),
                        mesh_index: key.1,
                        primitive_index,
                    }
                })?;
                mesh.record_draw(recording, layout, view_projection, instance.model)?;
                draw_count += 1;
            }
        }
        Ok(draw_count)
    }

    /// Returns the imported CPU asset pinned for a project-relative source URI.
    #[must_use]
    pub fn imported_asset(&self, uri: &AssetUri) -> Option<&GltfImport> {
        self.assets.get(uri).map(AssetHandle::get)
    }

    /// Resolves the optional imported material assigned to one uploaded primitive.
    ///
    /// The returned material borrows the asset handle pinned by this adapter.
    pub fn material_for_primitive(
        &self,
        uri: &AssetUri,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<Option<&ImportedMaterial>, VulkanSceneMeshesError> {
        let asset = self
            .assets
            .get(uri)
            .ok_or_else(|| VulkanSceneMeshesError::MissingAsset { uri: uri.clone() })?
            .get();
        let mesh = asset
            .meshes
            .iter()
            .find(|mesh| mesh.source_index == mesh_index)
            .ok_or_else(|| VulkanSceneMeshesError::MissingMesh {
                uri: uri.clone(),
                mesh_index,
            })?;
        let primitive = mesh.primitives.get(primitive_index).ok_or_else(|| {
            VulkanSceneMeshesError::MissingPrimitive {
                uri: uri.clone(),
                mesh_index,
                primitive_index,
            }
        })?;
        let Some(material_index) = primitive.material_index else {
            return Ok(None);
        };
        asset
            .materials
            .get(material_index)
            .map(Some)
            .ok_or_else(|| VulkanSceneMeshesError::MissingMaterial {
                uri: uri.clone(),
                mesh_index,
                primitive_index,
                material_index,
            })
    }

    /// Returns the number of distinct GPU primitives owned by this adapter.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.meshes.len()
    }

    /// Returns the number of imported glTF assets pinned by this adapter.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }
}

/// Failures while resolving imported mesh data and creating or recording GPU resources.
#[derive(Debug, thiserror::Error)]
pub enum VulkanSceneMeshesError {
    /// A referenced source asset is not present in the supplied typed cache.
    #[error("mesh asset {uri} is not present in the asset cache")]
    MissingAsset { uri: AssetUri },
    /// The imported asset does not contain the requested source mesh index.
    #[error("asset {uri} does not contain mesh index {mesh_index}")]
    MissingMesh { uri: AssetUri, mesh_index: usize },
    /// The imported asset contains duplicate source mesh indices.
    #[error("asset {uri} contains more than one mesh with source index {mesh_index}")]
    AmbiguousMesh { uri: AssetUri, mesh_index: usize },
    /// The requested source primitive index is outside the imported mesh.
    #[error("asset {uri} mesh {mesh_index} does not contain primitive {primitive_index}")]
    MissingPrimitive {
        uri: AssetUri,
        mesh_index: usize,
        primitive_index: usize,
    },
    /// The imported primitive refers to a missing material table entry.
    #[error(
        "asset {uri} mesh {mesh_index} primitive {primitive_index} refers to missing material {material_index}"
    )]
    MissingMaterial {
        uri: AssetUri,
        mesh_index: usize,
        primitive_index: usize,
        material_index: usize,
    },
    /// The referenced imported mesh contains no primitives.
    #[error("asset {uri} mesh {mesh_index} contains no primitives")]
    EmptyMesh { uri: AssetUri, mesh_index: usize },
    /// Upload of one imported primitive failed.
    #[error("could not upload asset {uri} mesh {mesh_index} primitive {primitive_index}: {source}")]
    PrimitiveUpload {
        uri: AssetUri,
        mesh_index: usize,
        primitive_index: usize,
        #[source]
        source: MeshUploadError,
    },
    /// Temporary host-side storage for an upload could not be reserved.
    #[error("could not reserve scene mesh upload storage: {0}")]
    HostAllocation(String),
    /// The snapshot references a mesh that this adapter did not upload.
    #[error("GPU resources for asset {uri} mesh {mesh_index} were not uploaded")]
    MeshNotUploaded { uri: AssetUri, mesh_index: usize },
    /// The adapter is missing one primitive from a previously uploaded mesh.
    #[error(
        "GPU resources for asset {uri} mesh {mesh_index} primitive {primitive_index} are missing"
    )]
    PrimitiveNotUploaded {
        uri: AssetUri,
        mesh_index: usize,
        primitive_index: usize,
    },
    /// The combined camera view-projection matrix is not finite.
    #[error("scene camera view-projection matrix is not finite")]
    NonFiniteViewProjection,
    /// A draw operation rejected its transform or command recording.
    #[error(transparent)]
    Draw(#[from] MeshDrawError),
    /// Vulkan command recording or synchronization failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}
