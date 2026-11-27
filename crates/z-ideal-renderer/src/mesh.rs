//! Validated indexed triangle-mesh upload into Vulkan vertex and index buffers.
use ash::vk;
use z_ideal_assets::{ImportedPrimitive, PrimitiveTopology};

use crate::{BufferConfig, BufferMemoryLocation, VulkanBuffer, VulkanDevice, VulkanError};

/// Interleaved vertex layout used by the first unlit mesh pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// First texture coordinate set; defaults to zero when absent.
    pub tex_coords_0: [f32; 2],
}

/// Owns GPU vertex/index buffers for one imported triangle primitive.
pub struct MeshGpu<'device> {
    vertex_buffer: VulkanBuffer<'device>,
    index_buffer: VulkanBuffer<'device>,
    vertex_count: u32,
    index_count: u32,
}

impl<'device> MeshGpu<'device> {
    /// Validates and uploads one glTF triangle primitive using host-visible upload memory.
    ///
    /// Upload buffers are directly host-visible. This avoids hidden staging submissions;
    /// callers must establish any required host-write-to-vertex/index-read dependency
    /// before drawing, as with every other upload buffer in the renderer API.
    pub fn upload(
        device: &'device VulkanDevice,
        primitive: &ImportedPrimitive,
    ) -> Result<Self, MeshUploadError> {
        if primitive.mode != PrimitiveTopology::Triangles {
            return Err(MeshUploadError::UnsupportedTopology(primitive.mode));
        }
        let vertices = build_vertices(primitive)?;
        if primitive.indices.is_empty() || primitive.indices.len() % 3 != 0 {
            return Err(MeshUploadError::InvalidPrimitive(
                "triangle indices must be non-empty and divisible by three".to_owned(),
            ));
        }
        if primitive
            .indices
            .iter()
            .any(|index| *index as usize >= vertices.len())
        {
            return Err(MeshUploadError::InvalidPrimitive(
                "an index is outside the vertex array".to_owned(),
            ));
        }
        let vertex_count =
            u32::try_from(vertices.len()).map_err(|_| MeshUploadError::TooManyVertices)?;
        let index_count =
            u32::try_from(primitive.indices.len()).map_err(|_| MeshUploadError::TooManyIndices)?;
        let vertex_bytes = encode_vertices(&vertices)?;
        let index_bytes = encode_indices(&primitive.indices)?;
        let vertex_size =
            u64::try_from(vertex_bytes.len()).map_err(|_| MeshUploadError::BufferSizeOverflow)?;
        let index_size =
            u64::try_from(index_bytes.len()).map_err(|_| MeshUploadError::BufferSizeOverflow)?;

        let mut vertex_buffer = VulkanBuffer::new(
            device,
            BufferConfig {
                size: vertex_size,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                memory: BufferMemoryLocation::Upload,
            },
        )?;
        vertex_buffer.write(0, &vertex_bytes)?;
        let mut index_buffer = VulkanBuffer::new(
            device,
            BufferConfig {
                size: index_size,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                memory: BufferMemoryLocation::Upload,
            },
        )?;
        index_buffer.write(0, &index_bytes)?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            vertex_count,
            index_count,
        })
    }

    /// Returns the vertex buffer for command recording.
    #[must_use]
    pub const fn vertex_buffer(&self) -> &VulkanBuffer<'device> {
        &self.vertex_buffer
    }
    /// Returns the index buffer for command recording.
    #[must_use]
    pub const fn index_buffer(&self) -> &VulkanBuffer<'device> {
        &self.index_buffer
    }
    /// Returns the number of uploaded vertices.
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    /// Returns the number of uploaded indices.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }
    /// Returns the Vulkan index type used by this mesh.
    #[must_use]
    pub const fn index_type(&self) -> vk::IndexType {
        vk::IndexType::UINT32
    }
    /// Returns the binding description for the interleaved vertex layout.
    pub const fn vertex_binding() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<MeshVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }
    /// Returns position and UV attribute descriptions for the interleaved vertex layout.
    pub const fn vertex_attributes() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 12,
            },
        ]
    }
}

fn build_vertices(primitive: &ImportedPrimitive) -> Result<Vec<MeshVertex>, MeshUploadError> {
    if primitive.positions.is_empty() {
        return Err(MeshUploadError::InvalidPrimitive(
            "positions must not be empty".to_owned(),
        ));
    }
    if primitive
        .positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(MeshUploadError::InvalidPrimitive(
            "positions must be finite".to_owned(),
        ));
    }
    if primitive.tex_coords_0.as_ref().is_some_and(|uvs| {
        uvs.len() != primitive.positions.len()
            || uvs.iter().flatten().any(|value| !value.is_finite())
    }) {
        return Err(MeshUploadError::InvalidPrimitive(
            "texture coordinates must be finite and match the position count".to_owned(),
        ));
    }
    Ok(primitive
        .positions
        .iter()
        .enumerate()
        .map(|(index, position)| MeshVertex {
            position: *position,
            tex_coords_0: primitive
                .tex_coords_0
                .as_ref()
                .map_or([0.0, 0.0], |uvs| uvs[index]),
        })
        .collect())
}

fn encode_vertices(vertices: &[MeshVertex]) -> Result<Vec<u8>, MeshUploadError> {
    let capacity = vertices
        .len()
        .checked_mul(std::mem::size_of::<MeshVertex>())
        .ok_or(MeshUploadError::BufferSizeOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| MeshUploadError::HostAllocation(error.to_string()))?;
    for vertex in vertices {
        for value in vertex.position.into_iter().chain(vertex.tex_coords_0) {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(bytes)
}

fn encode_indices(indices: &[u32]) -> Result<Vec<u8>, MeshUploadError> {
    let capacity = indices
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(MeshUploadError::BufferSizeOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| MeshUploadError::HostAllocation(error.to_string()))?;
    for index in indices {
        bytes.extend_from_slice(&index.to_ne_bytes());
    }
    Ok(bytes)
}

/// Mesh upload validation and allocation failures.
#[derive(Debug, thiserror::Error)]
pub enum MeshUploadError {
    /// Vulkan buffer creation or host upload failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    /// The initial mesh renderer supports indexed triangle lists only.
    #[error("unsupported mesh primitive topology: {0:?}")]
    UnsupportedTopology(PrimitiveTopology),
    /// CPU mesh data is malformed.
    #[error("invalid mesh primitive: {0}")]
    InvalidPrimitive(String),
    /// Vertex count exceeds Vulkan's draw count range.
    #[error("mesh contains more vertices than the renderer can address")]
    TooManyVertices,
    /// Index count exceeds Vulkan's draw count range.
    #[error("mesh contains more indices than the renderer can draw")]
    TooManyIndices,
    /// Encoded vertex or index data exceeds the Vulkan buffer size range.
    #[error("mesh buffer byte size overflowed")]
    BufferSizeOverflow,
    /// Host allocation for encoded mesh data failed.
    #[error("could not allocate CPU mesh upload storage: {0}")]
    HostAllocation(String),
}

#[cfg(test)]
mod tests {
    use super::{
        MeshUploadError, MeshVertex, PrimitiveTopology, build_vertices, encode_indices,
        encode_vertices,
    };
    use z_ideal_assets::ImportedPrimitive;

    fn triangle() -> ImportedPrimitive {
        ImportedPrimitive {
            mode: PrimitiveTopology::Triangles,
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: None,
            tex_coords_0: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            indices: vec![0, 1, 2],
            material_index: None,
        }
    }

    #[test]
    fn interleaved_vertex_encoding_matches_declared_layout() {
        let vertices = build_vertices(&triangle()).expect("valid triangle");
        assert_eq!(std::mem::size_of::<MeshVertex>(), 20);
        assert_eq!(vertices[1].tex_coords_0, [1.0, 0.0]);
        assert_eq!(
            encode_vertices(&vertices).expect("encode vertices").len(),
            60
        );
        assert_eq!(
            encode_indices(&[0, 1, 2]).expect("encode indices").len(),
            12
        );
    }

    #[test]
    fn missing_uvs_receive_zero_coordinates() {
        let mut primitive = triangle();
        primitive.tex_coords_0 = None;
        let vertices = build_vertices(&primitive).expect("valid triangle");
        assert!(
            vertices
                .iter()
                .all(|vertex| vertex.tex_coords_0 == [0.0, 0.0])
        );
    }

    #[test]
    fn rejects_non_finite_positions_and_mismatched_uvs() {
        let mut primitive = triangle();
        primitive.positions[0][0] = f32::NAN;
        assert!(matches!(
            build_vertices(&primitive),
            Err(MeshUploadError::InvalidPrimitive(_))
        ));
        let mut primitive = triangle();
        primitive.tex_coords_0 = Some(vec![[0.0, 0.0]]);
        assert!(matches!(
            build_vertices(&primitive),
            Err(MeshUploadError::InvalidPrimitive(_))
        ));
    }
}
