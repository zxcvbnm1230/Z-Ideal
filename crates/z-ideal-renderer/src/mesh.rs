//! Validated indexed triangle-mesh upload into Vulkan vertex and index buffers.
use ash::vk;
use z_ideal_assets::{ImportedPrimitive, PrimitiveTopology};
use z_ideal_math::Mat4;

use crate::{
    BufferConfig, BufferMemoryLocation, CommandBufferRecording, VulkanBuffer, VulkanDevice,
    VulkanError, VulkanPipelineLayout,
};

/// Interleaved vertex layout shared by the unlit and glTF PBR mesh pipelines.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// First texture coordinate set; defaults to zero when absent.
    pub tex_coords_0: [f32; 2],
    /// Object-space unit normal; generated from triangle faces when absent in the asset.
    pub normal: [f32; 3],
    /// Object-space tangent and handedness; generated from positions and UV0 when absent.
    pub tangent: [f32; 4],
    /// Second texture coordinate set; defaults to zero when absent.
    pub tex_coords_1: [f32; 2],
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
        if primitive.indices.is_empty() || !primitive.indices.len().is_multiple_of(3) {
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

    /// Records an indexed draw using camera and object transforms.
    ///
    /// The graphics pipeline must already be bound and use a 64-byte vertex-stage
    /// push-constant range at offset zero containing one column-major `mat4`. The
    /// supplied view-projection and model matrices are multiplied on the CPU and
    /// uploaded as that matrix. The caller remains responsible for the render pass,
    /// viewport/scissor, and host-write-to-vertex-input synchronization.
    pub fn record_draw<'recording, 'pool>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
        layout: &'recording VulkanPipelineLayout<'_, 'device>,
        view_projection: Mat4,
        model: Mat4,
    ) -> Result<(), MeshDrawError> {
        let mvp = view_projection * model;
        if !view_projection.is_finite() || !model.is_finite() || !mvp.is_finite() {
            return Err(MeshDrawError::NonFiniteTransform);
        }
        let bytes = encode_matrix(mvp);
        recording.push_constants(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &bytes,
        )?;
        recording.bind_vertex_buffers(0, &[&self.vertex_buffer], &[0])?;
        recording.bind_index_buffer(&self.index_buffer, 0, self.index_type())?;
        recording.draw_indexed(self.index_count, 1, 0, 0, 0)?;
        Ok(())
    }
    /// Records an indexed draw for the glTF PBR shader using separate MVP and model matrices.
    ///
    /// The pipeline must expose a 128-byte vertex-stage push-constant range at offset zero:
    /// column-major MVP followed by column-major model matrix. This allows the vertex shader
    /// to transform positions and normals independently for world-space PBR lighting.
    pub fn record_draw_pbr<'recording, 'pool>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
        layout: &'recording VulkanPipelineLayout<'_, 'device>,
        view_projection: Mat4,
        model: Mat4,
    ) -> Result<(), MeshDrawError> {
        let mvp = view_projection * model;
        if !view_projection.is_finite() || !model.is_finite() || !mvp.is_finite() {
            return Err(MeshDrawError::NonFiniteTransform);
        }
        let mut bytes = [0_u8; 128];
        bytes[..64].copy_from_slice(&encode_matrix(mvp));
        bytes[64..].copy_from_slice(&encode_matrix(model));
        recording.push_constants(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &bytes,
        )?;
        recording.bind_vertex_buffers(0, &[&self.vertex_buffer], &[0])?;
        recording.bind_index_buffer(&self.index_buffer, 0, self.index_type())?;
        recording.draw_indexed(self.index_count, 1, 0, 0, 0)?;
        Ok(())
    }

    /// Returns the binding description for the interleaved vertex layout.
    pub const fn vertex_binding() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<MeshVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }
    /// Returns position, UV0, normal, tangent, and UV1 attribute descriptions in shader-location order.
    pub const fn vertex_attributes() -> [vk::VertexInputAttributeDescription; 5] {
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
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 20,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 48,
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
    if primitive.indices.is_empty()
        || !primitive.indices.len().is_multiple_of(3)
        || primitive
            .indices
            .iter()
            .any(|index| *index as usize >= primitive.positions.len())
    {
        return Err(MeshUploadError::InvalidPrimitive(
            "triangle indices must be non-empty, divisible by three, and within the position array"
                .to_owned(),
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
    for (name, values) in [
        (
            "normals",
            primitive.normals.as_deref().map(|values| values.len()),
        ),
        (
            "tangents",
            primitive.tangents.as_deref().map(|values| values.len()),
        ),
        (
            "texture coordinates 0",
            primitive.tex_coords_0.as_deref().map(|values| values.len()),
        ),
        (
            "texture coordinates 1",
            primitive.tex_coords_1.as_deref().map(|values| values.len()),
        ),
    ] {
        if values.is_some_and(|count| count != primitive.positions.len()) {
            return Err(MeshUploadError::InvalidPrimitive(format!(
                "{name} must match the position count"
            )));
        }
    }
    if primitive
        .normals
        .as_ref()
        .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
        || primitive
            .tangents
            .as_ref()
            .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
        || primitive
            .tex_coords_0
            .as_ref()
            .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
        || primitive
            .tex_coords_1
            .as_ref()
            .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
    {
        return Err(MeshUploadError::InvalidPrimitive(
            "normals, tangents, and texture coordinates must be finite".to_owned(),
        ));
    }

    let normals = match &primitive.normals {
        Some(normals) => normals
            .iter()
            .map(|normal| normalize3(*normal))
            .collect::<Result<Vec<_>, _>>()?,
        None => generate_normals(&primitive.positions, &primitive.indices)?,
    };
    let tangents = match &primitive.tangents {
        Some(tangents) => tangents
            .iter()
            .zip(&normals)
            .map(|(tangent, normal)| normalize_imported_tangent(*tangent, *normal))
            .collect::<Result<Vec<_>, _>>()?,
        None => generate_tangents(
            &primitive.positions,
            &normals,
            primitive.tex_coords_0.as_deref(),
            &primitive.indices,
        )?,
    };

    Ok(primitive
        .positions
        .iter()
        .enumerate()
        .map(|(index, position)| MeshVertex {
            position: *position,
            tex_coords_0: primitive
                .tex_coords_0
                .as_ref()
                .map_or([0.0; 2], |values| values[index]),
            normal: normals[index],
            tangent: tangents[index],
            tex_coords_1: primitive
                .tex_coords_1
                .as_ref()
                .map_or([0.0; 2], |values| values[index]),
        })
        .collect())
}

fn generate_normals(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<Vec<[f32; 3]>, MeshUploadError> {
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(MeshUploadError::InvalidPrimitive(
            "normal generation requires triangle indices".to_owned(),
        ));
    }
    let mut accumulated = vec![[0.0_f32; 3]; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let a = positions[triangle[0] as usize];
        let b = positions[triangle[1] as usize];
        let c = positions[triangle[2] as usize];
        let normal = cross3(sub3(b, a), sub3(c, a));
        if length_squared3(normal) <= f32::EPSILON {
            continue;
        }
        for index in triangle {
            let value = &mut accumulated[*index as usize];
            *value = add3(*value, normal);
        }
    }
    accumulated.into_iter().map(normalize3).collect()
}

fn generate_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    tex_coords: Option<&[[f32; 2]]>,
    indices: &[u32],
) -> Result<Vec<[f32; 4]>, MeshUploadError> {
    let Some(tex_coords) = tex_coords else {
        return Ok(normals
            .iter()
            .map(|normal| {
                let tangent = fallback_tangent(*normal);
                [tangent[0], tangent[1], tangent[2], 1.0]
            })
            .collect());
    };
    let mut accumulated_tangents = vec![[0.0_f32; 3]; positions.len()];
    let mut accumulated_bitangents = vec![[0.0_f32; 3]; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let i0 = triangle[0] as usize;
        let i1 = triangle[1] as usize;
        let i2 = triangle[2] as usize;
        let edge1 = sub3(positions[i1], positions[i0]);
        let edge2 = sub3(positions[i2], positions[i0]);
        let duv1 = [
            tex_coords[i1][0] - tex_coords[i0][0],
            tex_coords[i1][1] - tex_coords[i0][1],
        ];
        let duv2 = [
            tex_coords[i2][0] - tex_coords[i0][0],
            tex_coords[i2][1] - tex_coords[i0][1],
        ];
        let determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if determinant.abs() <= f32::EPSILON {
            continue;
        }
        let reciprocal = determinant.recip();
        let tangent = [
            (edge1[0] * duv2[1] - edge2[0] * duv1[1]) * reciprocal,
            (edge1[1] * duv2[1] - edge2[1] * duv1[1]) * reciprocal,
            (edge1[2] * duv2[1] - edge2[2] * duv1[1]) * reciprocal,
        ];
        let bitangent = [
            (edge2[0] * duv1[0] - edge1[0] * duv2[0]) * reciprocal,
            (edge2[1] * duv1[0] - edge1[1] * duv2[0]) * reciprocal,
            (edge2[2] * duv1[0] - edge1[2] * duv2[0]) * reciprocal,
        ];
        for index in triangle {
            let vertex = *index as usize;
            accumulated_tangents[vertex] = add3(accumulated_tangents[vertex], tangent);
            accumulated_bitangents[vertex] = add3(accumulated_bitangents[vertex], bitangent);
        }
    }
    Ok(accumulated_tangents
        .into_iter()
        .zip(accumulated_bitangents)
        .zip(normals)
        .map(|((tangent, bitangent), normal)| {
            let orthogonal = sub3(tangent, scale3(*normal, dot3(*normal, tangent)));
            let normalized = normalize3(orthogonal).unwrap_or_else(|_| fallback_tangent(*normal));
            let handedness = if dot3(cross3(*normal, normalized), bitangent) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [normalized[0], normalized[1], normalized[2], handedness]
        })
        .collect())
}

fn normalize_imported_tangent(
    tangent: [f32; 4],
    normal: [f32; 3],
) -> Result<[f32; 4], MeshUploadError> {
    if (tangent[3].abs() - 1.0).abs() > 1e-4 {
        return Err(MeshUploadError::InvalidPrimitive(
            "tangent handedness must be -1 or +1".to_owned(),
        ));
    }
    let orthogonal = sub3(
        [tangent[0], tangent[1], tangent[2]],
        scale3(normal, dot3(normal, [tangent[0], tangent[1], tangent[2]])),
    );
    let normalized = normalize3(orthogonal)?;
    Ok([normalized[0], normalized[1], normalized[2], tangent[3]])
}

fn normalize3(value: [f32; 3]) -> Result<[f32; 3], MeshUploadError> {
    let length_squared = length_squared3(value);
    if !length_squared.is_finite() || length_squared <= f32::EPSILON {
        return Err(MeshUploadError::InvalidPrimitive(
            "normal or tangent basis contains a zero-length vector".to_owned(),
        ));
    }
    let inverse = length_squared.sqrt().recip();
    Ok(scale3(value, inverse))
}

fn fallback_tangent(normal: [f32; 3]) -> [f32; 3] {
    let axis = if normal[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    normalize3(cross3(axis, normal)).unwrap_or([1.0, 0.0, 0.0])
}
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale3(a: [f32; 3], scale: f32) -> [f32; 3] {
    [a[0] * scale, a[1] * scale, a[2] * scale]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length_squared3(value: [f32; 3]) -> f32 {
    dot3(value, value)
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
        for value in vertex
            .position
            .into_iter()
            .chain(vertex.tex_coords_0)
            .chain(vertex.normal)
            .chain(vertex.tangent)
            .chain(vertex.tex_coords_1)
        {
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

/// Failures while recording a camera-driven mesh draw.
#[derive(Debug, thiserror::Error)]
pub enum MeshDrawError {
    /// A camera, model, or combined transform matrix contains non-finite values.
    #[error("mesh draw transform matrices must be finite")]
    NonFiniteTransform,
    /// Command recording rejected the draw or push-constant write.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

fn encode_matrix(matrix: Mat4) -> [u8; 64] {
    let mut bytes = [0_u8; 64];
    for (index, value) in matrix.to_cols_array().into_iter().enumerate() {
        let offset = index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
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
        encode_matrix, encode_vertices,
    };
    use z_ideal_assets::ImportedPrimitive;
    use z_ideal_math::Mat4;

    fn triangle() -> ImportedPrimitive {
        ImportedPrimitive {
            mode: PrimitiveTopology::Triangles,
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: None,
            tangents: None,
            tex_coords_0: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            tex_coords_1: None,
            indices: vec![0, 1, 2],
            material_index: None,
        }
    }

    #[test]
    fn interleaved_vertex_encoding_matches_declared_layout() {
        let vertices = build_vertices(&triangle()).expect("valid triangle");
        assert_eq!(std::mem::size_of::<MeshVertex>(), 56);
        assert_eq!(vertices[1].tex_coords_0, [1.0, 0.0]);
        assert_eq!(vertices[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(vertices[0].tangent, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            encode_vertices(&vertices).expect("encode vertices").len(),
            168
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
    fn carries_uv1_and_generates_a_finite_tangent_basis() {
        let mut primitive = triangle();
        primitive.tex_coords_1 = Some(vec![[0.25, 0.5], [0.75, 0.5], [0.5, 1.0]]);
        let vertices = build_vertices(&primitive).expect("valid two-UV triangle");
        assert_eq!(vertices[0].tex_coords_1, [0.25, 0.5]);
        assert_eq!(vertices[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(vertices[0].tangent, [1.0, 0.0, 0.0, 1.0]);
        assert!(
            vertices
                .iter()
                .flat_map(|vertex| vertex.tangent)
                .all(f32::is_finite)
        );
    }

    #[test]
    fn rejects_out_of_range_indices_before_generating_vertex_basis() {
        let mut primitive = triangle();
        primitive.indices[2] = 99;
        assert!(matches!(
            build_vertices(&primitive),
            Err(MeshUploadError::InvalidPrimitive(_))
        ));
    }

    #[test]
    fn rejects_invalid_imported_tangent_handedness_and_zero_vectors() {
        let mut primitive = triangle();
        primitive.tangents = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
        assert!(matches!(
            build_vertices(&primitive),
            Err(MeshUploadError::InvalidPrimitive(_))
        ));
        primitive.tangents = Some(vec![[0.0, 0.0, 0.0, 1.0]; 3]);
        assert!(matches!(
            build_vertices(&primitive),
            Err(MeshUploadError::InvalidPrimitive(_))
        ));
    }

    #[test]
    fn generated_tangent_preserves_mirrored_uv_handedness() {
        let mut primitive = triangle();
        primitive.tex_coords_0 = Some(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
        let vertices = build_vertices(&primitive).expect("valid mirrored-UV triangle");
        assert!(vertices.iter().all(|vertex| vertex.tangent[3] == -1.0));
    }

    #[test]
    fn matrix_push_constant_encoding_is_column_major_and_exactly_64_bytes() {
        let matrix = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let bytes = encode_matrix(matrix);
        assert_eq!(bytes.len(), 64);
        for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(f32::from_ne_bytes(*chunk), index as f32 + 1.0);
        }
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
