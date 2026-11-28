//! Equirectangular-to-cubemap conversion and explicit Vulkan cube-image upload.
use ash::vk;
use std::{cell::Cell, f32::consts::PI};
use thiserror::Error;
use z_ideal_assets::HdrEnvironmentMap;

use crate::{
    BufferBarrier, BufferConfig, BufferMemoryLocation, CommandBufferRecording, ImageBarrier,
    ImageConfig, ImageViewConfig, SamplerConfig, VulkanBuffer, VulkanDevice, VulkanError,
    VulkanImage, VulkanImageView, VulkanSampler,
};

const MAX_CUBE_FACE_SIZE: u32 = 4_096;
const MAX_CUBE_STORAGE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadState {
    Pending,
    Recorded,
    Failed,
}

/// The six cube-map faces in Vulkan-compatible layer order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum EnvironmentCubeFace {
    /// Positive X direction.
    PositiveX = 0,
    /// Negative X direction.
    NegativeX = 1,
    /// Positive Y direction.
    PositiveY = 2,
    /// Negative Y direction.
    NegativeY = 3,
    /// Positive Z direction.
    PositiveZ = 4,
    /// Negative Z direction.
    NegativeZ = 5,
}

impl EnvironmentCubeFace {
    /// All faces in Vulkan array-layer order.
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];
}

/// A CPU-side, scene-linear RGBA32F cube environment with six square faces.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentCubeMap {
    face_size: u32,
    faces: [Vec<[f32; 4]>; 6],
}

impl EnvironmentCubeMap {
    /// Converts a validated 2:1 equirectangular HDRI into six bilinearly sampled faces.
    ///
    /// Face orientation follows Vulkan cube-array layer order: +X, -X, +Y, -Y, +Z, -Z.
    /// Horizontal source sampling wraps at the longitude seam and vertical sampling clamps
    /// at the poles. The result remains scene-linear and is not tone-mapped.
    pub fn from_equirectangular(
        source: &HdrEnvironmentMap,
        face_size: u32,
    ) -> Result<Self, EnvironmentCubeError> {
        if face_size == 0
            || face_size > MAX_CUBE_FACE_SIZE
            || source.width == 0
            || source.height == 0
            || u64::from(source.width) != u64::from(source.height) * 2
            || usize::try_from(u64::from(source.width) * u64::from(source.height)).ok()
                != Some(source.pixels.len())
            || source
                .pixels
                .iter()
                .flatten()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(EnvironmentCubeError::InvalidSourceOrFaceSize);
        }
        let output_bytes = u64::from(face_size)
            .checked_mul(u64::from(face_size))
            .and_then(|value| value.checked_mul(6))
            .and_then(|value| value.checked_mul(16))
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        if output_bytes > MAX_CUBE_STORAGE_BYTES {
            return Err(EnvironmentCubeError::ResourceLimitExceeded);
        }
        let face_pixel_count = usize::try_from(u64::from(face_size) * u64::from(face_size))
            .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
        let mut face_buffers = Vec::with_capacity(6);
        for _ in 0..6 {
            let mut face = Vec::new();
            face.try_reserve_exact(face_pixel_count)
                .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
            face_buffers.push(face);
        }
        let faces: [Vec<[f32; 4]>; 6] = face_buffers
            .try_into()
            .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
        let mut faces = faces;
        for face in EnvironmentCubeFace::ALL {
            let output = &mut faces[face as usize];
            for y in 0..face_size {
                for x in 0..face_size {
                    let u = (2.0 * (x as f32 + 0.5) / face_size as f32) - 1.0;
                    let v = (2.0 * (y as f32 + 0.5) / face_size as f32) - 1.0;
                    let direction = face_direction(face, u, v);
                    let rgb = sample_equirectangular(source, direction);
                    output.push([rgb[0], rgb[1], rgb[2], 1.0]);
                }
            }
        }
        Ok(Self { face_size, faces })
    }

    /// Returns the width and height of each square face.
    #[must_use]
    pub const fn face_size(&self) -> u32 {
        self.face_size
    }

    /// Returns one face's row-major RGBA32F pixels.
    #[must_use]
    pub fn face(&self, face: EnvironmentCubeFace) -> &[[f32; 4]] {
        &self.faces[face as usize]
    }

    /// Returns all face pixels in Vulkan array-layer order.
    #[must_use]
    pub fn faces(&self) -> &[Vec<[f32; 4]>; 6] {
        &self.faces
    }

    fn to_staging_bytes(&self) -> Result<Vec<u8>, EnvironmentCubeError> {
        let pixel_count = self.faces.iter().map(Vec::len).sum::<usize>();
        let byte_count = pixel_count
            .checked_mul(16)
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_count)
            .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
        for face in &self.faces {
            for pixel in face {
                for channel in pixel {
                    bytes.extend_from_slice(&channel.to_ne_bytes());
                }
            }
        }
        Ok(bytes)
    }
}

/// Failures while converting or uploading an environment cube.
#[derive(Debug, Error)]
pub enum EnvironmentCubeError {
    /// The source is not a valid equirectangular map or face size is invalid.
    #[error("environment source or cube face size is invalid")]
    InvalidSourceOrFaceSize,
    /// Cube output would exceed the fixed decoded storage budget.
    #[error("environment cube output exceeds the 512 MiB storage budget")]
    ResourceLimitExceeded,
    /// The selected device cannot sample and linearly filter RGBA32F images.
    #[error(
        "Vulkan format {format:?} lacks environment cube features {required:?}; available {available:?}"
    )]
    UnsupportedFormatFeatures {
        format: vk::Format,
        required: vk::FormatFeatureFlags,
        available: vk::FormatFeatureFlags,
    },
    /// Upload was already recorded successfully.
    #[error("environment cube upload has already been recorded")]
    AlreadyRecorded,
    /// A prior partial upload recording failed; discard this object and command buffer.
    #[error("environment cube upload recording previously failed; discard this object")]
    RecordingFailed,
    /// Vulkan resource creation or command recording failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

/// Owns a Vulkan cube-compatible image, staging buffer, and linear sampler.
pub struct VulkanEnvironmentCube<'device> {
    image: VulkanImage<'device>,
    staging: VulkanBuffer<'device>,
    sampler: VulkanSampler<'device>,
    upload_state: Cell<UploadState>,
}

impl<'device> VulkanEnvironmentCube<'device> {
    /// Converts and creates a single-mip RGBA32F Vulkan cube from an equirectangular HDRI.
    pub fn from_equirectangular(
        device: &'device VulkanDevice,
        source: &HdrEnvironmentMap,
        face_size: u32,
    ) -> Result<Self, EnvironmentCubeError> {
        let cube = EnvironmentCubeMap::from_equirectangular(source, face_size)?;
        let format = vk::Format::R32G32B32A32_SFLOAT;
        let required = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        // SAFETY: Physical-device properties belong to the instance retained by this device.
        let properties = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_format_properties(device.physical_device, format)
        };
        let available = properties.optimal_tiling_features;
        if !available.contains(required) {
            return Err(EnvironmentCubeError::UnsupportedFormatFeatures {
                format,
                required,
                available,
            });
        }
        // SAFETY: Physical-device limits belong to the instance retained by this device.
        let limits = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_properties(device.physical_device)
                .limits
        };
        if face_size > limits.max_image_dimension_cube {
            return Err(EnvironmentCubeError::InvalidSourceOrFaceSize);
        }
        let image = VulkanImage::new(
            device,
            ImageConfig {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: face_size,
                    height: face_size,
                    depth: 1,
                },
                format,
                mip_levels: 1,
                array_layers: 6,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            },
        )?;
        let bytes = cube.to_staging_bytes()?;
        let staging = VulkanBuffer::new(
            device,
            BufferConfig {
                size: u64::try_from(bytes.len())
                    .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory: BufferMemoryLocation::Upload,
            },
        )?;
        let mut staging = staging;
        staging.write(0, &bytes)?;
        let sampler = VulkanSampler::new(
            device,
            SamplerConfig {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                min_lod: 0.0,
                max_lod: 0.0,
                ..SamplerConfig::default()
            },
        )?;
        Ok(Self {
            image,
            staging,
            sampler,
            upload_state: Cell::new(UploadState::Pending),
        })
    }

    /// Returns the cube-compatible Vulkan image.
    #[must_use]
    pub const fn image(&self) -> &VulkanImage<'device> {
        &self.image
    }

    /// Returns the cube sampler.
    #[must_use]
    pub const fn sampler(&self) -> &VulkanSampler<'device> {
        &self.sampler
    }

    /// Creates a cube image view spanning all six faces.
    pub fn create_view(&self) -> Result<VulkanImageView<'_, 'device>, VulkanError> {
        self.image.create_view(ImageViewConfig {
            view_type: vk::ImageViewType::CUBE,
            level_count: 1,
            layer_count: 6,
            ..ImageViewConfig::default()
        })
    }

    /// Records staging upload and transitions all six layers to shader-read layout.
    ///
    /// Must be called outside a render pass on a queue supporting HOST, TRANSFER, and
    /// FRAGMENT_SHADER stages. Discard the command buffer and cube if recording fails.
    pub fn record_upload<'recording>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, '_, 'device>,
    ) -> Result<(), EnvironmentCubeError> {
        match self.upload_state.get() {
            UploadState::Pending => {}
            UploadState::Recorded => return Err(EnvironmentCubeError::AlreadyRecorded),
            UploadState::Failed => return Err(EnvironmentCubeError::RecordingFailed),
        }
        let result = self.record_upload_commands(recording);
        self.upload_state.set(if result.is_ok() {
            UploadState::Recorded
        } else {
            UploadState::Failed
        });
        result
    }

    fn record_upload_commands<'recording>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, '_, 'device>,
    ) -> Result<(), EnvironmentCubeError> {
        recording.pipeline_barrier(
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[BufferBarrier {
                buffer: &self.staging,
                source_access: vk::AccessFlags::HOST_WRITE,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            }],
            &[ImageBarrier {
                image: &self.image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                source_access: vk::AccessFlags::empty(),
                destination_access: vk::AccessFlags::TRANSFER_WRITE,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            }],
        )?;
        let face_bytes = u64::from(self.image.config().extent.width)
            * u64::from(self.image.config().extent.height)
            * 16;
        let regions: Vec<_> = (0..6_u32)
            .map(|layer| {
                vk::BufferImageCopy::default()
                    .buffer_offset(face_bytes * u64::from(layer))
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(layer)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(self.image.config().extent)
            })
            .collect();
        recording.copy_buffer_to_image(
            &self.staging,
            &self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        )?;
        recording.pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: &self.image,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                source_access: vk::AccessFlags::TRANSFER_WRITE,
                destination_access: vk::AccessFlags::SHADER_READ,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            }],
        )?;
        Ok(())
    }
}

fn face_direction(face: EnvironmentCubeFace, u: f32, v: f32) -> [f32; 3] {
    let direction = match face {
        EnvironmentCubeFace::PositiveX => [1.0, -v, -u],
        EnvironmentCubeFace::NegativeX => [-1.0, -v, u],
        EnvironmentCubeFace::PositiveY => [u, 1.0, v],
        EnvironmentCubeFace::NegativeY => [u, -1.0, -v],
        EnvironmentCubeFace::PositiveZ => [u, -v, 1.0],
        EnvironmentCubeFace::NegativeZ => [-u, -v, -1.0],
    };
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ]
}

fn sample_equirectangular(source: &HdrEnvironmentMap, direction: [f32; 3]) -> [f32; 3] {
    let longitude = direction[2].atan2(direction[0]);
    let latitude = direction[1].clamp(-1.0, 1.0).asin();
    let u = longitude / (2.0 * PI) + 0.5;
    let v = 0.5 - latitude / PI;
    let x = u * source.width as f32 - 0.5;
    let y = v * source.height as f32 - 0.5;
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let tx = x - x.floor();
    let ty = y - y.floor();
    let sample = |px: i64, py: i64| -> [f32; 3] {
        let wrapped_x = usize::try_from(px.rem_euclid(i64::from(source.width)))
            .expect("wrapped x coordinate is non-negative");
        let clamped_y = usize::try_from(py.clamp(0, i64::from(source.height) - 1))
            .expect("clamped y coordinate is non-negative");
        let width = usize::try_from(source.width).expect("source width fits usize");
        source.pixels[clamped_y * width + wrapped_x]
    };
    let a = sample(x0, y0);
    let b = sample(x0 + 1, y0);
    let c = sample(x0, y0 + 1);
    let d = sample(x0 + 1, y0 + 1);
    std::array::from_fn(|channel| {
        let top = a[channel] * (1.0 - tx) + b[channel] * tx;
        let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
        top * (1.0 - ty) + bottom * ty
    })
}

#[cfg(test)]
mod tests {
    use super::{EnvironmentCubeError, EnvironmentCubeFace, EnvironmentCubeMap};
    use z_ideal_assets::HdrEnvironmentMap;

    fn constant_source() -> HdrEnvironmentMap {
        HdrEnvironmentMap {
            source_path: "constant.hdr".into(),
            width: 4,
            height: 2,
            pixels: vec![[3.0, 2.0, 1.0]; 8],
        }
    }

    #[test]
    fn conversion_preserves_constant_hdr_radiance_on_all_six_faces() {
        let cube = EnvironmentCubeMap::from_equirectangular(&constant_source(), 4)
            .expect("convert valid equirectangular map");
        assert_eq!(cube.face_size(), 4);
        for face in EnvironmentCubeFace::ALL {
            assert_eq!(cube.face(face), &[[3.0, 2.0, 1.0, 1.0]; 16]);
        }
    }

    #[test]
    fn conversion_uses_vulkan_face_order_and_samples_directionally() {
        let source = HdrEnvironmentMap {
            source_path: "longitude.hdr".into(),
            width: 64,
            height: 32,
            pixels: (0..32)
                .flat_map(|_| {
                    (0..64).map(|x| {
                        let u = (x as f32 + 0.5) / 64.0;
                        [u, 0.0, 0.0]
                    })
                })
                .collect(),
        };
        let cube = EnvironmentCubeMap::from_equirectangular(&source, 1)
            .expect("convert directional source");
        let positive_z = cube.face(EnvironmentCubeFace::PositiveZ)[0][0];
        let negative_z = cube.face(EnvironmentCubeFace::NegativeZ)[0][0];
        assert!(
            (positive_z - 0.75).abs() < 0.02,
            "positive Z longitude: {positive_z}"
        );
        assert!(
            (negative_z - 0.25).abs() < 0.02,
            "negative Z longitude: {negative_z}"
        );
    }

    #[test]
    fn conversion_rejects_invalid_source_and_bounded_output() {
        assert!(matches!(
            EnvironmentCubeMap::from_equirectangular(&constant_source(), 0),
            Err(EnvironmentCubeError::InvalidSourceOrFaceSize)
        ));
        assert!(matches!(
            EnvironmentCubeMap::from_equirectangular(&constant_source(), 4_097),
            Err(EnvironmentCubeError::InvalidSourceOrFaceSize)
        ));
    }
}
