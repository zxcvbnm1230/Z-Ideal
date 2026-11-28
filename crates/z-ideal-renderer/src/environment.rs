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

/// Deterministically convolved diffuse irradiance and roughness-prefiltered specular cubes.
///
/// Diffuse values store the cosine-weighted hemispherical irradiance integral (not irradiance
/// divided by PI). Specular mip zero is the sharp environment and the final mip is the roughest.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentLightingMaps {
    irradiance: EnvironmentCubeMap,
    specular: Vec<EnvironmentCubeMap>,
}

impl EnvironmentLightingMaps {
    /// Convolves an equirectangular HDRI using deterministic Hammersley/GGX sampling.
    ///
    /// `sample_count` applies independently to every output texel. The operation is bounded
    /// to 100 million source samples and generated image storage is bounded to 512 MiB.
    pub fn from_equirectangular(
        source: &HdrEnvironmentMap,
        irradiance_face_size: u32,
        specular_face_size: u32,
        specular_mip_levels: u32,
        sample_count: u32,
    ) -> Result<Self, EnvironmentCubeError> {
        validate_source(source)?;
        if irradiance_face_size == 0
            || specular_face_size == 0
            || irradiance_face_size > MAX_CUBE_FACE_SIZE
            || specular_face_size > MAX_CUBE_FACE_SIZE
            || sample_count == 0
            || sample_count > 4_096
            || specular_mip_levels == 0
            || specular_mip_levels > u32::BITS - specular_face_size.leading_zeros()
        {
            return Err(EnvironmentCubeError::InvalidConvolutionOptions);
        }
        let mut specular_sizes = Vec::with_capacity(specular_mip_levels as usize);
        for level in 0..specular_mip_levels {
            specular_sizes.push((specular_face_size >> level).max(1));
        }
        let output_pixels = u64::from(irradiance_face_size)
            .checked_mul(u64::from(irradiance_face_size))
            .and_then(|value| value.checked_mul(6))
            .and_then(|value| {
                specular_sizes.iter().try_fold(value, |total, size| {
                    total.checked_add(u64::from(*size) * u64::from(*size) * 6)
                })
            })
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        let output_bytes = output_pixels
            .checked_mul(16)
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        let work = output_pixels
            .checked_mul(u64::from(sample_count))
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        if output_bytes > MAX_CUBE_STORAGE_BYTES || work > 100_000_000 {
            return Err(EnvironmentCubeError::ResourceLimitExceeded);
        }
        let irradiance = convolve_cube(irradiance_face_size, |normal| {
            let (tangent, bitangent) = tangent_basis(normal);
            let mut sum = [0.0_f32; 3];
            for index in 0..sample_count {
                let xi = hammersley(index, sample_count);
                let radius = xi[0].sqrt();
                let phi = 2.0 * PI * xi[1];
                let local = [radius * phi.cos(), radius * phi.sin(), (1.0 - xi[0]).sqrt()];
                let direction = local_to_world(local, tangent, bitangent, normal);
                let radiance = sample_equirectangular(source, direction);
                for channel in 0..3 {
                    sum[channel] += radiance[channel];
                }
            }
            for channel in &mut sum {
                *channel *= PI / sample_count as f32;
            }
            sum
        })?;
        let mut specular = Vec::with_capacity(specular_sizes.len());
        for (level, size) in specular_sizes.into_iter().enumerate() {
            let roughness = if specular_mip_levels == 1 {
                0.0
            } else {
                level as f32 / (specular_mip_levels - 1) as f32
            };
            specular.push(convolve_cube(size, |normal| {
                if roughness <= f32::EPSILON {
                    return sample_equirectangular(source, normal);
                }
                let (tangent, bitangent) = tangent_basis(normal);
                let mut sum = [0.0_f32; 3];
                let mut total_weight = 0.0_f32;
                for index in 0..sample_count {
                    let xi = hammersley(index, sample_count);
                    let phi = 2.0 * PI * xi[0];
                    let alpha = roughness * roughness;
                    let alpha_squared = alpha * alpha;
                    let cos_theta = ((1.0 - xi[1]) / (1.0 + (alpha_squared - 1.0) * xi[1])).sqrt();
                    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                    let half_local = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
                    let half_vector = local_to_world(half_local, tangent, bitangent, normal);
                    let view_dot_half = dot(normal, half_vector).max(0.0);
                    let light = normalize([
                        2.0 * view_dot_half * half_vector[0] - normal[0],
                        2.0 * view_dot_half * half_vector[1] - normal[1],
                        2.0 * view_dot_half * half_vector[2] - normal[2],
                    ]);
                    let n_dot_l = dot(normal, light).max(0.0);
                    if n_dot_l > 0.0 {
                        let radiance = sample_equirectangular(source, light);
                        for channel in 0..3 {
                            sum[channel] += radiance[channel] * n_dot_l;
                        }
                        total_weight += n_dot_l;
                    }
                }
                if total_weight > 0.0 {
                    for channel in &mut sum {
                        *channel /= total_weight;
                    }
                }
                sum
            })?);
        }
        Ok(Self {
            irradiance,
            specular,
        })
    }

    /// Returns the diffuse irradiance cube; values are not divided by PI.
    #[must_use]
    pub const fn irradiance(&self) -> &EnvironmentCubeMap {
        &self.irradiance
    }

    /// Returns the GGX-prefiltered specular cube mip chain, sharpest first.
    #[must_use]
    pub fn specular_mips(&self) -> &[EnvironmentCubeMap] {
        &self.specular
    }
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
        validate_source(source)?;
        if face_size == 0 || face_size > MAX_CUBE_FACE_SIZE {
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
}

/// Failures while converting or uploading an environment cube.
#[derive(Debug, Error)]
pub enum EnvironmentCubeError {
    /// The source is not a valid equirectangular map or face size is invalid.
    #[error("environment source or cube face size is invalid")]
    InvalidSourceOrFaceSize,
    /// Convolution dimensions, mip count, or sample count are invalid.
    #[error("environment convolution options are invalid")]
    InvalidConvolutionOptions,
    /// Cube output would exceed the fixed decoded storage budget.
    #[error("environment cube output exceeds the 512 MiB storage budget")]
    ResourceLimitExceeded,
    /// Convolution produced a non-finite or negative radiance value.
    #[error("environment convolution produced invalid radiance")]
    InvalidConvolutionResult,
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
        Self::from_cube_mips(device, std::slice::from_ref(&cube))
    }

    /// Creates a cube image from a validated mip chain in sharpest-to-roughest order.
    ///
    /// Every level must have six faces and dimensions `max(base_size >> level, 1)`.
    /// The image, staging allocation, and sampler are owned together. All mip levels are
    /// uploaded explicitly; ordinary Vulkan mip blits are not used to approximate GGX filtering.
    pub fn from_cube_mips(
        device: &'device VulkanDevice,
        mips: &[EnvironmentCubeMap],
    ) -> Result<Self, EnvironmentCubeError> {
        let Some(base) = mips.first() else {
            return Err(EnvironmentCubeError::InvalidConvolutionOptions);
        };
        let base_size = base.face_size;
        let max_levels = u32::BITS - base_size.leading_zeros();
        if base_size == 0
            || base_size > MAX_CUBE_FACE_SIZE
            || mips.len() > max_levels as usize
            || mips.iter().enumerate().any(|(level, mip)| {
                mip.face_size != (base_size >> level).max(1)
                    || mip.faces.iter().any(|face| {
                        face.len()
                            != (mip.face_size as usize).saturating_mul(mip.face_size as usize)
                    })
            })
        {
            return Err(EnvironmentCubeError::InvalidConvolutionOptions);
        }
        let total_pixels = mips
            .iter()
            .try_fold(0_u64, |total, mip| {
                total.checked_add(u64::from(mip.face_size) * u64::from(mip.face_size) * 6)
            })
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        let total_bytes = total_pixels
            .checked_mul(16)
            .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
        if total_bytes > MAX_CUBE_STORAGE_BYTES {
            return Err(EnvironmentCubeError::ResourceLimitExceeded);
        }
        let format = vk::Format::R32G32B32A32_SFLOAT;
        let required = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::TRANSFER_SRC
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
        if base_size > limits.max_image_dimension_cube {
            return Err(EnvironmentCubeError::InvalidSourceOrFaceSize);
        }
        let image = VulkanImage::new(
            device,
            ImageConfig {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: base_size,
                    height: base_size,
                    depth: 1,
                },
                format,
                mip_levels: u32::try_from(mips.len())
                    .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?,
                array_layers: 6,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            },
        )?;
        let byte_count = usize::try_from(total_bytes)
            .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_count)
            .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
        for mip in mips {
            for face in &mip.faces {
                for pixel in face {
                    for channel in pixel {
                        bytes.extend_from_slice(&channel.to_ne_bytes());
                    }
                }
            }
        }
        let mut staging = VulkanBuffer::new(
            device,
            BufferConfig {
                size: total_bytes,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory: BufferMemoryLocation::Upload,
            },
        )?;
        staging.write(0, &bytes)?;
        let sampler = VulkanSampler::new(
            device,
            SamplerConfig {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                min_lod: 0.0,
                max_lod: (mips.len() - 1) as f32,
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

    /// Returns the number of explicitly uploaded mip levels.
    #[must_use]
    pub const fn mip_levels(&self) -> u32 {
        self.image.config().mip_levels
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
            level_count: self.image.config().mip_levels,
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
                level_count: self.image.config().mip_levels,
                base_array_layer: 0,
                layer_count: 6,
            }],
        )?;
        let mut buffer_offset = 0_u64;
        let mut regions = Vec::with_capacity(self.image.config().mip_levels as usize * 6);
        for mip_level in 0..self.image.config().mip_levels {
            let face_size = (self.image.config().extent.width >> mip_level).max(1);
            let face_bytes = u64::from(face_size) * u64::from(face_size) * 16;
            for layer in 0..6_u32 {
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(buffer_offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(mip_level)
                                .base_array_layer(layer)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: face_size,
                            height: face_size,
                            depth: 1,
                        }),
                );
                buffer_offset = buffer_offset
                    .checked_add(face_bytes)
                    .ok_or(EnvironmentCubeError::ResourceLimitExceeded)?;
            }
        }
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
                level_count: self.image.config().mip_levels,
                base_array_layer: 0,
                layer_count: 6,
            }],
        )?;
        Ok(())
    }
}

fn validate_source(source: &HdrEnvironmentMap) -> Result<(), EnvironmentCubeError> {
    if source.width == 0
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
    Ok(())
}

fn convolve_cube(
    face_size: u32,
    mut filter: impl FnMut([f32; 3]) -> [f32; 3],
) -> Result<EnvironmentCubeMap, EnvironmentCubeError> {
    let count = usize::try_from(u64::from(face_size) * u64::from(face_size))
        .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
    let mut buffers = Vec::with_capacity(6);
    for _ in 0..6 {
        let mut face = Vec::new();
        face.try_reserve_exact(count)
            .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
        buffers.push(face);
    }
    let mut faces: [Vec<[f32; 4]>; 6] = buffers
        .try_into()
        .map_err(|_| EnvironmentCubeError::ResourceLimitExceeded)?;
    for face in EnvironmentCubeFace::ALL {
        for y in 0..face_size {
            for x in 0..face_size {
                let u = 2.0 * (x as f32 + 0.5) / face_size as f32 - 1.0;
                let v = 2.0 * (y as f32 + 0.5) / face_size as f32 - 1.0;
                let rgb = filter(face_direction(face, u, v));
                if rgb.iter().any(|value| !value.is_finite() || *value < 0.0) {
                    return Err(EnvironmentCubeError::InvalidConvolutionResult);
                }
                faces[face as usize].push([rgb[0], rgb[1], rgb[2], 1.0]);
            }
        }
    }
    Ok(EnvironmentCubeMap { face_size, faces })
}

fn hammersley(index: u32, count: u32) -> [f32; 2] {
    let mut bits = index;
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    [index as f32 / count as f32, bits as f32 * 2.328_306_4e-10]
}

fn tangent_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if normal[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize(cross(up, normal));
    let bitangent = normalize(cross(normal, tangent));
    (tangent, bitangent)
}

fn local_to_world(
    local: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    normal: [f32; 3],
) -> [f32; 3] {
    normalize([
        tangent[0] * local[0] + bitangent[0] * local[1] + normal[0] * local[2],
        tangent[1] * local[0] + bitangent[1] * local[1] + normal[1] * local[2],
        tangent[2] * local[0] + bitangent[2] * local[1] + normal[2] * local[2],
    ])
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = dot(value, value).sqrt();
    [value[0] / length, value[1] / length, value[2] / length]
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
    use super::{
        EnvironmentCubeError, EnvironmentCubeFace, EnvironmentCubeMap, EnvironmentLightingMaps,
    };
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

    #[test]
    fn diffuse_and_specular_convolution_preserve_constant_hdr_radiance() {
        let maps = EnvironmentLightingMaps::from_equirectangular(&constant_source(), 2, 4, 3, 64)
            .expect("convolve constant HDR environment");
        for face in EnvironmentCubeFace::ALL {
            for pixel in maps.irradiance().face(face) {
                for (actual, expected) in pixel[..3].iter().zip([
                    3.0 * std::f32::consts::PI,
                    2.0 * std::f32::consts::PI,
                    std::f32::consts::PI,
                ]) {
                    assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
                }
            }
        }
        assert_eq!(maps.specular_mips().len(), 3);
        assert_eq!(
            maps.specular_mips()
                .iter()
                .map(EnvironmentCubeMap::face_size)
                .collect::<Vec<_>>(),
            [4, 2, 1]
        );
        for mip in maps.specular_mips() {
            for face in EnvironmentCubeFace::ALL {
                for pixel in mip.face(face) {
                    assert!((pixel[0] - 3.0).abs() < 1e-4);
                    assert!((pixel[1] - 2.0).abs() < 1e-4);
                    assert!((pixel[2] - 1.0).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn rougher_specular_mip_spreads_a_directional_hdr_source() {
        let mut pixels = vec![[0.0, 0.0, 0.0]; 64 * 32];
        for y in 15..=16 {
            for x in 47..=48 {
                pixels[y * 64 + x] = [10.0, 5.0, 2.0];
            }
        }
        let source = HdrEnvironmentMap {
            source_path: "bright-lobe.hdr".into(),
            width: 64,
            height: 32,
            pixels,
        };
        let maps = EnvironmentLightingMaps::from_equirectangular(&source, 1, 3, 2, 1_024)
            .expect("convolve bright directional lobe");
        let sharp = maps.specular_mips()[0].face(EnvironmentCubeFace::PositiveZ)[4][0];
        let rough = maps.specular_mips()[1].face(EnvironmentCubeFace::PositiveZ)[0][0];
        assert!(
            sharp > 9.0,
            "sharp specular sample should retain the source lobe: {sharp}"
        );
        assert!(
            rough > 0.0 && rough < sharp,
            "rough mip should spread and reduce the lobe: {rough} vs {sharp}"
        );
    }

    #[test]
    fn convolution_is_deterministic_and_rejects_excessive_work() {
        let source = constant_source();
        let first = EnvironmentLightingMaps::from_equirectangular(&source, 2, 4, 3, 32)
            .expect("first deterministic convolution");
        let second = EnvironmentLightingMaps::from_equirectangular(&source, 2, 4, 3, 32)
            .expect("second deterministic convolution");
        assert_eq!(first, second);
        assert!(matches!(
            EnvironmentLightingMaps::from_equirectangular(&source, 2, 4, 3, 0),
            Err(EnvironmentCubeError::InvalidConvolutionOptions)
        ));
        assert!(matches!(
            EnvironmentLightingMaps::from_equirectangular(&source, 512, 512, 10, 4_096),
            Err(EnvironmentCubeError::ResourceLimitExceeded)
        ));
    }
}
