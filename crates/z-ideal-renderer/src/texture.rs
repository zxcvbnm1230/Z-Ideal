//! Explicit Vulkan upload of imported glTF 2D images and sampler state.
use ash::vk;
use std::cell::Cell;
use thiserror::Error;
use z_ideal_assets::{
    GltfImport, HdrEnvironmentMap, ImagePixelFormat, ImportedImage, ImportedTexture,
    TextureMagFilter, TextureMinFilter, TextureWrapMode,
};

use crate::{
    BufferBarrier, BufferConfig, BufferMemoryLocation, CommandBufferRecording, ImageBarrier,
    ImageConfig, ImageViewConfig, SamplerConfig, VulkanBuffer, VulkanDevice, VulkanError,
    VulkanImage, VulkanImageView, VulkanSampler,
};

/// Failures while converting an imported glTF texture into Vulkan resources.
#[derive(Debug, Error)]
pub enum TextureUploadError {
    /// The requested texture index is outside the imported texture table.
    #[error("imported glTF texture index {0} does not exist")]
    MissingTexture(usize),
    /// The texture references an image outside the imported image table.
    #[error("imported glTF texture references missing image index {0}")]
    MissingImage(usize),
    /// The imported image dimensions or pixel storage are invalid.
    #[error("imported image dimensions or pixel storage are invalid")]
    InvalidImageData,
    /// The requested color space has no compatible Vulkan format for this image representation.
    #[error("image format {format:?} cannot represent requested {color_space:?} color space")]
    UnsupportedColorSpaceFormat {
        format: ImagePixelFormat,
        color_space: TextureColorSpace,
    },
    /// The physical device cannot generate mipmaps for this optimal-tiling image format.
    #[error(
        "Vulkan format {format:?} does not support required mip-generation features {required:?}; available {available:?}"
    )]
    UnsupportedMipFeatures {
        format: vk::Format,
        required: vk::FormatFeatureFlags,
        available: vk::FormatFeatureFlags,
    },
    /// The physical device does not support the required optimal-tiling image operations.
    #[error(
        "Vulkan format {format:?} lacks required features {required:?}; available {available:?}"
    )]
    UnsupportedFormatFeatures {
        format: vk::Format,
        required: vk::FormatFeatureFlags,
        available: vk::FormatFeatureFlags,
    },
    /// Upload recording has already succeeded for this texture.
    #[error("texture upload commands have already been recorded")]
    AlreadyRecorded,
    /// An earlier attempt partially recorded commands and this upload object cannot be reused.
    #[error("a previous texture upload recording failed; discard this upload object")]
    RecordingFailed,
    /// A Vulkan resource creation or command-recording operation failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

/// Interpretation of normalized 8-bit image channels during sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureColorSpace {
    /// Numeric channels are sampled without transfer-function conversion.
    Linear,
    /// Supported 8-bit channels are sampled using the sRGB transfer function.
    Srgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadState {
    Pending,
    Recorded,
    Failed,
}

/// Owns an imported glTF image's Vulkan image, staging buffer, and sampler.
///
/// Create one instance per imported glTF texture and semantic color-space use so each
/// texture retains its own sampler state and linear/sRGB interpretation.
/// The image starts in UNDEFINED layout. Call record_upload exactly once, outside
/// a render pass, and submit the command buffer before sampling it. The staging
/// buffer remains owned by this object and is also retained by the submission
/// recording until its fence completes.
///
/// The upload generates a complete mip chain with linear blits. Formats and devices
/// that cannot support sampled-image linear filtering and transfer blits are rejected.
pub struct VulkanTexture2D<'device> {
    image: VulkanImage<'device>,
    staging: VulkanBuffer<'device>,
    sampler: VulkanSampler<'device>,
    upload_state: Cell<UploadState>,
}

impl<'device> VulkanTexture2D<'device> {
    /// Creates a Vulkan texture from one imported glTF texture table entry.
    /// The caller selects sRGB for base-color/emissive data and linear for data maps.
    pub fn from_import(
        device: &'device VulkanDevice,
        import: &GltfImport,
        texture_index: usize,
        color_space: TextureColorSpace,
    ) -> Result<Self, TextureUploadError> {
        let texture = import
            .textures
            .get(texture_index)
            .ok_or(TextureUploadError::MissingTexture(texture_index))?;
        let image = import
            .images
            .get(texture.source_image)
            .ok_or(TextureUploadError::MissingImage(texture.source_image))?;
        Self::new(device, image, texture, color_space)
    }

    /// Creates a linearly sampled equirectangular HDR environment texture.
    ///
    /// Scene-linear RGB radiance is uploaded as RGBA32F without tone mapping, clamping, or
    /// sRGB conversion. The sampler repeats horizontally and clamps vertically.
    /// A conventional mip chain is generated by linear blits; these mip levels are
    /// not a substitute for roughness-prefiltered environment maps.
    /// The physical device must support sampled RGBA32F images and linear filtering.
    pub fn from_hdr_environment(
        device: &'device VulkanDevice,
        environment: &HdrEnvironmentMap,
    ) -> Result<Self, TextureUploadError> {
        let imported_image = hdr_environment_imported_image(environment)?;
        let imported_texture = ImportedTexture {
            source_image: 0,
            mag_filter: Some(TextureMagFilter::Linear),
            min_filter: Some(TextureMinFilter::LinearMipmapLinear),
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::ClampToEdge,
        };
        Self::new(
            device,
            &imported_image,
            &imported_texture,
            TextureColorSpace::Linear,
        )
    }

    /// Creates a Vulkan texture from decoded image pixels and imported sampler state.
    pub fn new(
        device: &'device VulkanDevice,
        imported_image: &ImportedImage,
        imported_texture: &ImportedTexture,
        color_space: TextureColorSpace,
    ) -> Result<Self, TextureUploadError> {
        let format = image_vk_format(imported_image.format, color_space)?;
        let expected_bytes =
            expected_pixel_bytes(imported_image).ok_or(TextureUploadError::InvalidImageData)?;
        if imported_image.width == 0
            || imported_image.height == 0
            || imported_image.pixels.len() != expected_bytes
        {
            return Err(TextureUploadError::InvalidImageData);
        }

        // SAFETY: Physical-device properties belong to the instance retained by the device.
        let physical_properties = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_properties(device.physical_device)
        };
        if imported_image.width > physical_properties.limits.max_image_dimension2_d
            || imported_image.height > physical_properties.limits.max_image_dimension2_d
        {
            return Err(TextureUploadError::InvalidImageData);
        }

        let mip_levels = mip_level_count(imported_image.width, imported_image.height);
        let sampler_config = sampler_config(imported_texture, mip_levels);
        let mut required_features =
            vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;
        if mip_levels > 1 {
            required_features |= vk::FormatFeatureFlags::BLIT_SRC
                | vk::FormatFeatureFlags::BLIT_DST
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        }
        if sampler_uses_linear_filter(sampler_config) {
            required_features |= vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        }
        // SAFETY: The physical device is owned by the instance retained by this device.
        let format_properties = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_format_properties(device.physical_device, format)
        };
        let available_features = format_properties.optimal_tiling_features;
        if !available_features.contains(required_features) {
            if mip_levels > 1 {
                return Err(TextureUploadError::UnsupportedMipFeatures {
                    format,
                    required: required_features,
                    available: available_features,
                });
            }
            return Err(TextureUploadError::UnsupportedFormatFeatures {
                format,
                required: required_features,
                available: available_features,
            });
        }

        let image = VulkanImage::new(
            device,
            ImageConfig {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: imported_image.width,
                    height: imported_image.height,
                    depth: 1,
                },
                format,
                mip_levels,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
        )?;
        let staging = VulkanBuffer::new(
            device,
            BufferConfig {
                size: u64::try_from(imported_image.pixels.len())
                    .map_err(|_| TextureUploadError::InvalidImageData)?,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory: BufferMemoryLocation::Upload,
            },
        )?;
        let mut staging = staging;
        staging.write(0, &imported_image.pixels)?;
        let sampler = VulkanSampler::new(device, sampler_config)?;

        Ok(Self {
            image,
            staging,
            sampler,
            upload_state: Cell::new(UploadState::Pending),
        })
    }

    /// Returns the owned Vulkan image.
    #[must_use]
    pub const fn image(&self) -> &VulkanImage<'device> {
        &self.image
    }

    /// Creates an image view for descriptor updates.
    pub fn create_view(&self) -> Result<VulkanImageView<'_, 'device>, VulkanError> {
        self.image.create_view(ImageViewConfig {
            level_count: self.image.config().mip_levels,
            ..ImageViewConfig::default()
        })
    }

    /// Returns the imported sampler resource for descriptor updates.
    #[must_use]
    pub const fn sampler(&self) -> &VulkanSampler<'device> {
        &self.sampler
    }

    /// Records the upload and transition to SHADER_READ_ONLY_OPTIMAL.
    ///
    /// This operation must be recorded outside a render pass on a queue supporting
    /// HOST, TRANSFER, and FRAGMENT_SHADER pipeline stages. On error, the command
    /// buffer may contain a partial upload sequence; discard that command buffer
    /// and this upload object rather than retrying either one.
    pub fn record_upload<'recording>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, '_, 'device>,
    ) -> Result<(), TextureUploadError> {
        match self.upload_state.get() {
            UploadState::Pending => {}
            UploadState::Recorded => return Err(TextureUploadError::AlreadyRecorded),
            UploadState::Failed => return Err(TextureUploadError::RecordingFailed),
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
    ) -> Result<(), TextureUploadError> {
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
                layer_count: 1,
            }],
        )?;

        let region = [vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(self.image.config().extent)];
        recording.copy_buffer_to_image(
            &self.staging,
            &self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        )?;

        let mut mip_width = self.image.config().extent.width;
        let mut mip_height = self.image.config().extent.height;
        for mip_level in 1..self.image.config().mip_levels {
            recording.pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[ImageBarrier {
                    image: &self.image,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::TRANSFER_READ,
                    aspect: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_level - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            )?;
            let next_width = (mip_width / 2).max(1);
            let next_height = (mip_height / 2).max(1);
            let blit = [vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip_level - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width as i32,
                        y: mip_height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip_level)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_width as i32,
                        y: next_height as i32,
                        z: 1,
                    },
                ])];
            recording.blit_image(
                &self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blit,
                vk::Filter::LINEAR,
            )?;
            recording.pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[ImageBarrier {
                    image: &self.image,
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    source_access: vk::AccessFlags::TRANSFER_READ,
                    destination_access: vk::AccessFlags::SHADER_READ,
                    aspect: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_level - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            )?;
            mip_width = next_width;
            mip_height = next_height;
        }
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
                base_mip_level: self.image.config().mip_levels - 1,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )?;
        Ok(())
    }
}

fn hdr_environment_imported_image(
    environment: &HdrEnvironmentMap,
) -> Result<ImportedImage, TextureUploadError> {
    let pixel_count = usize::try_from(u64::from(environment.width) * u64::from(environment.height))
        .map_err(|_| TextureUploadError::InvalidImageData)?;
    if environment.width == 0
        || environment.height == 0
        || u64::from(environment.width) != u64::from(environment.height) * 2
        || environment.pixels.len() != pixel_count
    {
        return Err(TextureUploadError::InvalidImageData);
    }
    let byte_count = pixel_count
        .checked_mul(16)
        .ok_or(TextureUploadError::InvalidImageData)?;
    let mut pixels = Vec::with_capacity(byte_count);
    for pixel in &environment.pixels {
        for sample in pixel {
            if !sample.is_finite() || *sample < 0.0 {
                return Err(TextureUploadError::InvalidImageData);
            }
            pixels.extend_from_slice(&sample.to_ne_bytes());
        }
        pixels.extend_from_slice(&1.0_f32.to_ne_bytes());
    }
    if pixels.len() != byte_count {
        return Err(TextureUploadError::InvalidImageData);
    }
    Ok(ImportedImage {
        width: environment.width,
        height: environment.height,
        format: ImagePixelFormat::Rgba32Float,
        pixels,
    })
}

fn image_vk_format(
    format: ImagePixelFormat,
    color_space: TextureColorSpace,
) -> Result<vk::Format, TextureUploadError> {
    let vk_format = match (format, color_space) {
        (ImagePixelFormat::R8, TextureColorSpace::Linear) => vk::Format::R8_UNORM,
        (ImagePixelFormat::Rg8, TextureColorSpace::Linear) => vk::Format::R8G8_UNORM,
        (ImagePixelFormat::Rgb8, TextureColorSpace::Linear) => vk::Format::R8G8B8_UNORM,
        (ImagePixelFormat::Rgba8, TextureColorSpace::Linear) => vk::Format::R8G8B8A8_UNORM,
        (ImagePixelFormat::R16, TextureColorSpace::Linear) => vk::Format::R16_UNORM,
        (ImagePixelFormat::Rg16, TextureColorSpace::Linear) => vk::Format::R16G16_UNORM,
        (ImagePixelFormat::Rgb16, TextureColorSpace::Linear) => vk::Format::R16G16B16_UNORM,
        (ImagePixelFormat::Rgba16, TextureColorSpace::Linear) => vk::Format::R16G16B16A16_UNORM,
        (ImagePixelFormat::Rgb32Float, TextureColorSpace::Linear) => vk::Format::R32G32B32_SFLOAT,
        (ImagePixelFormat::Rgba32Float, TextureColorSpace::Linear) => {
            vk::Format::R32G32B32A32_SFLOAT
        }
        (ImagePixelFormat::R8, TextureColorSpace::Srgb) => vk::Format::R8_SRGB,
        (ImagePixelFormat::Rg8, TextureColorSpace::Srgb) => vk::Format::R8G8_SRGB,
        (ImagePixelFormat::Rgb8, TextureColorSpace::Srgb) => vk::Format::R8G8B8_SRGB,
        (ImagePixelFormat::Rgba8, TextureColorSpace::Srgb) => vk::Format::R8G8B8A8_SRGB,
        (format, TextureColorSpace::Srgb) => {
            return Err(TextureUploadError::UnsupportedColorSpaceFormat {
                format,
                color_space,
            });
        }
    };
    Ok(vk_format)
}

fn mip_level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).leading_zeros()
}

fn expected_pixel_bytes(image: &ImportedImage) -> Option<usize> {
    let bytes_per_pixel = match image.format {
        ImagePixelFormat::R8 => 1_usize,
        ImagePixelFormat::Rg8 => 2,
        ImagePixelFormat::Rgb8 => 3,
        ImagePixelFormat::Rgba8 => 4,
        ImagePixelFormat::R16 => 2,
        ImagePixelFormat::Rg16 => 4,
        ImagePixelFormat::Rgb16 => 6,
        ImagePixelFormat::Rgba16 => 8,
        ImagePixelFormat::Rgb32Float => 12,
        ImagePixelFormat::Rgba32Float => 16,
    };
    usize::try_from(image.width)
        .ok()?
        .checked_mul(usize::try_from(image.height).ok()?)?
        .checked_mul(bytes_per_pixel)
}

fn sampler_config(texture: &ImportedTexture, mip_levels: u32) -> SamplerConfig {
    let (mag_filter, min_filter, mipmap_mode) = match texture.min_filter {
        Some(TextureMinFilter::Nearest) => (
            texture.mag_filter,
            vk::Filter::NEAREST,
            vk::SamplerMipmapMode::NEAREST,
        ),
        Some(TextureMinFilter::Linear) => (
            texture.mag_filter,
            vk::Filter::LINEAR,
            vk::SamplerMipmapMode::NEAREST,
        ),
        Some(TextureMinFilter::NearestMipmapNearest) => (
            texture.mag_filter,
            vk::Filter::NEAREST,
            vk::SamplerMipmapMode::NEAREST,
        ),
        Some(TextureMinFilter::LinearMipmapNearest) => (
            texture.mag_filter,
            vk::Filter::LINEAR,
            vk::SamplerMipmapMode::NEAREST,
        ),
        Some(TextureMinFilter::NearestMipmapLinear) => (
            texture.mag_filter,
            vk::Filter::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ),
        Some(TextureMinFilter::LinearMipmapLinear) | None => (
            texture.mag_filter,
            vk::Filter::LINEAR,
            vk::SamplerMipmapMode::LINEAR,
        ),
    };
    let mag_filter = match mag_filter {
        Some(TextureMagFilter::Nearest) => vk::Filter::NEAREST,
        Some(TextureMagFilter::Linear) | None => vk::Filter::LINEAR,
    };
    SamplerConfig {
        mag_filter,
        min_filter,
        mipmap_mode,
        address_mode_u: address_mode(texture.wrap_s),
        address_mode_v: address_mode(texture.wrap_t),
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        min_lod: 0.0,
        max_lod: (mip_levels - 1) as f32,
        ..SamplerConfig::default()
    }
}

fn address_mode(mode: TextureWrapMode) -> vk::SamplerAddressMode {
    match mode {
        TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        TextureWrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}

fn sampler_uses_linear_filter(config: SamplerConfig) -> bool {
    config.mag_filter == vk::Filter::LINEAR
        || config.min_filter == vk::Filter::LINEAR
        || config.mipmap_mode == vk::SamplerMipmapMode::LINEAR
}

#[cfg(test)]
mod tests {
    use super::{
        TextureColorSpace, TextureUploadError, address_mode, expected_pixel_bytes,
        hdr_environment_imported_image, image_vk_format, mip_level_count, sampler_config,
        sampler_uses_linear_filter,
    };
    use ash::vk;
    use z_ideal_assets::{
        HdrEnvironmentMap, ImagePixelFormat, ImportedImage, ImportedTexture, TextureMagFilter,
        TextureMinFilter, TextureWrapMode,
    };

    #[test]
    fn hdr_environment_conversion_preserves_scene_linear_rgb32f_bytes() {
        let environment = HdrEnvironmentMap {
            source_path: "studio.hdr".into(),
            width: 2,
            height: 1,
            pixels: vec![[1.0, 0.5, 0.25], [4.0, 2.0, 8.0]],
        };
        let imported = hdr_environment_imported_image(&environment)
            .expect("valid equirectangular HDRI converts to RGBA32F image");
        assert_eq!(imported.width, 2);
        assert_eq!(imported.height, 1);
        assert_eq!(imported.format, ImagePixelFormat::Rgba32Float);
        assert_eq!(imported.pixels.len(), 32);
        let decoded: Vec<f32> = imported
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_ne_bytes(*bytes))
            .collect();
        assert_eq!(decoded, [1.0, 0.5, 0.25, 1.0, 4.0, 2.0, 8.0, 1.0]);
    }

    #[test]
    fn hdr_environment_conversion_rejects_invalid_dimensions_storage_and_samples() {
        let mut environment = HdrEnvironmentMap {
            source_path: "invalid.hdr".into(),
            width: 2,
            height: 1,
            pixels: vec![[1.0, 1.0, 1.0]],
        };
        assert!(matches!(
            hdr_environment_imported_image(&environment),
            Err(TextureUploadError::InvalidImageData)
        ));
        environment.pixels = vec![[1.0, f32::NAN, 1.0], [1.0, 1.0, 1.0]];
        assert!(matches!(
            hdr_environment_imported_image(&environment),
            Err(TextureUploadError::InvalidImageData)
        ));
        environment.pixels = vec![[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]];
        assert!(matches!(
            hdr_environment_imported_image(&environment),
            Err(TextureUploadError::InvalidImageData)
        ));
        environment.width = 3;
        assert!(matches!(
            hdr_environment_imported_image(&environment),
            Err(TextureUploadError::InvalidImageData)
        ));
    }

    #[test]
    fn maps_imported_image_formats_to_vulkan_formats_and_checked_byte_lengths() {
        let cases = [
            (ImagePixelFormat::R8, vk::Format::R8_UNORM, 4),
            (ImagePixelFormat::Rg8, vk::Format::R8G8_UNORM, 8),
            (ImagePixelFormat::Rgb8, vk::Format::R8G8B8_UNORM, 12),
            (ImagePixelFormat::Rgba8, vk::Format::R8G8B8A8_UNORM, 16),
            (ImagePixelFormat::R16, vk::Format::R16_UNORM, 8),
            (ImagePixelFormat::Rg16, vk::Format::R16G16_UNORM, 16),
            (ImagePixelFormat::Rgb16, vk::Format::R16G16B16_UNORM, 24),
            (ImagePixelFormat::Rgba16, vk::Format::R16G16B16A16_UNORM, 32),
            (
                ImagePixelFormat::Rgb32Float,
                vk::Format::R32G32B32_SFLOAT,
                48,
            ),
            (
                ImagePixelFormat::Rgba32Float,
                vk::Format::R32G32B32A32_SFLOAT,
                64,
            ),
        ];
        for (format, expected_format, expected_bytes) in cases {
            let image = ImportedImage {
                width: 2,
                height: 2,
                format,
                pixels: vec![0; expected_bytes],
            };
            assert_eq!(
                image_vk_format(format, TextureColorSpace::Linear).expect("linear format"),
                expected_format
            );
            assert_eq!(expected_pixel_bytes(&image), Some(expected_bytes));
        }
    }

    #[test]
    fn maps_imported_filtering_and_address_modes_without_mip_access() {
        let texture = ImportedTexture {
            source_image: 0,
            mag_filter: Some(TextureMagFilter::Nearest),
            min_filter: Some(TextureMinFilter::LinearMipmapLinear),
            wrap_s: TextureWrapMode::MirroredRepeat,
            wrap_t: TextureWrapMode::ClampToEdge,
        };
        let config = sampler_config(&texture, 5);
        assert_eq!(config.mag_filter, vk::Filter::NEAREST);
        assert_eq!(config.min_filter, vk::Filter::LINEAR);
        assert_eq!(config.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!(
            config.address_mode_u,
            vk::SamplerAddressMode::MIRRORED_REPEAT
        );
        assert_eq!(config.address_mode_v, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(config.max_lod, 4.0);
        assert_eq!(
            address_mode(TextureWrapMode::Repeat),
            vk::SamplerAddressMode::REPEAT
        );
    }

    #[test]
    fn gltf_default_sampler_filters_to_linear_level_zero() {
        let texture = ImportedTexture {
            source_image: 0,
            mag_filter: None,
            min_filter: None,
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
        };
        let config = sampler_config(&texture, 1);
        assert_eq!(config.mag_filter, vk::Filter::LINEAR);
        assert_eq!(config.min_filter, vk::Filter::LINEAR);
        assert_eq!(config.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!(config.min_lod, 0.0);
        assert_eq!(config.max_lod, 0.0);
    }

    #[test]
    fn selects_srgb_formats_only_for_supported_eight_bit_channels() {
        assert_eq!(
            image_vk_format(ImagePixelFormat::Rgba8, TextureColorSpace::Srgb).expect("sRGB RGBA"),
            vk::Format::R8G8B8A8_SRGB
        );
        assert!(matches!(
            image_vk_format(ImagePixelFormat::Rgba16, TextureColorSpace::Srgb),
            Err(super::TextureUploadError::UnsupportedColorSpaceFormat { .. })
        ));
    }

    #[test]
    fn computes_complete_mip_chain_for_non_power_of_two_extents() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(2, 2), 2);
        assert_eq!(mip_level_count(8, 4), 4);
        assert_eq!(mip_level_count(7, 3), 3);
    }

    #[test]
    fn mip_interpolation_requires_linear_sampling_support() {
        let texture = ImportedTexture {
            source_image: 0,
            mag_filter: Some(TextureMagFilter::Nearest),
            min_filter: Some(TextureMinFilter::NearestMipmapLinear),
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
        };
        assert!(sampler_uses_linear_filter(sampler_config(&texture, 4)));
    }
}
