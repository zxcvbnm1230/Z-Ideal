//! Validated floating-point color targets for high-dynamic-range scene rendering.
use ash::vk;
use thiserror::Error;

use crate::{
    ImageConfig, ImageViewConfig, VulkanDevice, VulkanError, VulkanImage, VulkanImageView,
};

/// Floating-point format used for scene-linear HDR color.
pub const HDR_COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Errors while creating a high-dynamic-range color target.
#[derive(Debug, Error)]
pub enum HdrTargetError {
    /// The requested two-dimensional target extent is zero or exceeds device limits.
    #[error("invalid HDR target extent {width}x{height}")]
    InvalidExtent { width: u32, height: u32 },
    /// The device cannot render to and sample the selected floating-point format.
    #[error(
        "Vulkan format {format:?} lacks required HDR target features {required:?}; available {available:?}"
    )]
    UnsupportedFormatFeatures {
        format: vk::Format,
        required: vk::FormatFeatureFlags,
        available: vk::FormatFeatureFlags,
    },
    /// Vulkan image or view creation failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

/// Owns a single-sample, scene-linear HDR color image.
///
/// The target is intended for a scene render pass with a color attachment and a
/// subsequent tone-mapping pass that samples its view. Render-pass/framebuffer
/// ownership and the transition to shader-read layout remain explicit.
pub struct VulkanHdrColorTarget<'device> {
    image: VulkanImage<'device>,
}

impl<'device> VulkanHdrColorTarget<'device> {
    /// Creates an R16G16B16A16_SFLOAT target after checking attachment and sampling support.
    pub fn new(
        device: &'device VulkanDevice,
        width: u32,
        height: u32,
    ) -> Result<Self, HdrTargetError> {
        // SAFETY: Physical-device properties belong to the instance retained by this device.
        let properties = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_properties(device.physical_device)
        };
        if width == 0
            || height == 0
            || width > properties.limits.max_image_dimension2_d
            || height > properties.limits.max_image_dimension2_d
        {
            return Err(HdrTargetError::InvalidExtent { width, height });
        }
        // SAFETY: The physical device is owned by the instance retained by this device.
        let format_properties = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_format_properties(device.physical_device, HDR_COLOR_FORMAT)
        };
        let required =
            vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        let available = format_properties.optimal_tiling_features;
        if !available.contains(required) {
            return Err(HdrTargetError::UnsupportedFormatFeatures {
                format: HDR_COLOR_FORMAT,
                required,
                available,
            });
        }
        let image = VulkanImage::new(
            device,
            ImageConfig {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
                format: HDR_COLOR_FORMAT,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
        )?;
        Ok(Self { image })
    }

    /// Returns the owned floating-point image.
    #[must_use]
    pub const fn image(&self) -> &VulkanImage<'device> {
        &self.image
    }

    /// Creates a color view for framebuffer attachment or sampled-image descriptors.
    pub fn create_view(&self) -> Result<VulkanImageView<'_, 'device>, VulkanError> {
        self.image.create_view(ImageViewConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::HDR_COLOR_FORMAT;
    use ash::vk;

    #[test]
    fn hdr_target_uses_a_linear_floating_point_format() {
        assert_eq!(HDR_COLOR_FORMAT, vk::Format::R16G16B16A16_SFLOAT);
    }
}
