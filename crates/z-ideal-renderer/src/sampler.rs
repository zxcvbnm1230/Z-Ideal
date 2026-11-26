//! Explicitly owned Vulkan sampler resources.
use ash::vk;

use crate::{VulkanDevice, VulkanError, api_error};

/// Configuration for a Vulkan texture sampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// Filtering used when sampling magnified texels.
    pub mag_filter: vk::Filter,
    /// Filtering used when sampling minified texels.
    pub min_filter: vk::Filter,
    /// Filtering between mip levels.
    pub mipmap_mode: vk::SamplerMipmapMode,
    /// Addressing policy for the U texture coordinate.
    pub address_mode_u: vk::SamplerAddressMode,
    /// Addressing policy for the V texture coordinate.
    pub address_mode_v: vk::SamplerAddressMode,
    /// Addressing policy for the W texture coordinate.
    pub address_mode_w: vk::SamplerAddressMode,
    /// Constant LOD bias applied during sampling.
    pub mip_lod_bias: f32,
    /// Optional maximum anisotropy; requires the device feature.
    pub max_anisotropy: Option<f32>,
    /// Minimum accessible mip level.
    pub min_lod: f32,
    /// Maximum accessible mip level.
    pub max_lod: f32,
    /// Border color used by clamp-to-border addressing modes.
    pub border_color: vk::BorderColor,
    /// Optional depth comparison operation for shadow sampling.
    pub compare_op: Option<vk::CompareOp>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: 0.0,
            max_anisotropy: None,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            compare_op: None,
        }
    }
}

/// Owns a Vulkan sampler and borrows the logical device that created it.
///
/// The sampler does not own an image or image view. Descriptor bindings and
/// resource synchronization remain the caller's responsibility.
pub struct VulkanSampler<'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::Sampler,
    config: SamplerConfig,
}

impl<'device> VulkanSampler<'device> {
    /// Creates a sampler after validating device-dependent limits and features.
    pub fn new(device: &'device VulkanDevice, config: SamplerConfig) -> Result<Self, VulkanError> {
        validate_sampler_config(
            config,
            device.sampler_anisotropy_enabled,
            device.max_sampler_anisotropy,
            device.max_sampler_lod_bias,
        )?;

        let info = vk::SamplerCreateInfo::default()
            .mag_filter(config.mag_filter)
            .min_filter(config.min_filter)
            .mipmap_mode(config.mipmap_mode)
            .address_mode_u(config.address_mode_u)
            .address_mode_v(config.address_mode_v)
            .address_mode_w(config.address_mode_w)
            .mip_lod_bias(config.mip_lod_bias)
            .anisotropy_enable(config.max_anisotropy.is_some())
            .max_anisotropy(config.max_anisotropy.unwrap_or(1.0))
            .compare_enable(config.compare_op.is_some())
            .compare_op(config.compare_op.unwrap_or(vk::CompareOp::ALWAYS))
            .min_lod(config.min_lod)
            .max_lod(config.max_lod)
            .border_color(config.border_color)
            .unnormalized_coordinates(false);

        // SAFETY: The device is live and the create info contains no borrowed pointers.
        let handle = unsafe {
            device
                .handle
                .create_sampler(&info, None)
                .map_err(|error| api_error("create Vulkan sampler", error))?
        };

        Ok(Self {
            device,
            handle,
            config,
        })
    }

    /// Returns the raw sampler handle for descriptor updates.
    #[must_use]
    pub const fn handle(&self) -> vk::Sampler {
        self.handle
    }

    /// Returns the sampler configuration used at creation.
    #[must_use]
    pub const fn config(&self) -> SamplerConfig {
        self.config
    }
}

impl Drop for VulkanSampler<'_> {
    fn drop(&mut self) {
        // SAFETY: Descriptor leases retained by submitted command buffers borrow this sampler
        // until their submission fences complete. Raw Vulkan users synchronize manually.
        unsafe { self.device.handle.destroy_sampler(self.handle, None) };
    }
}

fn validate_sampler_config(
    config: SamplerConfig,
    anisotropy_enabled: bool,
    max_anisotropy: f32,
    max_lod_bias: f32,
) -> Result<(), VulkanError> {
    let valid_filter = |filter| matches!(filter, vk::Filter::NEAREST | vk::Filter::LINEAR);
    let valid_address_mode = |mode| {
        matches!(
            mode,
            vk::SamplerAddressMode::REPEAT
                | vk::SamplerAddressMode::MIRRORED_REPEAT
                | vk::SamplerAddressMode::CLAMP_TO_EDGE
                | vk::SamplerAddressMode::CLAMP_TO_BORDER
        )
    };
    let valid_border_color = matches!(
        config.border_color,
        vk::BorderColor::FLOAT_TRANSPARENT_BLACK
            | vk::BorderColor::INT_TRANSPARENT_BLACK
            | vk::BorderColor::FLOAT_OPAQUE_BLACK
            | vk::BorderColor::INT_OPAQUE_BLACK
            | vk::BorderColor::FLOAT_OPAQUE_WHITE
            | vk::BorderColor::INT_OPAQUE_WHITE
    );
    let valid_compare_op = config.compare_op.is_none_or(|op| {
        matches!(
            op,
            vk::CompareOp::NEVER
                | vk::CompareOp::LESS
                | vk::CompareOp::EQUAL
                | vk::CompareOp::LESS_OR_EQUAL
                | vk::CompareOp::GREATER
                | vk::CompareOp::NOT_EQUAL
                | vk::CompareOp::GREATER_OR_EQUAL
                | vk::CompareOp::ALWAYS
        )
    });
    if !valid_filter(config.mag_filter)
        || !valid_filter(config.min_filter)
        || !matches!(
            config.mipmap_mode,
            vk::SamplerMipmapMode::NEAREST | vk::SamplerMipmapMode::LINEAR
        )
        || !valid_address_mode(config.address_mode_u)
        || !valid_address_mode(config.address_mode_v)
        || !valid_address_mode(config.address_mode_w)
        || !valid_border_color
        || !valid_compare_op
    {
        return Err(VulkanError::InvalidSamplerEnum);
    }

    if !config.mip_lod_bias.is_finite() || config.mip_lod_bias.abs() > max_lod_bias {
        return Err(VulkanError::InvalidSamplerLodBias);
    }
    if !config.min_lod.is_finite() || !config.max_lod.is_finite() || config.min_lod > config.max_lod
    {
        return Err(VulkanError::InvalidSamplerLodRange);
    }
    if let Some(anisotropy) = config.max_anisotropy {
        if !anisotropy.is_finite() || anisotropy < 1.0 || anisotropy > max_anisotropy {
            return Err(VulkanError::InvalidSamplerAnisotropy);
        }
        if !anisotropy_enabled {
            return Err(VulkanError::SamplerAnisotropyUnavailable);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SamplerConfig, validate_sampler_config};
    use crate::VulkanError;
    use ash::vk;

    const MAX_ANISOTROPY: f32 = 16.0;
    const MAX_LOD_BIAS: f32 = 4.0;

    #[test]
    fn default_sampler_configuration_is_valid_without_optional_features() {
        assert!(
            validate_sampler_config(
                SamplerConfig::default(),
                false,
                MAX_ANISOTROPY,
                MAX_LOD_BIAS
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_non_finite_or_out_of_range_lod_bias() {
        for bias in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 4.01, -4.01] {
            let config = SamplerConfig {
                mip_lod_bias: bias,
                ..SamplerConfig::default()
            };
            assert!(matches!(
                validate_sampler_config(config, true, MAX_ANISOTROPY, MAX_LOD_BIAS),
                Err(VulkanError::InvalidSamplerLodBias)
            ));
        }
    }

    #[test]
    fn rejects_invalid_lod_ranges() {
        for (min_lod, max_lod) in [(f32::NAN, 1.0), (0.0, f32::INFINITY), (2.0, 1.0)] {
            let config = SamplerConfig {
                min_lod,
                max_lod,
                ..SamplerConfig::default()
            };
            assert!(validate_sampler_config(config, true, MAX_ANISOTROPY, MAX_LOD_BIAS).is_err());
        }
    }

    #[test]
    fn rejects_unknown_vulkan_enum_values() {
        let config = SamplerConfig {
            mag_filter: vk::Filter::from_raw(-1),
            ..SamplerConfig::default()
        };
        assert!(matches!(
            validate_sampler_config(config, false, 1.0, MAX_LOD_BIAS),
            Err(VulkanError::InvalidSamplerEnum)
        ));
    }

    #[test]
    fn anisotropy_requires_enabled_feature_and_device_supported_range() {
        let config = SamplerConfig {
            max_anisotropy: Some(8.0),
            ..SamplerConfig::default()
        };
        assert!(validate_sampler_config(config, true, MAX_ANISOTROPY, MAX_LOD_BIAS).is_ok());
        assert!(validate_sampler_config(config, false, MAX_ANISOTROPY, MAX_LOD_BIAS).is_err());

        for value in [f32::NAN, f32::INFINITY, 0.5, 16.1] {
            let config = SamplerConfig {
                max_anisotropy: Some(value),
                ..SamplerConfig::default()
            };
            assert!(validate_sampler_config(config, true, MAX_ANISOTROPY, MAX_LOD_BIAS).is_err());
        }
    }

    #[test]
    fn comparison_sampler_configuration_is_preserved() {
        let config = SamplerConfig {
            compare_op: Some(vk::CompareOp::LESS_OR_EQUAL),
            ..SamplerConfig::default()
        };
        assert!(validate_sampler_config(config, false, 1.0, MAX_LOD_BIAS).is_ok());
        assert_eq!(config.compare_op, Some(vk::CompareOp::LESS_OR_EQUAL));
    }
}
