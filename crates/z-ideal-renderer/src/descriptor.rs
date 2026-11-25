//! Explicitly owned Vulkan descriptor layouts and pools.
use ash::vk;

use crate::{VulkanDevice, VulkanError, api_error};

/// One binding in a descriptor-set layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBinding {
    /// Binding number visible to shaders.
    pub binding: u32,
    /// Kind of resource accepted by this binding.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors in this binding.
    pub descriptor_count: u32,
    /// Shader stages allowed to access this binding.
    pub stage_flags: vk::ShaderStageFlags,
}

/// Configuration for a descriptor-set layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescriptorSetLayoutConfig {
    /// Bindings in the layout; an empty layout is valid.
    pub bindings: Vec<DescriptorBinding>,
}

/// Owns a descriptor-set layout and borrows its creating logical device.
pub struct VulkanDescriptorSetLayout<'device> {
    device: &'device VulkanDevice,
    handle: vk::DescriptorSetLayout,
    config: DescriptorSetLayoutConfig,
}

impl<'device> VulkanDescriptorSetLayout<'device> {
    /// Creates a descriptor-set layout after validating binding declarations.
    pub fn new(
        device: &'device VulkanDevice,
        config: DescriptorSetLayoutConfig,
    ) -> Result<Self, VulkanError> {
        validate_layout_config(&config)?;
        let bindings: Vec<_> = config
            .bindings
            .iter()
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(binding.descriptor_type)
                    .descriptor_count(binding.descriptor_count)
                    .stage_flags(binding.stage_flags)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        // SAFETY: The device is live and all binding data remains valid for the call.
        let handle = unsafe {
            device
                .handle
                .create_descriptor_set_layout(&info, None)
                .map_err(|error| api_error("create Vulkan descriptor-set layout", error))?
        };

        Ok(Self {
            device,
            handle,
            config,
        })
    }

    /// Returns the raw layout handle for pipeline-layout creation.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    /// Returns the validated layout configuration.
    #[must_use]
    pub const fn config(&self) -> &DescriptorSetLayoutConfig {
        &self.config
    }
}

impl Drop for VulkanDescriptorSetLayout<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before descriptor-set-layout destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the layout and its device is still borrowed.
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

/// Descriptor counts available from a descriptor pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorPoolSize {
    /// Descriptor kind accounted for by this pool entry.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors of this kind.
    pub descriptor_count: u32,
}

/// Configuration for a descriptor pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorPoolConfig {
    /// Maximum number of descriptor sets that may be allocated.
    pub max_sets: u32,
    /// Descriptor capacities by descriptor kind.
    pub pool_sizes: Vec<DescriptorPoolSize>,
}

/// Owns a Vulkan descriptor pool and all sets allocated from it.
///
/// Individual sets are not freed. They borrow this pool and are released together
/// when the pool is destroyed.
pub struct VulkanDescriptorPool<'device> {
    device: &'device VulkanDevice,
    handle: vk::DescriptorPool,
}

impl<'device> VulkanDescriptorPool<'device> {
    /// Creates a descriptor pool with explicit set and descriptor capacities.
    pub fn new(
        device: &'device VulkanDevice,
        config: DescriptorPoolConfig,
    ) -> Result<Self, VulkanError> {
        validate_pool_config(&config)?;
        let pool_sizes: Vec<_> = config
            .pool_sizes
            .iter()
            .map(|size| {
                vk::DescriptorPoolSize::default()
                    .ty(size.descriptor_type)
                    .descriptor_count(size.descriptor_count)
            })
            .collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(config.max_sets)
            .pool_sizes(&pool_sizes);

        // SAFETY: The device is live and pool-size data remains valid for the call.
        let handle = unsafe {
            device
                .handle
                .create_descriptor_pool(&info, None)
                .map_err(|error| api_error("create Vulkan descriptor pool", error))?
        };

        Ok(Self { device, handle })
    }

    /// Returns the raw descriptor-pool handle.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorPool {
        self.handle
    }

    /// Allocates one descriptor set that borrows both its pool and layout.
    ///
    /// The descriptor set is automatically released with its pool. This initial
    /// API intentionally exposes allocation only; resource writes are handled by
    /// a later descriptor-update layer with explicit resource-lifetime contracts.
    pub fn allocate_set<'layout>(
        &self,
        layout: &'layout VulkanDescriptorSetLayout<'device>,
    ) -> Result<VulkanDescriptorSet<'_, 'layout, 'device>, VulkanError> {
        if !std::ptr::eq(self.device, layout.device) {
            return Err(VulkanError::DescriptorSetDeviceMismatch);
        }
        let layouts = [layout.handle];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(&layouts);

        // SAFETY: Pool and layout belong to the same live device and remain borrowed
        // for the full lifetime of the returned descriptor-set wrapper.
        let mut sets = unsafe {
            self.device
                .handle
                .allocate_descriptor_sets(&info)
                .map_err(|error| api_error("allocate Vulkan descriptor set", error))?
        };
        let handle = sets.pop().ok_or_else(|| {
            api_error("allocate Vulkan descriptor set", vk::Result::ERROR_UNKNOWN)
        })?;

        Ok(VulkanDescriptorSet {
            _pool: self,
            layout,
            handle,
        })
    }
}

impl Drop for VulkanDescriptorPool<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before descriptor-pool destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the pool; destroying it releases all sets.
        unsafe {
            self.device
                .handle
                .destroy_descriptor_pool(self.handle, None)
        };
    }
}

/// A descriptor set allocated from a pool and compatible with a borrowed layout.
///
/// The set itself is owned by the descriptor pool, so this wrapper does not
/// individually free the Vulkan descriptor-set handle.
pub struct VulkanDescriptorSet<'pool, 'layout, 'device> {
    _pool: &'pool VulkanDescriptorPool<'device>,
    layout: &'layout VulkanDescriptorSetLayout<'device>,
    handle: vk::DescriptorSet,
}

impl<'pool, 'layout, 'device> VulkanDescriptorSet<'pool, 'layout, 'device> {
    /// Returns the raw descriptor-set handle for binding or descriptor updates.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }

    /// Returns the layout used to allocate this descriptor set.
    #[must_use]
    pub const fn layout(&self) -> &'layout VulkanDescriptorSetLayout<'device> {
        self.layout
    }
}

fn validate_layout_config(config: &DescriptorSetLayoutConfig) -> Result<(), VulkanError> {
    for (index, binding) in config.bindings.iter().enumerate() {
        if binding.descriptor_count == 0 || binding.stage_flags.is_empty() {
            return Err(VulkanError::InvalidDescriptorSetLayoutBinding);
        }
        if config.bindings[..index]
            .iter()
            .any(|previous| previous.binding == binding.binding)
        {
            return Err(VulkanError::InvalidDescriptorSetLayoutBinding);
        }
    }
    Ok(())
}

fn validate_pool_config(config: &DescriptorPoolConfig) -> Result<(), VulkanError> {
    if config.max_sets == 0 || config.pool_sizes.is_empty() {
        return Err(VulkanError::InvalidDescriptorPoolConfig);
    }
    for (index, size) in config.pool_sizes.iter().enumerate() {
        if size.descriptor_count == 0
            || config.pool_sizes[..index]
                .iter()
                .any(|previous| previous.descriptor_type == size.descriptor_type)
        {
            return Err(VulkanError::InvalidDescriptorPoolConfig);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DescriptorBinding, DescriptorPoolConfig, DescriptorPoolSize, DescriptorSetLayoutConfig,
        validate_layout_config, validate_pool_config,
    };
    use crate::VulkanError;
    use ash::vk;

    #[test]
    fn accepts_empty_descriptor_set_layout() {
        assert!(validate_layout_config(&DescriptorSetLayoutConfig::default()).is_ok());
    }

    #[test]
    fn accepts_valid_descriptor_set_layout_bindings() {
        let config = DescriptorSetLayoutConfig {
            bindings: vec![
                DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                },
                DescriptorBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 4,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                },
            ],
        };
        assert!(validate_layout_config(&config).is_ok());
    }

    #[test]
    fn rejects_zero_count_empty_stages_and_duplicate_bindings() {
        let invalid = [
            DescriptorSetLayoutConfig {
                bindings: vec![DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 0,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                }],
            },
            DescriptorSetLayoutConfig {
                bindings: vec![DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::empty(),
                }],
            },
            DescriptorSetLayoutConfig {
                bindings: vec![
                    DescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX,
                    },
                    DescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                ],
            },
        ];
        for config in invalid {
            assert!(matches!(
                validate_layout_config(&config),
                Err(VulkanError::InvalidDescriptorSetLayoutBinding)
            ));
        }
    }

    #[test]
    fn accepts_valid_descriptor_pool_configuration() {
        let config = DescriptorPoolConfig {
            max_sets: 8,
            pool_sizes: vec![
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 8,
                },
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 16,
                },
            ],
        };
        assert!(validate_pool_config(&config).is_ok());
    }

    #[test]
    fn rejects_empty_or_invalid_descriptor_pool_configuration() {
        let invalid = [
            DescriptorPoolConfig {
                max_sets: 0,
                pool_sizes: vec![DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                }],
            },
            DescriptorPoolConfig {
                max_sets: 1,
                pool_sizes: vec![],
            },
            DescriptorPoolConfig {
                max_sets: 1,
                pool_sizes: vec![DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 0,
                }],
            },
            DescriptorPoolConfig {
                max_sets: 1,
                pool_sizes: vec![
                    DescriptorPoolSize {
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                    },
                    DescriptorPoolSize {
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 2,
                    },
                ],
            },
        ];
        for config in invalid {
            assert!(matches!(
                validate_pool_config(&config),
                Err(VulkanError::InvalidDescriptorPoolConfig)
            ));
        }
    }
}
