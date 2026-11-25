//! Vulkan loader, instance, validation and logical-device bootstrap.
#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    ffi::{CStr, CString, NulError},
    sync::{Arc, Mutex},
};

use ash::{Entry, vk};
use thiserror::Error;
use winit::window::Window;

mod buffer;
mod command;
mod descriptor;
mod graphics_pipeline;
mod image;
mod pipeline;
mod pipeline_layout;
mod render_pass;
mod sampler;
mod shader;
mod surface;
mod swapchain;
mod sync;

pub use buffer::{BufferConfig, BufferMemoryLocation, VulkanBuffer};
pub use command::{
    BufferBarrier, CommandBufferRecording, CommandPoolConfig, ExecutableCommandBuffer,
    ImageBarrier, VulkanCommandBuffer, VulkanCommandPool, VulkanQueue, VulkanSubmission,
};
pub use descriptor::{
    BufferDescriptorWrite, DescriptorBinding, DescriptorPoolConfig, DescriptorPoolSize,
    DescriptorResourceLease, DescriptorResourceWrite, DescriptorSetLayoutConfig,
    DescriptorSetLeaseRef, ImageDescriptorWrite, SamplerDescriptorWrite, VulkanDescriptorPool,
    VulkanDescriptorSet, VulkanDescriptorSetLayout,
};
pub use graphics_pipeline::{GraphicsPipelineConfig, VulkanGraphicsPipeline};
pub use image::{ImageConfig, ImageViewConfig, VulkanImage, VulkanImageView};
pub use pipeline::VulkanComputePipeline;
pub use pipeline_layout::VulkanPipelineLayout;
pub use render_pass::{
    RenderPassAttachmentConfig, RenderPassAttachmentReference, RenderPassConfig,
    RenderPassDependency, RenderPassSubpassConfig, VulkanFramebuffer, VulkanRenderPass,
};
pub use sampler::{SamplerConfig, VulkanSampler};
pub use shader::{
    CompiledShader, ShaderCompileConfig, ShaderDescriptorBinding, ShaderReflection, ShaderStage,
    ShaderVertexInput, VulkanShaderModule, compile_glsl,
};
pub use surface::VulkanSurface;
pub use swapchain::{AcquireResult, PresentResult, SwapchainConfig, VulkanSwapchain};
pub use sync::{FenceConfig, FenceWaitResult, VulkanFence, VulkanSemaphore};

/// Stable package identifier.
pub const CRATE_NAME: &str = "z-ideal-renderer";

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";
const DEBUG_UTILS_EXTENSION: &CStr = c"VK_EXT_debug_utils";
const PORTABILITY_ENUMERATION_EXTENSION: &CStr = c"VK_KHR_portability_enumeration";
const PORTABILITY_SUBSET_EXTENSION: &CStr = c"VK_KHR_portability_subset";
const SWAPCHAIN_EXTENSION: &CStr = ash::khr::swapchain::NAME;

/// Vulkan instance and device creation options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanConfig {
    /// Application name reported to the driver.
    pub application_name: String,
    /// Require the Khronos validation layer and debug-utils extension.
    pub enable_validation: bool,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self {
            application_name: "Z Ideal".to_owned(),
            enable_validation: cfg!(debug_assertions),
        }
    }
}

/// Vulkan initialization and device-selection failures.
#[derive(Debug, Error)]
pub enum VulkanError {
    /// The system Vulkan loader could not be loaded.
    #[error("could not load Vulkan loader: {0}")]
    Loader(#[from] ash::LoadingError),
    /// A Vulkan API operation failed.
    #[error("{operation} failed: {result:?}")]
    Api {
        operation: &'static str,
        result: vk::Result,
    },
    /// A supplied name contained an interior NUL byte.
    #[error("invalid Vulkan application name: {0}")]
    InvalidName(#[from] NulError),
    /// An instance extension required by the selected configuration is missing.
    #[error("required Vulkan instance extension is unavailable: {0}")]
    MissingInstanceExtension(String),
    /// The requested validation layer is missing.
    #[error("requested Vulkan validation layer is unavailable: {0}")]
    MissingValidationLayer(String),
    /// No graphics-capable device is available.
    #[error("no Vulkan physical device exposes a graphics-capable queue family")]
    NoGraphicsDevice,
    /// No graphics device can present to the selected window surface.
    #[error("no Vulkan physical device supports both graphics and presentation for this surface")]
    NoPresentDevice,
    /// Native surface creation or compatibility failed.
    #[error("Vulkan surface operation failed: {0}")]
    Surface(String),
    /// No surface format is available.
    #[error("the Vulkan surface exposes no image formats")]
    NoSurfaceFormats,
    /// No presentation mode is available.
    #[error("the Vulkan surface exposes no presentation modes")]
    NoPresentModes,
    /// The selected surface cannot be used as a color attachment.
    #[error("the Vulkan surface does not support color-attachment swapchain images")]
    UnsupportedSurfaceUsage,
    /// No supported composite-alpha mode is available.
    #[error("the Vulkan surface exposes no supported composite-alpha mode")]
    NoCompositeAlphaMode,
    /// The requested swapchain extent is zero, typically because the window is minimized.
    #[error("the window has a zero drawable extent; swapchain creation is deferred")]
    ZeroExtent,
    /// No window surface is attached to this context.
    #[error("this Vulkan context has no presentation surface")]
    MissingSurface,
    /// The logical device was created without presentation and swapchain support.
    #[error("the Vulkan logical device was not created with presentation support")]
    PresentationNotEnabled,
    /// No swapchain is currently available.
    #[error("no Vulkan swapchain is currently available")]
    MissingSwapchain,
    /// An image index is outside the current swapchain image range.
    #[error("swapchain image index {index} is out of range for {image_count} images")]
    InvalidSwapchainImageIndex { index: u32, image_count: usize },
    /// Image acquisition requires either a semaphore or fence.
    #[error("swapchain image acquisition requires a semaphore or fence")]
    MissingAcquireSynchronization,
    /// A Vulkan buffer must have a nonzero size.
    #[error("Vulkan buffer size must be greater than zero")]
    InvalidBufferSize,
    /// A Vulkan buffer must declare at least one usage flag.
    #[error("Vulkan buffer usage must not be empty")]
    InvalidBufferUsage,
    /// No compatible memory type satisfies the required properties.
    #[error("no compatible Vulkan memory type satisfies required flags {required:?}")]
    NoCompatibleMemoryType { required: vk::MemoryPropertyFlags },
    /// A host read or write range is outside the logical buffer size.
    #[error("buffer range offset {offset} length {length} exceeds buffer size {capacity}")]
    BufferRangeOutOfBounds {
        offset: vk::DeviceSize,
        length: vk::DeviceSize,
        capacity: vk::DeviceSize,
    },
    /// The selected buffer allocation is not host-visible.
    #[error("Vulkan buffer memory is not host-visible")]
    BufferNotHostVisible,
    /// A host buffer offset cannot be represented as a Rust pointer offset.
    #[error("Vulkan buffer offset exceeds the host addressable range")]
    BufferOffsetTooLarge,
    /// A non-coherent mapped-memory range could not be safely aligned.
    #[error("Vulkan mapped-memory range alignment overflowed or exceeded its allocation")]
    BufferMemoryRangeOverflow,
    /// An image extent contains a zero dimension or violates its dimensionality.
    #[error("Vulkan image extent is invalid for the selected image type")]
    InvalidImageExtent,
    /// An image format must be explicitly selected.
    #[error("Vulkan image format must not be UNDEFINED")]
    InvalidImageFormat,
    /// Image mip-level or array-layer counts are invalid.
    #[error("Vulkan image mip-level and array-layer counts must be valid")]
    InvalidImageSubresourceCount,
    /// Image usage must include at least one usage flag.
    #[error("Vulkan image usage must not be empty")]
    InvalidImageUsage,
    /// Image sample count must contain exactly one supported sample-count bit.
    #[error("Vulkan image sample count must select exactly one sample count")]
    InvalidImageSampleCount,
    /// Image tiling must be linear or optimal.
    #[error("Vulkan image tiling must be LINEAR or OPTIMAL")]
    InvalidImageTiling,
    /// Image type is not a supported Vulkan dimensionality.
    #[error("Vulkan image type is invalid")]
    InvalidImageType,
    /// Requested mip count exceeds the maximum for the image extent.
    #[error("Vulkan image mip-level count exceeds the extent's maximum")]
    InvalidImageMipLevels,
    /// No compatible memory type is available for the image.
    #[error("no compatible Vulkan memory type is available for the image")]
    NoCompatibleImageMemoryType,
    /// Image-view subresource ranges are empty, overflowed, or out of bounds.
    #[error("Vulkan image-view subresource range is invalid")]
    InvalidImageViewRange,
    /// Image and image-view dimensionalities or cube requirements do not match.
    #[error("Vulkan image-view type is incompatible with the image")]
    IncompatibleImageViewType,
    /// Sampler enum values are not recognized by the supported Vulkan core API.
    #[error("Vulkan sampler contains an unsupported enum value")]
    InvalidSamplerEnum,
    /// Sampler LOD bounds are non-finite or inverted.
    #[error("Vulkan sampler LOD range must be finite and ordered")]
    InvalidSamplerLodRange,
    /// Sampler LOD bias is non-finite or exceeds the physical-device limit.
    #[error("Vulkan sampler mip LOD bias exceeds the device limit or is not finite")]
    InvalidSamplerLodBias,
    /// Requested sampler anisotropy is outside the physical-device limit.
    #[error("Vulkan sampler anisotropy must be finite and within the device limit")]
    InvalidSamplerAnisotropy,
    /// Sampler anisotropy was requested but the logical-device feature is unavailable.
    #[error("Vulkan sampler anisotropy feature is not enabled on this device")]
    SamplerAnisotropyUnavailable,
    /// Descriptor-set layout bindings are invalid or duplicated.
    #[error("Vulkan descriptor-set layout contains an invalid or duplicate binding")]
    InvalidDescriptorSetLayoutBinding,
    /// Descriptor-pool capacities are empty, zero, or duplicated.
    #[error("Vulkan descriptor-pool configuration is invalid")]
    InvalidDescriptorPoolConfig,
    /// Descriptor pool and set layout belong to different logical devices.
    #[error("Vulkan descriptor pool and set layout belong to different devices")]
    DescriptorSetDeviceMismatch,
    /// A pipeline layout references a descriptor-set layout from another device.
    #[error("Vulkan pipeline layout and descriptor-set layouts belong to different devices")]
    PipelineLayoutDeviceMismatch,
    /// A push-constant range has an invalid stage mask, alignment, or size.
    #[error("Vulkan push-constant range has invalid stages, alignment, or size")]
    InvalidPushConstantRange,
    /// Push-constant ranges overlap for at least one common shader stage.
    #[error("Vulkan push-constant ranges overlap for a shared shader stage")]
    OverlappingPushConstantRanges,
    /// A descriptor update repeats the same binding and array element.
    #[error("Vulkan descriptor update contains a duplicate binding array element")]
    DuplicateDescriptorWrite,
    /// A descriptor update references a binding absent from its layout.
    #[error("Vulkan descriptor update references an unknown binding")]
    DescriptorBindingNotFound,
    /// A descriptor update references an array element outside its binding.
    #[error("Vulkan descriptor array element is outside the binding count")]
    DescriptorArrayElementOutOfBounds,
    /// The descriptor binding is not a supported uniform/storage buffer type.
    #[error("Vulkan descriptor binding is not a supported buffer descriptor type")]
    UnsupportedBufferDescriptorType,
    /// The buffer does not have the usage required by its descriptor type.
    #[error("Vulkan buffer usage does not match its descriptor type")]
    BufferDescriptorUsageMismatch,
    /// The descriptor buffer offset or range violates device limits or buffer bounds.
    #[error("Vulkan descriptor buffer offset or range is invalid")]
    InvalidBufferDescriptorRange,
    /// A descriptor update does not populate every layout slot.
    #[error("Vulkan descriptor update must populate every binding array element")]
    IncompleteDescriptorWrite,
    /// The descriptor type is not supported by the current descriptor update API.
    #[error("Vulkan descriptor type is not supported by the current update API")]
    UnsupportedDescriptorType,
    /// A combined image-sampler descriptor requires a sampler resource.
    #[error("combined image-sampler descriptor requires a sampler")]
    DescriptorSamplerRequired,
    /// Image usage or layout does not match the descriptor declaration.
    #[error("Vulkan image usage or layout is incompatible with its descriptor type")]
    InvalidImageDescriptor,
    /// The physical device does not support the image format feature required by the descriptor.
    #[error("Vulkan image format does not support the descriptor's sampling or storage operation")]
    UnsupportedImageFormatFeature,
    /// GLSL source could not be parsed into a shader module.
    #[error("GLSL shader compilation failed: {0}")]
    ShaderCompilation(String),
    /// Shader uses a descriptor or resource form unsupported by renderer reflection.
    #[error("shader uses an unsupported descriptor resource or descriptor array form")]
    UnsupportedShaderResource,
    /// Shader declares an invalid or unsupported push-constant block.
    #[error("shader push-constant declaration is invalid or unsupported")]
    InvalidShaderPushConstants,
    /// SPIR-V binary header or instruction stream is malformed.
    #[error("SPIR-V binary is malformed or structurally invalid")]
    InvalidSpirv,
    /// A Vulkan shader module was created from a different logical device.
    #[error("Vulkan shader module belongs to a different logical device")]
    ShaderModuleDeviceMismatch,
    /// A shader stage is incompatible with the requested pipeline type.
    #[error("Vulkan shader stage is incompatible with the requested pipeline")]
    PipelineShaderStageMismatch,
    /// Reflected shader descriptors or push constants are not covered by the pipeline layout.
    #[error("Vulkan pipeline layout does not match the compiled shader interface")]
    ShaderInterfaceMismatch,
    /// Render-pass attachment or subpass references are invalid.
    #[error("Vulkan render-pass attachment, subpass, or dependency configuration is invalid")]
    InvalidRenderPassConfig,
    /// Framebuffer extent, layer count, attachments, or compatibility is invalid.
    #[error("Vulkan framebuffer extent, layers, or attachment configuration is invalid")]
    InvalidFramebufferConfig,
    /// A framebuffer attachment belongs to a different logical device.
    #[error("Vulkan framebuffer attachments belong to a different logical device")]
    FramebufferDeviceMismatch,
    /// Graphics pipeline fixed-function state or render-pass compatibility is invalid.
    #[error("Vulkan graphics pipeline fixed-function configuration is invalid")]
    InvalidGraphicsPipelineConfig,
    /// Command-pool flags or selected queue family are invalid.
    #[error("Vulkan command-pool configuration or queue family is invalid")]
    InvalidCommandPoolConfig,
    /// Command-buffer recording, render-pass, or pipeline state is invalid.
    #[error("Vulkan command-buffer operation is invalid for its current state")]
    InvalidCommandBufferState,
    /// Queue submission parameters or command-buffer state are invalid.
    #[error("Vulkan queue submission parameters or command-buffer state are invalid")]
    InvalidQueueSubmission,
    /// Dynamic descriptor offsets violate count, alignment, or buffer bounds.
    #[error("Vulkan dynamic descriptor offsets are invalid")]
    InvalidDynamicDescriptorOffset,
    /// Push-constant write is not covered by the pipeline layout.
    #[error("Vulkan push-constant write is invalid or not covered by the pipeline layout")]
    InvalidPushConstantWrite,
    /// Host synchronization for queue submission was poisoned.
    #[error("Vulkan queue submission synchronization lock was poisoned")]
    QueueSynchronizationPoisoned,
    /// A shader resource is missing an explicit descriptor set or binding.
    #[error("shader descriptor resources must declare both set and binding")]
    MissingShaderDescriptorBinding,
    /// A shader declares the same descriptor set and binding more than once.
    #[error("shader contains duplicate descriptor set and binding declarations")]
    DuplicateShaderDescriptorBinding,
}

fn api_error(operation: &'static str, result: vk::Result) -> VulkanError {
    VulkanError::Api { operation, result }
}

fn has_extension(properties: &[vk::ExtensionProperties], name: &CStr) -> bool {
    properties.iter().any(|property| {
        // SAFETY: Vulkan returns fixed-size extension names terminated by NUL.
        unsafe { CStr::from_ptr(property.extension_name.as_ptr()) == name }
    })
}

fn has_layer(properties: &[vk::LayerProperties], name: &CStr) -> bool {
    properties.iter().any(|property| {
        // SAFETY: Vulkan returns fixed-size layer names terminated by NUL.
        unsafe { CStr::from_ptr(property.layer_name.as_ptr()) == name }
    })
}

/// Owns the Vulkan loader, instance and optional debug messenger.
pub struct VulkanInstance {
    _entry: Entry,
    handle: ash::Instance,
    debug_messenger: Option<DebugMessenger>,
}

impl VulkanInstance {
    /// Loads Vulkan and creates an instance after checking requested capabilities.
    pub fn new(config: &VulkanConfig) -> Result<Self, VulkanError> {
        Self::new_with_extensions(config, &[])
    }

    fn new_with_extensions(
        config: &VulkanConfig,
        required_extensions: &[CString],
    ) -> Result<Self, VulkanError> {
        let application_name = CString::new(config.application_name.as_str())?;
        let engine_name = CString::new("Z Ideal")?;

        // SAFETY: The dynamically loaded Entry is retained until after the instance is destroyed.
        let entry = unsafe { Entry::load()? };
        // SAFETY: These calls query loader-owned metadata and create no Vulkan objects.
        let available_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .map_err(|error| api_error("enumerate instance extensions", error))?
        };
        // SAFETY: This call queries loader-owned layer metadata.
        let available_layers = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .map_err(|error| api_error("enumerate instance layers", error))?
        };

        let mut enabled_extensions = Vec::<CString>::new();
        for required in required_extensions {
            if !has_extension(&available_extensions, required.as_c_str()) {
                return Err(VulkanError::MissingInstanceExtension(
                    required.to_string_lossy().into_owned(),
                ));
            }
            if !enabled_extensions
                .iter()
                .any(|enabled| enabled.as_c_str() == required.as_c_str())
            {
                enabled_extensions.push(required.clone());
            }
        }
        let mut flags = vk::InstanceCreateFlags::empty();

        if config.enable_validation {
            if !has_layer(&available_layers, VALIDATION_LAYER) {
                return Err(VulkanError::MissingValidationLayer(
                    VALIDATION_LAYER.to_string_lossy().into_owned(),
                ));
            }
            if !has_extension(&available_extensions, DEBUG_UTILS_EXTENSION) {
                return Err(VulkanError::MissingInstanceExtension(
                    DEBUG_UTILS_EXTENSION.to_string_lossy().into_owned(),
                ));
            }
            enabled_extensions.push(DEBUG_UTILS_EXTENSION.to_owned());
        }

        if has_extension(&available_extensions, PORTABILITY_ENUMERATION_EXTENSION) {
            enabled_extensions.push(PORTABILITY_ENUMERATION_EXTENSION.to_owned());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        #[cfg(target_os = "macos")]
        if !enabled_extensions
            .iter()
            .any(|name| name.as_c_str() == PORTABILITY_ENUMERATION_EXTENSION)
        {
            return Err(VulkanError::MissingInstanceExtension(
                PORTABILITY_ENUMERATION_EXTENSION
                    .to_string_lossy()
                    .into_owned(),
            ));
        }

        let extension_pointers: Vec<*const i8> = enabled_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let layer_pointers: Vec<*const i8> = if config.enable_validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };
        let application_info = vk::ApplicationInfo {
            p_application_name: application_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::API_VERSION_1_0,
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &application_info,
            enabled_extension_count: extension_pointers.len() as u32,
            pp_enabled_extension_names: extension_pointers.as_ptr(),
            enabled_layer_count: layer_pointers.len() as u32,
            pp_enabled_layer_names: layer_pointers.as_ptr(),
            flags,
            ..Default::default()
        };

        // SAFETY: All create-info pointers refer to live, NUL-terminated values for this call.
        let handle = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|error| api_error("create Vulkan instance", error))?
        };

        let debug_messenger = if config.enable_validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &handle);
            let info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(validation_callback),
                ..Default::default()
            };
            // SAFETY: The callback is static; the instance and loader outlive this messenger.
            match unsafe { loader.create_debug_utils_messenger(&info, None) } {
                Ok(messenger) => Some(DebugMessenger { loader, messenger }),
                Err(error) => {
                    // SAFETY: No child objects were created after the instance.
                    unsafe { handle.destroy_instance(None) };
                    return Err(api_error("create Vulkan debug messenger", error));
                }
            }
        } else {
            None
        };

        Ok(Self {
            _entry: entry,
            handle,
            debug_messenger,
        })
    }

    /// Returns the raw Vulkan instance for renderer-internal integration.
    #[must_use]
    pub fn handle(&self) -> &ash::Instance {
        &self.handle
    }

    /// Returns whether validation diagnostics are enabled.
    #[must_use]
    pub fn validation_enabled(&self) -> bool {
        self.debug_messenger.is_some()
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        // SAFETY: The messenger is destroyed before its parent instance.
        drop(self.debug_messenger.take());
        // SAFETY: Devices retain an Arc to this instance, so this runs only after they are dropped.
        unsafe { self.handle.destroy_instance(None) };
    }
}

struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the messenger created by this loader.
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

unsafe extern "system" fn validation_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !callback_data.is_null() {
        // SAFETY: Vulkan guarantees callback_data and p_message are valid during this callback.
        let message = unsafe { CStr::from_ptr((*callback_data).p_message) };
        tracing::warn!(
            target: "z_ideal::vulkan_validation",
            ?severity,
            message = %message.to_string_lossy(),
            "Vulkan validation message"
        );
    }
    vk::FALSE
}

/// Stable summary of a selected physical device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalDeviceInfo {
    /// Device name.
    pub name: String,
    /// Device category.
    pub device_type: vk::PhysicalDeviceType,
    /// Vendor identifier.
    pub vendor_id: u32,
    /// Device identifier.
    pub device_id: u32,
    /// Supported Vulkan API version.
    pub api_version: u32,
}

/// Owns a logical device and its selected graphics queue.
pub struct VulkanDevice {
    _instance: Arc<VulkanInstance>,
    handle: ash::Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    graphics_queue_family: u32,
    queue_submit_lock: Mutex<()>,
    present_queue: vk::Queue,
    present_queue_family: u32,
    presentation_enabled: bool,
    sampler_anisotropy_enabled: bool,
    max_sampler_anisotropy: f32,
    max_sampler_lod_bias: f32,
    max_push_constants_size: u32,
    min_uniform_buffer_offset_alignment: vk::DeviceSize,
    min_storage_buffer_offset_alignment: vk::DeviceSize,
    max_uniform_buffer_range: vk::DeviceSize,
    max_storage_buffer_range: vk::DeviceSize,
    info: PhysicalDeviceInfo,
}

impl VulkanDevice {
    /// Selects a graphics-capable device and creates its logical device.
    pub fn new(instance: Arc<VulkanInstance>) -> Result<Self, VulkanError> {
        Self::new_internal(instance, None)
    }

    /// Selects a device with graphics and presentation queues for a window surface.
    pub fn new_for_surface(
        instance: Arc<VulkanInstance>,
        surface: &VulkanSurface,
    ) -> Result<Self, VulkanError> {
        if !Arc::ptr_eq(&instance, &surface.instance) {
            return Err(VulkanError::Surface(
                "surface and device must belong to the same Vulkan instance".to_owned(),
            ));
        }
        Self::new_internal(instance, Some(surface))
    }

    fn new_internal(
        instance: Arc<VulkanInstance>,
        surface: Option<&VulkanSurface>,
    ) -> Result<Self, VulkanError> {
        // SAFETY: The instance is live and retained by this device.
        let physical_devices = unsafe {
            instance
                .handle
                .enumerate_physical_devices()
                .map_err(|error| api_error("enumerate physical devices", error))?
        };
        let mut candidates = Vec::new();

        for physical_device in physical_devices {
            // SAFETY: The handle was returned by this live instance.
            let properties = unsafe {
                instance
                    .handle
                    .get_physical_device_properties(physical_device)
            };
            // SAFETY: The handle was returned by this live instance.
            let queue_families = unsafe {
                instance
                    .handle
                    .get_physical_device_queue_family_properties(physical_device)
            };
            let Some((graphics_index, _)) =
                queue_families.iter().enumerate().find(|(_, family)| {
                    family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
            else {
                continue;
            };
            let graphics_queue_family = graphics_index as u32;

            let present_queue_family = if let Some(surface) = surface {
                let mut present_index = None;
                for (index, family) in queue_families.iter().enumerate() {
                    if family.queue_count == 0 {
                        continue;
                    }
                    if surface.supports_present(physical_device, index as u32)? {
                        present_index = Some(index as u32);
                        break;
                    }
                }
                let Some(present_index) = present_index else {
                    continue;
                };

                // Presentation-capable devices must expose the swapchain extension.
                // SAFETY: physical_device belongs to the retained instance.
                let extensions = unsafe {
                    instance
                        .handle
                        .enumerate_device_extension_properties(physical_device)
                        .map_err(|error| api_error("enumerate device extensions", error))?
                };
                if !has_extension(&extensions, SWAPCHAIN_EXTENSION) {
                    continue;
                }
                present_index
            } else {
                graphics_queue_family
            };

            // SAFETY: Vulkan provides a NUL-terminated device_name array.
            let name = unsafe {
                CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            candidates.push((
                physical_device,
                graphics_queue_family,
                present_queue_family,
                PhysicalDeviceInfo {
                    name,
                    device_type: properties.device_type,
                    vendor_id: properties.vendor_id,
                    device_id: properties.device_id,
                    api_version: properties.api_version,
                },
            ));
        }

        // Prefer discrete, then integrated, virtual, CPU and other devices.
        // Ties are resolved by stable reported identity fields.
        candidates.sort_by(|left, right| {
            device_rank(right.3.device_type)
                .cmp(&device_rank(left.3.device_type))
                .then_with(|| left.3.name.cmp(&right.3.name))
                .then_with(|| left.3.vendor_id.cmp(&right.3.vendor_id))
                .then_with(|| left.3.device_id.cmp(&right.3.device_id))
        });
        let (physical_device, graphics_queue_family, present_queue_family, info) =
            candidates.into_iter().next().ok_or_else(|| {
                if surface.is_some() {
                    VulkanError::NoPresentDevice
                } else {
                    VulkanError::NoGraphicsDevice
                }
            })?;

        // SAFETY: physical_device belongs to the retained instance.
        let available_extensions = unsafe {
            instance
                .handle
                .enumerate_device_extension_properties(physical_device)
                .map_err(|error| api_error("enumerate device extensions", error))?
        };
        let mut enabled_extensions = Vec::<CString>::new();
        if surface.is_some() {
            enabled_extensions.push(SWAPCHAIN_EXTENSION.to_owned());
        }
        if has_extension(&available_extensions, PORTABILITY_SUBSET_EXTENSION) {
            enabled_extensions.push(PORTABILITY_SUBSET_EXTENSION.to_owned());
        }
        let extension_pointers: Vec<*const i8> = enabled_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        // Enable sampler anisotropy only when the selected physical device supports it.
        // Other optional features remain disabled until explicitly required by the renderer.
        // SAFETY: The selected physical device belongs to the retained instance.
        let supported_features = unsafe {
            instance
                .handle
                .get_physical_device_features(physical_device)
        };
        let sampler_anisotropy_enabled = supported_features.sampler_anisotropy == vk::TRUE;
        let enabled_features = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: if sampler_anisotropy_enabled {
                vk::TRUE
            } else {
                vk::FALSE
            },
            ..Default::default()
        };
        // SAFETY: The selected physical device belongs to the retained instance.
        let physical_device_properties = unsafe {
            instance
                .handle
                .get_physical_device_properties(physical_device)
        };
        let max_sampler_anisotropy = physical_device_properties.limits.max_sampler_anisotropy;
        let max_sampler_lod_bias = physical_device_properties.limits.max_sampler_lod_bias;
        let max_push_constants_size = physical_device_properties.limits.max_push_constants_size;
        let min_uniform_buffer_offset_alignment = physical_device_properties
            .limits
            .min_uniform_buffer_offset_alignment
            .max(1);
        let min_storage_buffer_offset_alignment = physical_device_properties
            .limits
            .min_storage_buffer_offset_alignment
            .max(1);
        let max_uniform_buffer_range =
            u64::from(physical_device_properties.limits.max_uniform_buffer_range);
        let max_storage_buffer_range =
            u64::from(physical_device_properties.limits.max_storage_buffer_range);
        let priorities = [1.0_f32];
        let mut queue_family_indices = vec![graphics_queue_family];
        if present_queue_family != graphics_queue_family {
            queue_family_indices.push(present_queue_family);
        }
        let queue_info: Vec<_> = queue_family_indices
            .iter()
            .map(|&queue_family_index| vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count: 1,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            })
            .collect();
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_info.len() as u32,
            p_queue_create_infos: queue_info.as_ptr(),
            enabled_extension_count: extension_pointers.len() as u32,
            pp_enabled_extension_names: extension_pointers.as_ptr(),
            p_enabled_features: &enabled_features,
            ..Default::default()
        };

        // SAFETY: Queue and extension arrays remain alive for the call and the
        // selected physical device belongs to the retained instance.
        let handle = unsafe {
            instance
                .handle
                .create_device(physical_device, &device_info, None)
                .map_err(|error| api_error("create Vulkan logical device", error))?
        };
        // SAFETY: This queue family and queue index were included in device creation.
        let graphics_queue = unsafe { handle.get_device_queue(graphics_queue_family, 0) };
        // SAFETY: This queue family was included in logical-device creation.
        let present_queue = unsafe { handle.get_device_queue(present_queue_family, 0) };

        Ok(Self {
            _instance: instance,
            handle,
            physical_device,
            graphics_queue,
            graphics_queue_family,
            queue_submit_lock: Mutex::new(()),
            present_queue,
            present_queue_family,
            presentation_enabled: surface.is_some(),
            sampler_anisotropy_enabled,
            max_sampler_anisotropy,
            max_sampler_lod_bias,
            max_push_constants_size,
            min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment,
            max_uniform_buffer_range,
            max_storage_buffer_range,
            info,
        })
    }

    /// Returns the raw logical device for renderer-internal integration.
    #[must_use]
    pub fn handle(&self) -> &ash::Device {
        &self.handle
    }

    /// Returns the device limit for push-constant storage, in bytes.
    #[must_use]
    pub const fn max_push_constants_size(&self) -> u32 {
        self.max_push_constants_size
    }

    /// Returns the selected physical device handle.
    #[must_use]
    pub const fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns the graphics queue.
    #[must_use]
    pub const fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Returns the graphics queue-family index.
    #[must_use]
    pub const fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    /// Whether this device was created with presentation support and VK_KHR_swapchain.
    #[must_use]
    pub const fn presentation_enabled(&self) -> bool {
        self.presentation_enabled
    }

    /// Returns whether sampler anisotropy was supported and enabled on this device.
    #[must_use]
    pub const fn sampler_anisotropy_enabled(&self) -> bool {
        self.sampler_anisotropy_enabled
    }

    /// Returns the maximum supported sampler anisotropy for this device.
    #[must_use]
    pub const fn max_sampler_anisotropy(&self) -> f32 {
        self.max_sampler_anisotropy
    }

    /// Returns the maximum absolute sampler mip LOD bias for this device.
    #[must_use]
    pub const fn max_sampler_lod_bias(&self) -> f32 {
        self.max_sampler_lod_bias
    }

    /// Returns the presentation queue.
    #[must_use]
    pub const fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }

    /// Returns the presentation queue-family index.
    #[must_use]
    pub const fn present_queue_family(&self) -> u32 {
        self.present_queue_family
    }

    /// Waits for all work submitted to this logical device to complete.
    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        let _queue_guard = self
            .queue_submit_lock
            .lock()
            .map_err(|_| VulkanError::QueueSynchronizationPoisoned)?;
        // SAFETY: The logical device is live and queue host access is serialized by the lock.
        unsafe {
            self.handle
                .device_wait_idle()
                .map_err(|error| api_error("wait for Vulkan device idle", error))
        }
    }

    /// Returns selected physical-device metadata.
    #[must_use]
    pub const fn info(&self) -> &PhysicalDeviceInfo {
        &self.info
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the logical device. Waiting idle ensures
        // no submitted work can still reference resources owned by this device.
        if let Err(error) = unsafe { self.handle.device_wait_idle() } {
            tracing::warn!(
                ?error,
                "Vulkan device was not idle during device destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the logical device.
        unsafe { self.handle.destroy_device(None) };
    }
}

/// Owns a Vulkan instance and logical device for one renderer session.
pub struct VulkanContext {
    instance: Arc<VulkanInstance>,
    surface: Option<VulkanSurface>,
    device: VulkanDevice,
}

impl VulkanContext {
    /// Creates an instance, selects a physical device and creates its logical device.
    pub fn new(config: &VulkanConfig) -> Result<Self, VulkanError> {
        let instance = Arc::new(VulkanInstance::new(config)?);
        let device = VulkanDevice::new(Arc::clone(&instance))?;
        Ok(Self {
            instance,
            surface: None,
            device,
        })
    }

    /// Creates an instance, window surface, presentation-capable device and queues.
    ///
    /// The caller must keep window alive until this context and any future swapchains
    /// created from its surface have been destroyed.
    pub fn new_for_window(config: &VulkanConfig, window: &Window) -> Result<Self, VulkanError> {
        let required_extensions = surface::required_instance_extensions(window)?;
        let instance = Arc::new(VulkanInstance::new_with_extensions(
            config,
            &required_extensions,
        )?);
        let surface = VulkanSurface::new(Arc::clone(&instance), window)?;
        let device = VulkanDevice::new_for_surface(Arc::clone(&instance), &surface)?;
        Ok(Self {
            instance,
            surface: Some(surface),
            device,
        })
    }

    /// Returns the optional presentation surface.
    #[must_use]
    pub fn surface(&self) -> Option<&VulkanSurface> {
        self.surface.as_ref()
    }

    /// Returns the instance owner.
    #[must_use]
    pub fn instance(&self) -> &Arc<VulkanInstance> {
        &self.instance
    }

    /// Returns the logical-device owner.
    #[must_use]
    pub const fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Creates a swapchain for the attached window surface.
    pub fn create_swapchain(
        &self,
        window_extent: [u32; 2],
        config: SwapchainConfig,
    ) -> Result<VulkanSwapchain<'_>, VulkanError> {
        let surface = self.surface.as_ref().ok_or(VulkanError::MissingSurface)?;
        VulkanSwapchain::new(&self.device, surface, window_extent, config)
    }
}

const fn device_rank(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 5,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 4,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
        vk::PhysicalDeviceType::CPU => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::{CRATE_NAME, VulkanConfig, device_rank};
    use ash::vk;

    #[test]
    fn crate_name_is_stable() {
        assert_eq!(CRATE_NAME, "z-ideal-renderer");
    }

    #[test]
    fn validation_default_matches_build_profile() {
        assert_eq!(
            VulkanConfig::default().enable_validation,
            cfg!(debug_assertions)
        );
    }

    #[test]
    fn physical_device_preference_is_ordered() {
        assert!(
            device_rank(vk::PhysicalDeviceType::DISCRETE_GPU)
                > device_rank(vk::PhysicalDeviceType::INTEGRATED_GPU)
        );
        assert!(
            device_rank(vk::PhysicalDeviceType::INTEGRATED_GPU)
                > device_rank(vk::PhysicalDeviceType::CPU)
        );
    }
}
