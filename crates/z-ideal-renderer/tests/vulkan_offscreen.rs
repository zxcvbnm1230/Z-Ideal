use std::sync::Arc;

use ash::vk;
use z_ideal_renderer::{
    BufferBarrier, BufferConfig, BufferMemoryLocation, BufferOwnershipTransfer, CommandPoolConfig,
    FenceWaitResult, GraphicsPipelineConfig, ImageBarrier, ImageConfig, ImageViewConfig,
    RenderPassAttachmentConfig, RenderPassAttachmentReference, RenderPassConfig,
    RenderPassDependency, RenderPassSubpassConfig, ShaderCompileConfig, ShaderStage, VulkanBuffer,
    VulkanCommandPool, VulkanConfig, VulkanDevice, VulkanFramebuffer, VulkanGraphicsPipeline,
    VulkanImage, VulkanInstance, VulkanPipelineLayout, VulkanQueue, VulkanRenderPass,
    VulkanSemaphore, VulkanShaderModule, compile_glsl,
};

/// Runs on the dedicated Mesa lavapipe + Khronos validation CI job.
#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run in the dedicated Vulkan CI job"]
fn offscreen_triangle_draw_copy_readback_is_deterministic_and_validation_clean() {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const BYTE_COUNT: usize = (WIDTH * HEIGHT * 4) as usize;

    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal Vulkan offscreen triangle validation".to_owned(),
            enable_validation: true,
        })
        .expect("create Vulkan instance with validation enabled"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan graphics device");
    let image_config = ImageConfig {
        image_type: vk::ImageType::TYPE_2D,
        extent: vk::Extent3D {
            width: WIDTH,
            height: HEIGHT,
            depth: 1,
        },
        format: vk::Format::R8G8B8A8_UNORM,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        flags: vk::ImageCreateFlags::empty(),
    };
    let image = VulkanImage::new(&device, image_config).expect("create offscreen color image");
    let image_view = image
        .create_view(ImageViewConfig::default())
        .expect("create offscreen image view");
    let render_pass = VulkanRenderPass::new(
        &device,
        RenderPassConfig {
            attachments: vec![RenderPassAttachmentConfig {
                format: image_config.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            }],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![RenderPassAttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![RenderPassDependency {
                source_subpass: 0,
                destination_subpass: vk::SUBPASS_EXTERNAL,
                source_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                destination_stage_mask: vk::PipelineStageFlags::TRANSFER,
                source_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                destination_access_mask: vk::AccessFlags::TRANSFER_READ,
                flags: vk::DependencyFlags::BY_REGION,
            }],
        },
    )
    .expect("create offscreen render pass");
    let framebuffer = VulkanFramebuffer::new(
        &render_pass,
        &[&image_view],
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
        1,
    )
    .expect("create offscreen framebuffer");

    let vertex_values = [
        -0.8_f32, -0.8, 1.0, 0.0, 0.0, 0.8, -0.8, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 1.0,
    ];
    let vertex_bytes: Vec<u8> = vertex_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let mut vertex_buffer = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: vertex_bytes.len() as u64,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            memory: BufferMemoryLocation::Upload,
        },
    )
    .expect("create host-visible vertex buffer");
    vertex_buffer
        .write(0, &vertex_bytes)
        .expect("upload triangle vertices");

    let vertex_shader = VulkanShaderModule::new(
        &device,
        compile_glsl(
            "#version 450\nlayout(location=0) in vec2 position;\nlayout(location=1) in vec3 vertex_color;\nlayout(location=0) out vec3 color;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); color = vertex_color; }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        )
        .expect("compile triangle vertex shader"),
    )
    .expect("create triangle vertex shader module");
    let fragment_shader = VulkanShaderModule::new(
        &device,
        compile_glsl(
            "#version 450\nlayout(location=0) in vec3 color;\nlayout(location=0) out vec4 output_color;\nvoid main() { output_color = vec4(color, 1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Fragment),
        )
        .expect("compile triangle fragment shader"),
    )
    .expect("create triangle fragment shader module");
    let pipeline_layout = VulkanPipelineLayout::new(&device, &[], &[])
        .expect("create empty graphics pipeline layout");
    let graphics_pipeline = VulkanGraphicsPipeline::new(
        &pipeline_layout,
        &vertex_shader,
        &fragment_shader,
        &render_pass,
        &GraphicsPipelineConfig {
            vertex_bindings: vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: 20,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            vertex_attributes: vec![
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 8,
                },
            ],
            ..GraphicsPipelineConfig::default()
        },
    )
    .expect("create triangle graphics pipeline");

    let mut readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: BYTE_COUNT as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create host-visible readback buffer");
    let mut second_readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: BYTE_COUNT as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create second host-visible readback buffer sharing the memory pool");

    let transfer_payload: Vec<u8> = (0..64).collect();
    let mut transfer_source = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: transfer_payload.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory: BufferMemoryLocation::Upload,
        },
    )
    .expect("create transfer upload buffer");
    transfer_source
        .write(0, &transfer_payload)
        .expect("upload transfer test payload");
    let transfer_intermediate = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: transfer_payload.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::DeviceLocal,
        },
    )
    .expect("create transfer intermediate buffer");
    let mut transfer_destination = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: transfer_payload.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create transfer readback buffer");
    let transfer_queue = VulkanQueue::transfer(&device);
    let transfer_pool =
        VulkanCommandPool::new_for_queue(&device, &transfer_queue, CommandPoolConfig::default())
            .expect("create transfer command pool");
    let ownership_transfer_required =
        device.graphics_queue_family() != device.transfer_queue_family();
    let transfer_semaphore = if ownership_transfer_required {
        Some(VulkanSemaphore::new(&device).expect("create queue-transfer semaphore"))
    } else {
        None
    };

    let command_pool = VulkanCommandPool::new(&device, CommandPoolConfig::default())
        .expect("create graphics command pool");
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate primary command buffer");
    let queue = VulkanQueue::graphics(&device);

    let clear = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];
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
        .image_extent(vk::Extent3D {
            width: WIDTH,
            height: HEIGHT,
            depth: 1,
        })];
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin command recording");
    recording
        .begin_render_pass(&render_pass, &framebuffer, &clear)
        .expect("begin offscreen render pass");
    recording
        .bind_graphics_pipeline(&graphics_pipeline)
        .expect("bind triangle graphics pipeline");
    recording
        .set_viewport(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .expect("set offscreen viewport");
    recording
        .set_scissor(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            },
        })
        .expect("set offscreen scissor");
    recording
        .bind_vertex_buffers(0, &[&vertex_buffer], &[0])
        .expect("bind triangle vertex buffer");
    recording.draw(3, 1, 0, 0).expect("draw triangle");
    recording
        .end_render_pass()
        .expect("end offscreen render pass");
    recording
        .copy_image_to_buffer(
            &image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &region,
        )
        .expect("copy rendered image to readback buffer");
    recording
        .copy_image_to_buffer(
            &image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &second_readback,
            &region,
        )
        .expect("copy rendered image to second readback buffer");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[
                BufferBarrier {
                    buffer: &readback,
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::HOST_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                },
                BufferBarrier {
                    buffer: &second_readback,
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::HOST_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                },
            ],
            &[],
        )
        .expect("make transfer writes visible to the host");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[BufferBarrier {
                buffer: &transfer_source,
                source_access: vk::AccessFlags::HOST_WRITE,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            }],
            &[],
        )
        .expect("make uploaded transfer payload visible to the graphics queue");
    recording
        .copy_buffer(
            &transfer_source,
            &transfer_intermediate,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: transfer_payload.len() as u64,
            }],
        )
        .expect("copy transfer payload on the graphics queue");
    if ownership_transfer_required {
        recording
            .release_queue_family_ownership(
                vk::PipelineStageFlags::TRANSFER,
                &[BufferOwnershipTransfer {
                    buffer: &transfer_intermediate,
                    source_queue_family: device.graphics_queue_family(),
                    destination_queue_family: device.transfer_queue_family(),
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::TRANSFER_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }],
                &[],
            )
            .expect("release intermediate buffer to transfer queue family");
    }
    let executable = recording.end().expect("finish command buffer");
    let signal_semaphores: Vec<_> = transfer_semaphore.iter().collect();
    let mut submission = queue
        .submit(executable, &[], &[], &signal_semaphores)
        .expect("submit offscreen render and copy");
    assert_eq!(
        submission
            .wait(u64::MAX)
            .expect("wait for offscreen submission"),
        FenceWaitResult::Signaled
    );
    drop(submission);

    let mut pixels = vec![0_u8; BYTE_COUNT];
    readback
        .read(0, &mut pixels)
        .expect("read back rendered pixels");
    let mut second_pixels = vec![0_u8; BYTE_COUNT];
    second_readback
        .read(0, &mut second_pixels)
        .expect("read back pixels from the second pooled buffer");
    assert_eq!(pixels, second_pixels, "pooled readback buffers disagree");
    let center = &pixels[((HEIGHT / 2 * WIDTH + WIDTH / 2) * 4) as usize..][..4];
    assert!(center[0] > 20, "triangle red channel too small: {center:?}");
    assert!(
        center[1] > 20,
        "triangle green channel too small: {center:?}"
    );
    assert!(
        center[2] > 20,
        "triangle blue channel too small: {center:?}"
    );
    assert_eq!(center[3], 255, "unexpected triangle alpha: {center:?}");
    let corner = &pixels[0..4];
    assert_eq!(
        corner,
        &[0, 0, 0, 255],
        "unexpected clear pixel: {corner:?}"
    );

    let mut transfer_command_buffer = transfer_pool
        .allocate_primary()
        .expect("allocate transfer command buffer");
    let mut transfer_recording = transfer_command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin transfer command recording");
    if ownership_transfer_required {
        transfer_recording
            .acquire_queue_family_ownership(
                vk::PipelineStageFlags::TRANSFER,
                &[BufferOwnershipTransfer {
                    buffer: &transfer_intermediate,
                    source_queue_family: device.graphics_queue_family(),
                    destination_queue_family: device.transfer_queue_family(),
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::TRANSFER_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }],
                &[],
            )
            .expect("acquire intermediate buffer on transfer queue family");
    } else {
        transfer_recording
            .pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[BufferBarrier {
                    buffer: &transfer_intermediate,
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::TRANSFER_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                }],
                &[],
            )
            .expect("make graphics-queue transfer writes visible on the shared family");
    }
    transfer_recording
        .copy_buffer(
            &transfer_intermediate,
            &transfer_destination,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: transfer_payload.len() as u64,
            }],
        )
        .expect("copy payload on transfer queue");
    transfer_recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[BufferBarrier {
                buffer: &transfer_destination,
                source_access: vk::AccessFlags::TRANSFER_WRITE,
                destination_access: vk::AccessFlags::HOST_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            }],
            &[],
        )
        .expect("make transfer queue writes visible to the host");
    let transfer_executable = transfer_recording
        .end()
        .expect("finish transfer command buffer");
    let wait_semaphores: Vec<_> = transfer_semaphore.iter().collect();
    let wait_stage_masks = if ownership_transfer_required {
        vec![vk::PipelineStageFlags::TRANSFER]
    } else {
        Vec::new()
    };
    let mut transfer_submission = transfer_queue
        .submit(
            transfer_executable,
            &wait_semaphores,
            &wait_stage_masks,
            &[],
        )
        .expect("submit transfer queue copy");
    assert_eq!(
        transfer_submission
            .wait(u64::MAX)
            .expect("wait for transfer queue submission"),
        FenceWaitResult::Signaled
    );
    drop(transfer_submission);
    let mut transfer_pixels = vec![0_u8; transfer_payload.len()];
    transfer_destination
        .read(0, &mut transfer_pixels)
        .expect("read transfer queue result");
    assert_eq!(
        transfer_pixels, transfer_payload,
        "transfer queue copy changed the payload"
    );

    drop(graphics_pipeline);
    drop(pipeline_layout);
    drop(fragment_shader);
    drop(vertex_shader);
    drop(framebuffer);
    drop(render_pass);
    drop(image_view);
    drop(transfer_destination);
    drop(transfer_intermediate);
    drop(transfer_source);
    drop(transfer_pool);
    drop(transfer_semaphore);
    drop(second_readback);
    drop(readback);
    drop(vertex_buffer);
    drop(image);
    drop(command_pool);
    drop(device);

    let validation_errors: Vec<_> = instance
        .validation_messages()
        .expect("read validation diagnostics")
        .into_iter()
        .filter(|message| {
            message
                .severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        })
        .collect();
    assert!(
        validation_errors.is_empty(),
        "Vulkan validation reported errors: {validation_errors:#?}"
    );
}

fn round_trip_buffer_image_copy(
    device: &VulkanDevice,
    queue: &VulkanQueue<'_>,
    command_pool: &VulkanCommandPool<'_>,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    payload: &[u8],
) {
    let extent = vk::Extent3D {
        width: 4,
        height: 4,
        depth: 1,
    };
    let mut upload = VulkanBuffer::new(
        device,
        BufferConfig {
            size: payload.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory: BufferMemoryLocation::Upload,
        },
    )
    .expect("create format-test upload buffer");
    upload.write(0, payload).expect("write format-test payload");
    let mut readback = VulkanBuffer::new(
        device,
        BufferConfig {
            size: payload.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create format-test readback buffer");
    let image = VulkanImage::new(
        device,
        ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent,
            format,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            flags: vk::ImageCreateFlags::empty(),
        },
    )
    .expect("create format-test image");

    let region = [vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: extent,
    }];
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate format-test command buffer");
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin format-test recording");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TOP_OF_PIPE | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[BufferBarrier {
                buffer: &upload,
                source_access: vk::AccessFlags::HOST_WRITE,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            }],
            &[ImageBarrier {
                image: &image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                source_access: vk::AccessFlags::empty(),
                destination_access: vk::AccessFlags::TRANSFER_WRITE,
                aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )
        .expect("transition format-test image for upload");
    recording
        .copy_buffer_to_image(
            &upload,
            &image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        )
        .expect("upload format-test image data");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: &image,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                source_access: vk::AccessFlags::TRANSFER_WRITE,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )
        .expect("transition format-test image for readback");
    recording
        .copy_image_to_buffer(
            &image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &region,
        )
        .expect("copy format-test image to readback buffer");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[BufferBarrier {
                buffer: &readback,
                source_access: vk::AccessFlags::TRANSFER_WRITE,
                destination_access: vk::AccessFlags::HOST_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            }],
            &[],
        )
        .expect("make format-test readback visible to host");
    let executable = recording.end().expect("finish format-test command buffer");
    let mut submission = queue
        .submit(executable, &[], &[], &[])
        .expect("submit format-test copy");
    assert_eq!(
        submission
            .wait(u64::MAX)
            .expect("wait for format-test copy"),
        FenceWaitResult::Signaled
    );
    drop(submission);

    let mut actual = vec![0_u8; payload.len()];
    readback
        .read(0, &mut actual)
        .expect("read format-test bytes");
    assert_eq!(
        actual, payload,
        "format {format:?} round trip changed bytes"
    );
}

/// Exercises compressed and depth buffer-image copies on formats supported by lavapipe.
#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run in the dedicated Vulkan CI job"]
fn compressed_and_depth_buffer_image_copies_round_trip_without_validation_errors() {
    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal Vulkan format transfer validation".to_owned(),
            enable_validation: true,
        })
        .expect("create Vulkan instance with validation enabled"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan graphics device");
    let queue = VulkanQueue::graphics(&device);
    let command_pool = VulkanCommandPool::new(&device, CommandPoolConfig::default())
        .expect("create format-test command pool");
    let required_usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    let cases = [
        (
            vk::Format::BC1_RGBA_UNORM_BLOCK,
            vk::ImageAspectFlags::COLOR,
            vec![0x13, 0x37, 0x42, 0xA5, 0x5A, 0xC3, 0x7E, 0x91],
        ),
        (
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
            (0..16)
                .flat_map(|index| (index as f32 / 16.0).to_ne_bytes())
                .collect(),
        ),
    ];
    let mut exercised = 0;
    for (format, aspect, payload) in cases {
        // SAFETY: The physical device belongs to this live Vulkan instance.
        let support = unsafe {
            instance
                .handle()
                .get_physical_device_image_format_properties(
                    device.physical_device(),
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    required_usage,
                    vk::ImageCreateFlags::empty(),
                )
        };
        match support {
            Ok(_) => {}
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
                eprintln!("Skipping unsupported optimal transfer format {format:?}");
                continue;
            }
            Err(error) => panic!("query image format support for {format:?}: {error:?}"),
        }
        round_trip_buffer_image_copy(&device, &queue, &command_pool, format, aspect, &payload);
        exercised += 1;
    }
    assert!(
        exercised > 0,
        "lavapipe exposes no supported compressed/depth transfer format"
    );

    drop(command_pool);
    drop(device);
    let validation_errors: Vec<_> = instance
        .validation_messages()
        .expect("read validation diagnostics")
        .into_iter()
        .filter(|message| {
            message
                .severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        })
        .collect();
    assert!(
        validation_errors.is_empty(),
        "Vulkan validation reported errors: {validation_errors:#?}"
    );
}
