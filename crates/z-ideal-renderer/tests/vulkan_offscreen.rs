use std::sync::Arc;

use ash::vk;
use z_ideal_renderer::{
    BufferBarrier, BufferConfig, BufferMemoryLocation, CommandPoolConfig, FenceWaitResult,
    ImageConfig, ImageViewConfig, RenderPassAttachmentConfig, RenderPassAttachmentReference,
    RenderPassConfig, RenderPassDependency, RenderPassSubpassConfig, VulkanBuffer,
    VulkanCommandPool, VulkanConfig, VulkanDevice, VulkanFramebuffer, VulkanImage, VulkanInstance,
    VulkanQueue, VulkanRenderPass,
};

/// Runs on the dedicated Mesa lavapipe + Khronos validation CI job.
#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run in the dedicated Vulkan CI job"]
fn offscreen_clear_copy_readback_is_deterministic_and_validation_clean() {
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const BYTE_COUNT: usize = (WIDTH * HEIGHT * 4) as usize;

    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal Vulkan offscreen validation".to_owned(),
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
    let mut readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: BYTE_COUNT as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create host-visible readback buffer");
    let command_pool = VulkanCommandPool::new(&device, CommandPoolConfig::default())
        .expect("create graphics command pool");
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate primary command buffer");
    let queue = VulkanQueue::graphics(&device);

    let clear = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.25, 0.5, 0.75, 1.0],
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
        .expect("make transfer writes visible to the host");
    let executable = recording.end().expect("finish command buffer");
    let mut submission = queue
        .submit(executable, &[], &[], &[])
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
    for pixel in pixels.chunks_exact(4) {
        assert!(
            (i16::from(pixel[0]) - 64).abs() <= 1,
            "unexpected red channel: {pixel:?}"
        );
        assert!(
            (i16::from(pixel[1]) - 128).abs() <= 1,
            "unexpected green channel: {pixel:?}"
        );
        assert!(
            (i16::from(pixel[2]) - 191).abs() <= 1,
            "unexpected blue channel: {pixel:?}"
        );
        assert_eq!(pixel[3], 255, "unexpected alpha channel: {pixel:?}");
    }

    drop(command_pool);
    drop(framebuffer);
    drop(render_pass);
    drop(image_view);
    drop(readback);
    drop(image);
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
