#![cfg(target_os = "linux")]

use std::sync::Arc;

use ash::vk;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event_loop::{ActiveEventLoop, EventLoop},
    platform::x11::EventLoopBuilderExtX11,
    window::{Window, WindowAttributes},
};
use z_ideal_renderer::{
    CommandPoolConfig, FenceWaitResult, FrameStart, ImageConfig, ImageViewConfig,
    RenderPassAttachmentConfig, RenderPassAttachmentReference, RenderPassConfig,
    RenderPassDependency, RenderPassSubpassConfig, SwapchainConfig, VulkanCommandPool,
    VulkanConfig, VulkanContext, VulkanFrameScheduler, VulkanImage, VulkanQueue, VulkanRenderPass,
    VulkanSwapchainFramebuffer,
};

/// Runs against Xvfb + Mesa lavapipe and the Khronos validation layer.
#[test]
#[ignore = "requires X11/Xvfb, a Vulkan ICD, and VK_LAYER_KHRONOS_validation"]
fn swapchain_frame_submission_presents_with_a_depth_attachment_validation_clean() {
    struct TestApplication {
        result: Option<Result<(), String>>,
    }

    impl ApplicationHandler for TestApplication {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let attributes: WindowAttributes = Window::default_attributes()
                .with_title("Z Ideal Vulkan presentation validation")
                .with_inner_size(PhysicalSize::new(64, 64));
            let result = event_loop
                .create_window(attributes)
                .map_err(|error| error.to_string())
                .and_then(|window| run_swapchain_validation(&window));
            self.result = Some(result);
            event_loop.exit();
        }

        fn window_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _window_id: winit::window::WindowId,
            _event: winit::event::WindowEvent,
        ) {
        }
    }

    let mut builder = EventLoop::builder();
    builder.with_x11().with_any_thread(true);
    let event_loop = builder.build().expect("create X11 event loop");
    let mut application = TestApplication { result: None };
    event_loop
        .run_app(&mut application)
        .expect("run X11 presentation test application");
    application
        .result
        .expect("application resumed and attempted Vulkan presentation")
        .expect("Vulkan swapchain presentation validation");
}

fn run_swapchain_validation(window: &Window) -> Result<(), String> {
    let context = VulkanContext::new_for_window(
        &VulkanConfig {
            application_name: "Z Ideal Vulkan swapchain validation".to_owned(),
            enable_validation: true,
        },
        window,
    )
    .map_err(|error| error.to_string())?;
    let instance = Arc::clone(context.instance());
    let device = context.device();
    let extent = [64, 64];
    let swapchain = context
        .create_swapchain(extent, SwapchainConfig::default())
        .map_err(|error| error.to_string())?;
    let swapchain_extent = swapchain
        .extent()
        .ok_or_else(|| "new swapchain has no drawable extent".to_owned())?;
    let swapchain_format = swapchain
        .format()
        .ok_or_else(|| "new swapchain has no selected format".to_owned())?;

    let depth_format = vk::Format::D32_SFLOAT;
    let depth_image = VulkanImage::new(
        device,
        ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                depth: 1,
            },
            format: depth_format,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
        },
    )
    .map_err(|error| error.to_string())?;
    let depth_view = depth_image
        .create_view(ImageViewConfig {
            aspect: vk::ImageAspectFlags::DEPTH,
            ..ImageViewConfig::default()
        })
        .map_err(|error| error.to_string())?;

    let render_pass = VulkanRenderPass::new(
        device,
        RenderPassConfig {
            attachments: vec![
                RenderPassAttachmentConfig {
                    format: swapchain_format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                },
                RenderPassAttachmentConfig {
                    format: depth_format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                },
            ],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![RenderPassAttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }],
                depth_stencil_attachment: Some(RenderPassAttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                }),
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![RenderPassDependency {
                source_subpass: 0,
                destination_subpass: vk::SUBPASS_EXTERNAL,
                source_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                destination_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                source_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                destination_access_mask: vk::AccessFlags::MEMORY_READ,
                flags: vk::DependencyFlags::BY_REGION,
            }],
        },
    )
    .map_err(|error| error.to_string())?;

    let mut scheduler = VulkanFrameScheduler::new(device, 1).map_err(|error| error.to_string())?;
    let queue = VulkanQueue::graphics(device);
    let command_pool = VulkanCommandPool::new(device, CommandPoolConfig::default())
        .map_err(|error| error.to_string())?;
    let mut command_buffer = command_pool
        .allocate_primary()
        .map_err(|error| error.to_string())?;

    let ticket = match scheduler
        .begin_frame(&swapchain, u64::MAX)
        .map_err(|error| error.to_string())?
    {
        FrameStart::Ready(ticket) => ticket,
        FrameStart::OutOfDate => {
            return Err("fresh swapchain was out of date on first acquire".to_owned());
        }
        FrameStart::Timeout => {
            return Err("first swapchain acquire unexpectedly timed out".to_owned());
        }
        FrameStart::Pending => {
            return Err("fresh frame scheduler unexpectedly reported pending".to_owned());
        }
    };
    let image_index = ticket.image_index();
    let framebuffer = VulkanSwapchainFramebuffer::with_attachments(
        &render_pass,
        &swapchain,
        image_index,
        0,
        &[&depth_view],
    )
    .map_err(|error| error.to_string())?;
    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.125, 0.25, 0.5, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .map_err(|error| error.to_string())?;
    recording
        .begin_swapchain_render_pass(&render_pass, &framebuffer, &clear_values)
        .map_err(|error| error.to_string())?;
    recording
        .end_render_pass()
        .map_err(|error| error.to_string())?;
    let executable = recording.end().map_err(|error| error.to_string())?;
    let (present_result, mut submission) = ticket
        .submit_and_present(&queue, executable)
        .map_err(|error| error.to_string())?;
    if !matches!(
        present_result,
        z_ideal_renderer::PresentResult::Presented { .. }
    ) {
        return Err(format!(
            "first swapchain present did not succeed: {present_result:?}"
        ));
    }
    if submission
        .wait(u64::MAX)
        .map_err(|error| error.to_string())?
        != FenceWaitResult::Signaled
    {
        return Err("swapchain frame submission did not complete".to_owned());
    }
    drop(submission);

    drop(framebuffer);
    drop(command_pool);
    drop(scheduler);
    drop(depth_view);
    drop(depth_image);
    drop(render_pass);
    device.wait_idle().map_err(|error| error.to_string())?;
    drop(swapchain);
    drop(context);

    let validation_errors: Vec<_> = instance
        .validation_messages()
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter(|message| {
            message
                .severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        })
        .collect();
    drop(instance);
    if validation_errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Vulkan validation reported errors: {validation_errors:#?}"
        ))
    }
}
