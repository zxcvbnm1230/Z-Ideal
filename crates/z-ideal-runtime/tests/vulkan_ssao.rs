use std::sync::Arc;

use ash::vk;
use z_ideal_math::PerspectiveCamera;
use z_ideal_renderer::{
    BufferConfig, BufferMemoryLocation, CommandPoolConfig, DescriptorBinding, DescriptorPoolConfig,
    DescriptorPoolSize, DescriptorSetLayoutConfig, FenceWaitResult, GraphicsPipelineConfig,
    ImageBarrier, ImageConfig, ImageViewConfig, RenderPassAttachmentConfig,
    RenderPassAttachmentReference, RenderPassConfig, RenderPassDependency, RenderPassSubpassConfig,
    SamplerConfig, ShaderCompileConfig, ShaderStage, VulkanBuffer, VulkanCommandPool, VulkanConfig,
    VulkanDescriptorPool, VulkanDescriptorSetLayout, VulkanDevice, VulkanFramebuffer,
    VulkanGraphicsPipeline, VulkanImage, VulkanInstance, VulkanPipelineLayout, VulkanQueue,
    VulkanRenderPass, VulkanSampler, VulkanShaderModule, compile_glsl,
};
use z_ideal_runtime::{
    SsaoDescriptorResources, SsaoParameters, VulkanSsaoRenderer, compile_ssao_shader,
    write_ssao_descriptors,
};

#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run with --ignored on a Vulkan host"]
fn ssao_compute_dispatch_detects_nearby_depth_occluder() {
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const PIXELS: usize = (WIDTH * HEIGHT) as usize;

    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal SSAO compute validation".to_owned(),
            enable_validation: true,
        })
        .expect("create validation-enabled Vulkan instance"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan device");
    let graphics_queue = VulkanQueue::graphics(&device);
    let compute_queue = VulkanQueue::compute(&device).expect("select compute-capable queue");
    assert_eq!(
        graphics_queue.family(),
        compute_queue.family(),
        "this integrated offscreen test uses one graphics+compute queue family",
    );

    let depth_image = VulkanImage::new(
        &device,
        ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            },
            format: vk::Format::D32_SFLOAT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        },
    )
    .expect("create SSAO scene depth image");
    let depth_view = depth_image
        .create_view(ImageViewConfig {
            aspect: vk::ImageAspectFlags::DEPTH,
            ..ImageViewConfig::default()
        })
        .expect("create SSAO depth view");
    let depth_sampler = VulkanSampler::new(
        &device,
        SamplerConfig {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_lod: 0.0,
            ..SamplerConfig::default()
        },
    )
    .expect("create depth sampler");

    let depth_render_pass = VulkanRenderPass::new(
        &device,
        RenderPassConfig {
            attachments: vec![RenderPassAttachmentConfig {
                format: vk::Format::D32_SFLOAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            }],
            subpasses: vec![RenderPassSubpassConfig {
                depth_stencil_attachment: Some(RenderPassAttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                }),
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![RenderPassDependency {
                source_subpass: 0,
                destination_subpass: vk::SUBPASS_EXTERNAL,
                source_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                destination_stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                source_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                destination_access_mask: vk::AccessFlags::SHADER_READ,
                flags: vk::DependencyFlags::BY_REGION,
            }],
        },
    )
    .expect("create SSAO depth prepass");
    let depth_framebuffer = VulkanFramebuffer::new(
        &depth_render_pass,
        &[&depth_view],
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
        1,
    )
    .expect("create SSAO depth framebuffer");

    let mut depth_vertex_config = ShaderCompileConfig::new(ShaderStage::Vertex);
    depth_vertex_config.source_name = "z-ideal/ssao_test_depth.vert".to_owned();
    let mut depth_fragment_config = ShaderCompileConfig::new(ShaderStage::Fragment);
    depth_fragment_config.source_name = "z-ideal/ssao_test_depth.frag".to_owned();
    let depth_vertex = VulkanShaderModule::new(
        &device,
        compile_glsl(
            "#version 450\nvoid main() { const vec2 p[9] = vec2[9](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0), vec2(0.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0), vec2(0.0,-1.0), vec2(1.0,1.0), vec2(0.0,1.0)); float z = gl_VertexIndex < 3 ? 0.6 : 0.4; gl_Position = vec4(p[gl_VertexIndex], z, 1.0); }",
            &depth_vertex_config,
        ).expect("compile depth prepass vertex shader"),
    ).expect("create depth prepass vertex shader");
    let depth_fragment = VulkanShaderModule::new(
        &device,
        compile_glsl("#version 450\nvoid main() {}", &depth_fragment_config)
            .expect("compile depth prepass fragment shader"),
    )
    .expect("create depth prepass fragment shader");
    let depth_pipeline_layout =
        VulkanPipelineLayout::new(&device, &[], &[]).expect("create depth prepass pipeline layout");
    let depth_pipeline_config = GraphicsPipelineConfig {
        depth_test_enable: true,
        depth_write_enable: true,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        ..GraphicsPipelineConfig::default()
    };
    let depth_pipeline = VulkanGraphicsPipeline::new(
        &depth_pipeline_layout,
        &depth_vertex,
        &depth_fragment,
        &depth_render_pass,
        &depth_pipeline_config,
    )
    .expect("create depth prepass pipeline");

    let ao_image = VulkanImage::new(
        &device,
        ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            },
            format: vk::Format::R32_SFLOAT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            flags: vk::ImageCreateFlags::empty(),
        },
    )
    .expect("create SSAO storage output image");
    let ao_view = ao_image
        .create_view(ImageViewConfig::default())
        .expect("create SSAO output view");

    let descriptor_layout = VulkanDescriptorSetLayout::new(
        &device,
        DescriptorSetLayoutConfig {
            bindings: vec![
                DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                },
                DescriptorBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                },
                DescriptorBinding {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                },
            ],
        },
    )
    .expect("create SSAO descriptor layout");
    let descriptor_pool = VulkanDescriptorPool::new(
        &device,
        DescriptorPoolConfig {
            max_sets: 1,
            pool_sizes: vec![
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                },
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                },
            ],
        },
    )
    .expect("create SSAO descriptor pool");
    let mut descriptor_set = descriptor_pool
        .allocate_set(&descriptor_layout)
        .expect("allocate SSAO descriptor set");
    let mut uniform_buffer = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: 96,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            memory: BufferMemoryLocation::Upload,
        },
    )
    .expect("create SSAO uniform buffer");
    let parameters = SsaoParameters {
        inverse_projection: PerspectiveCamera::new(1.0, 1.0, 0.1, 100.0)
            .expect("create SSAO test camera")
            .projection_matrix()
            .expect("build SSAO test projection")
            .inverse()
            .to_cols_array_2d(),
        radius_pixels: 4.0,
        radius_world: 1.0,
        depth_bias: 0.001,
        intensity: 4.0,
        extent: [WIDTH, HEIGHT],
    };
    let descriptor_lease = write_ssao_descriptors(
        &mut descriptor_set,
        &mut uniform_buffer,
        parameters,
        SsaoDescriptorResources {
            depth: &depth_view,
            depth_sampler: &depth_sampler,
            output: &ao_view,
        },
    )
    .expect("write SSAO descriptors");

    let compute_artifact = compile_ssao_shader().expect("compile SSAO compute shader");
    let compute_shader =
        VulkanShaderModule::new(&device, compute_artifact).expect("create SSAO compute shader");
    let compute_layout = VulkanPipelineLayout::new(&device, &[&descriptor_layout], &[])
        .expect("create SSAO compute pipeline layout");
    let ssao =
        VulkanSsaoRenderer::new(&compute_layout, &compute_shader).expect("create SSAO renderer");

    let mut readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: (PIXELS * 4) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create SSAO readback buffer");
    let command_pool =
        VulkanCommandPool::new_for_queue(&device, &compute_queue, CommandPoolConfig::default())
            .expect("create graphics+compute command pool");
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate SSAO command buffer");
    let depth_clear = [vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    }];
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin SSAO command recording");
    recording
        .begin_render_pass(&depth_render_pass, &depth_framebuffer, &depth_clear)
        .expect("begin depth prepass");
    recording
        .bind_graphics_pipeline(&depth_pipeline)
        .expect("bind depth prepass pipeline");
    recording
        .set_viewport(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .expect("set depth prepass viewport");
    recording
        .set_scissor(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            },
        })
        .expect("set depth prepass scissor");
    recording
        .draw(3, 1, 0, 0)
        .expect("draw depth prepass fullscreen triangle");
    recording.end_render_pass().expect("finish depth prepass");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: &ao_image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                source_access: vk::AccessFlags::empty(),
                destination_access: vk::AccessFlags::SHADER_WRITE,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )
        .expect("transition SSAO output to general layout");
    ssao.record(descriptor_lease.as_ref(), &ao_image, &mut recording)
        .expect("dispatch SSAO compute pass");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: &ao_image,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                source_access: vk::AccessFlags::SHADER_WRITE,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )
        .expect("transition SSAO output for readback");
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
    recording
        .copy_image_to_buffer(
            &ao_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &region,
        )
        .expect("copy SSAO output to readback buffer");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[z_ideal_renderer::BufferBarrier {
                buffer: &readback,
                source_access: vk::AccessFlags::TRANSFER_WRITE,
                destination_access: vk::AccessFlags::HOST_READ,
                offset: 0,
                size: vk::WHOLE_SIZE,
            }],
            &[],
        )
        .expect("make SSAO output visible to host");
    let executable = recording.end().expect("finish SSAO command buffer");
    let mut submission = compute_queue
        .submit(executable, &[], &[], &[])
        .expect("submit SSAO compute work");
    assert_eq!(
        submission.wait(u64::MAX).expect("wait for SSAO compute"),
        FenceWaitResult::Signaled
    );
    drop(submission);
    drop(descriptor_lease);

    let mut bytes = vec![0_u8; PIXELS * 4];
    readback.read(0, &mut bytes).expect("read SSAO output");
    for pixel in bytes.as_chunks::<4>().0 {
        let ao = f32::from_ne_bytes(*pixel);
        assert!(
            (ao - 1.0).abs() <= 1.0e-5,
            "flat unoccluded depth plane must produce AO=1, got {ao}"
        );
    }

    let errors: Vec<_> = instance
        .validation_messages()
        .expect("read SSAO validation messages")
        .into_iter()
        .filter(|message| {
            message
                .severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        })
        .collect();
    assert!(errors.is_empty(), "Vulkan validation errors: {errors:#?}");
}
