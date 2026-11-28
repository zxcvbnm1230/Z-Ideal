use std::sync::Arc;

use ash::vk;
use z_ideal_assets::{
    AssetCache, AssetId, AssetKind, AssetUri, GltfImport, GltfImporter, HdrEnvironmentMap,
    ImportedImage, ImportedMesh, ImportedPrimitive, ImportedTexture, PrimitiveTopology,
    TextureWrapMode,
};
use z_ideal_math::{Mat4, PerspectiveCamera, Quat, Transform, Vec3};
use z_ideal_renderer::{
    BufferBarrier, BufferConfig, BufferDescriptorWrite, BufferMemoryLocation, CommandPoolConfig,
    DescriptorBinding, DescriptorPoolConfig, DescriptorPoolSize, DescriptorResourceWrite,
    DescriptorSetLayoutConfig, EnvironmentLightingMaps, FenceWaitResult, GraphicsPipelineConfig,
    ImageConfig, ImageDescriptorWrite, ImageViewConfig, MeshUploadError,
    RenderPassAttachmentConfig, RenderPassAttachmentReference, RenderPassConfig,
    RenderPassDependency, RenderPassSubpassConfig, SamplerConfig, ShaderCompileConfig, ShaderStage,
    TextureColorSpace, TextureUploadError, VulkanBuffer, VulkanCommandPool, VulkanConfig,
    VulkanDescriptorPool, VulkanDescriptorSetLayout, VulkanDevice, VulkanEnvironmentCube,
    VulkanFramebuffer, VulkanGraphicsPipeline, VulkanHdrColorTarget, VulkanImage, VulkanInstance,
    VulkanPipelineLayout, VulkanQueue, VulkanRenderPass, VulkanSampler, VulkanShaderModule,
    VulkanTexture2D, compile_glsl,
};
use z_ideal_runtime::{
    GltfMaterialTextureBinding, GltfMaterialTextureSet, GltfMaterialTextureSlot, GltfMeshInstance,
    GltfPbrEnvironmentResources, GltfPbrLocalLight, GltfPbrLocalLightKind, GltfPbrSceneUniform,
    RenderMeshInstance, RenderSceneSnapshot, VulkanSceneMeshes, VulkanSceneMeshesError,
    VulkanSceneRenderer, VulkanSceneViewport, VulkanShadowRenderer, compile_gltf_pbr_shaders,
    compile_shadow_shaders, extract_render_scene, instantiate_default_gltf_scene,
    write_gltf_material_descriptors, write_gltf_pbr_scene_descriptor,
};
use z_ideal_scene::World;

fn triangle_import(mode: PrimitiveTopology) -> GltfImport {
    GltfImport {
        source_path: "models/triangle.gltf".into(),
        meshes: vec![ImportedMesh {
            source_index: 0,
            name: Some("triangle".to_owned()),
            primitives: vec![ImportedPrimitive {
                mode,
                positions: vec![[-0.8, -0.8, 0.5], [0.8, -0.8, 0.5], [0.0, 0.8, 0.5]],
                normals: None,
                tangents: None,
                tex_coords_0: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
                tex_coords_1: None,
                indices: vec![0, 1, 2],
                material_index: None,
            }],
        }],
        nodes: Vec::new(),
        scenes: Vec::new(),
        default_scene: None,
        materials: Vec::new(),
        textures: Vec::new(),
        images: Vec::new(),
        dependency_uris: Vec::new(),
    }
}

fn snapshot(uri: AssetUri, mesh_index: usize) -> RenderSceneSnapshot {
    let mut world = World::new();
    let entity = world.spawn().expect("create stable snapshot entity");
    RenderSceneSnapshot {
        camera_entity: entity,
        view: Mat4::IDENTITY,
        projection: Mat4::IDENTITY,
        meshes: vec![RenderMeshInstance {
            entity,
            source: GltfMeshInstance {
                source_uri: uri,
                mesh_index,
                source_node_index: 0,
            },
            model: Mat4::IDENTITY,
        }],
    }
}

fn record_texture_upload(device: &VulkanDevice, texture: &VulkanTexture2D<'_>) {
    let command_pool = VulkanCommandPool::new(device, CommandPoolConfig::default())
        .expect("create texture upload command pool");
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate texture upload command buffer");
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin texture upload recording");
    texture
        .record_upload(&mut recording)
        .expect("record imported texture upload and shader-read transition");
    assert!(matches!(
        texture.record_upload(&mut recording),
        Err(TextureUploadError::AlreadyRecorded)
    ));
    let executable = recording
        .end()
        .expect("finish texture upload command buffer");
    let queue = VulkanQueue::graphics(device);
    let mut submission = queue
        .submit(executable, &[], &[], &[])
        .expect("submit imported texture upload");
    assert_eq!(
        submission.wait(u64::MAX).expect("wait for texture upload"),
        FenceWaitResult::Signaled
    );
    drop(submission);
    let view = texture.create_view().expect("create imported texture view");
    assert_eq!(
        view.config().level_count,
        texture.image().config().mip_levels
    );
    assert_eq!(
        texture.sampler().config().max_lod,
        (texture.image().config().mip_levels - 1) as f32
    );
}

fn record_scene_draws(
    device: &VulkanDevice,
    gpu_scene: &VulkanSceneMeshes<'_>,
    scene: &RenderSceneSnapshot,
    imported_texture: Option<&VulkanTexture2D<'_>>,
    material_factor: [f32; 4],
) -> (usize, Vec<u8>) {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    let fallback_image = ImportedImage {
        width: 1,
        height: 1,
        format: z_ideal_assets::ImagePixelFormat::Rgba8,
        pixels: vec![255; 4],
    };
    let fallback_sampler = ImportedTexture {
        source_image: 0,
        mag_filter: None,
        min_filter: None,
        wrap_s: TextureWrapMode::Repeat,
        wrap_t: TextureWrapMode::Repeat,
    };
    let fallback_texture = if imported_texture.is_none() {
        let texture = VulkanTexture2D::new(
            device,
            &fallback_image,
            &fallback_sampler,
            TextureColorSpace::Srgb,
        )
        .expect("create white fallback texture");
        record_texture_upload(device, &texture);
        Some(texture)
    } else {
        None
    };
    let texture = imported_texture
        .or(fallback_texture.as_ref())
        .expect("a sampled texture is always supplied");
    let hdr_target = VulkanHdrColorTarget::new(device, WIDTH, HEIGHT)
        .expect("create floating-point HDR scene target");
    let hdr_view = hdr_target.create_view().expect("create HDR scene view");
    let hdr_render_pass = VulkanRenderPass::new(
        device,
        RenderPassConfig {
            attachments: vec![RenderPassAttachmentConfig {
                format: z_ideal_renderer::HDR_COLOR_FORMAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
                destination_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                source_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                destination_access_mask: vk::AccessFlags::SHADER_READ,
                flags: vk::DependencyFlags::BY_REGION,
            }],
        },
    )
    .expect("create HDR scene render pass");
    let hdr_framebuffer = VulkanFramebuffer::new(
        &hdr_render_pass,
        &[&hdr_view],
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
        1,
    )
    .expect("create HDR scene framebuffer");

    let output_config = ImageConfig {
        image_type: vk::ImageType::TYPE_2D,
        extent: vk::Extent3D {
            width: WIDTH,
            height: HEIGHT,
            depth: 1,
        },
        format: vk::Format::R8G8B8A8_SRGB,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        flags: vk::ImageCreateFlags::empty(),
    };
    let output_image =
        VulkanImage::new(device, output_config).expect("create tone-mapped output target");
    let output_view = output_image
        .create_view(ImageViewConfig::default())
        .expect("create tone-mapped output view");
    let tone_render_pass = VulkanRenderPass::new(
        device,
        RenderPassConfig {
            attachments: vec![RenderPassAttachmentConfig {
                format: output_config.format,
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
    .expect("create tone-mapping render pass");
    let tone_framebuffer = VulkanFramebuffer::new(
        &tone_render_pass,
        &[&output_view],
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
        1,
    )
    .expect("create tone-mapping framebuffer");

    let descriptor_layout = VulkanDescriptorSetLayout::new(
        device,
        DescriptorSetLayoutConfig {
            bindings: vec![
                DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                },
                DescriptorBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                },
            ],
        },
    )
    .expect("create glTF material descriptor layout");
    let descriptor_pool = VulkanDescriptorPool::new(
        device,
        DescriptorPoolConfig {
            max_sets: 1,
            pool_sizes: vec![
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                },
            ],
        },
    )
    .expect("create glTF material descriptor pool");
    let mut descriptor_set = descriptor_pool
        .allocate_set(&descriptor_layout)
        .expect("allocate glTF material descriptor set");
    let texture_view = texture.create_view().expect("create sampled texture view");
    let material_bytes: Vec<u8> = material_factor
        .iter()
        .flat_map(|component| component.to_ne_bytes())
        .collect();
    let mut material_buffer = VulkanBuffer::new(
        device,
        BufferConfig {
            size: material_bytes.len() as u64,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            memory: BufferMemoryLocation::Upload,
        },
    )
    .expect("create material factor uniform buffer");
    material_buffer
        .write(0, &material_bytes)
        .expect("write material factor uniform data");
    let descriptor_lease = descriptor_set
        .write_descriptors(&[
            DescriptorResourceWrite::Image(ImageDescriptorWrite {
                binding: 0,
                array_element: 0,
                view: &texture_view,
                sampler: Some(texture.sampler()),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }),
            DescriptorResourceWrite::Buffer(BufferDescriptorWrite {
                binding: 1,
                array_element: 0,
                buffer: &material_buffer,
                offset: 0,
                range: material_bytes.len() as u64,
            }),
        ])
        .expect("write glTF material texture and factor descriptors");

    let vertex_shader = VulkanShaderModule::new(
        device,
        compile_glsl(
            "#version 450\nlayout(location=0) in vec3 position;\nlayout(location=1) in vec2 tex_coord;\nlayout(location=0) out vec2 uv;\nlayout(push_constant) uniform Draw { mat4 mvp; } draw_data;\nvoid main() { gl_Position = draw_data.mvp * vec4(position, 1.0); uv = tex_coord; }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        )
        .expect("compile scene vertex shader"),
    )
    .expect("create scene vertex shader module");
    let fragment_shader = VulkanShaderModule::new(
        device,
        compile_glsl(
            "#version 450\nlayout(set=0,binding=0) uniform sampler2D base_color_texture;\nlayout(set=0,binding=1) uniform Material { vec4 base_color_factor; } material;\nlayout(location=0) in vec2 uv;\nlayout(location=0) out vec4 output_color;\nvoid main() { output_color = texture(base_color_texture, uv) * material.base_color_factor * 4.0; }",
            &ShaderCompileConfig::new(ShaderStage::Fragment),
        )
        .expect("compile scene fragment shader"),
    )
    .expect("create scene fragment shader module");
    let push_constants = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64)];
    let pipeline_layout = VulkanPipelineLayout::new(device, &[&descriptor_layout], &push_constants)
        .expect("create scene pipeline layout");
    let pipeline = VulkanGraphicsPipeline::new(
        &pipeline_layout,
        &vertex_shader,
        &fragment_shader,
        &hdr_render_pass,
        &GraphicsPipelineConfig {
            vertex_bindings: vec![z_ideal_renderer::MeshGpu::vertex_binding()],
            vertex_attributes: z_ideal_renderer::MeshGpu::vertex_attributes().to_vec(),
            ..GraphicsPipelineConfig::default()
        },
    )
    .expect("create scene graphics pipeline");

    let tone_descriptor_layout = VulkanDescriptorSetLayout::new(
        device,
        DescriptorSetLayoutConfig {
            bindings: vec![DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
            }],
        },
    )
    .expect("create tone-map descriptor layout");
    let tone_descriptor_pool = VulkanDescriptorPool::new(
        device,
        DescriptorPoolConfig {
            max_sets: 1,
            pool_sizes: vec![DescriptorPoolSize {
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            }],
        },
    )
    .expect("create tone-map descriptor pool");
    let mut tone_descriptor_set = tone_descriptor_pool
        .allocate_set(&tone_descriptor_layout)
        .expect("allocate tone-map descriptor set");
    let tone_descriptor_lease = tone_descriptor_set
        .write_descriptors(&[DescriptorResourceWrite::Image(ImageDescriptorWrite {
            binding: 0,
            array_element: 0,
            view: &hdr_view,
            sampler: Some(texture.sampler()),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        })])
        .expect("write HDR sampled-image descriptor");
    let tone_vertex_shader = VulkanShaderModule::new(
        device,
        compile_glsl(
            "#version 450\nlayout(location=0) out vec2 uv;\nvoid main() { vec2 p[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0)); vec2 pos = p[gl_VertexIndex]; gl_Position = vec4(pos,0.0,1.0); uv = pos * 0.5 + 0.5; }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        )
        .expect("compile tone-map vertex shader"),
    )
    .expect("create tone-map vertex module");
    let tone_fragment_shader = VulkanShaderModule::new(
        device,
        compile_glsl(
            "#version 450\nlayout(set=0,binding=0) uniform sampler2D hdr_color;\nlayout(location=0) in vec2 uv;\nlayout(location=0) out vec4 output_color;\nvoid main() { vec3 hdr = texture(hdr_color, uv).rgb; vec3 mapped = hdr / (vec3(1.0) + hdr); output_color = vec4(mapped, 1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Fragment),
        )
        .expect("compile tone-map fragment shader"),
    )
    .expect("create tone-map fragment module");
    let tone_pipeline_layout = VulkanPipelineLayout::new(device, &[&tone_descriptor_layout], &[])
        .expect("create tone-map pipeline layout");
    let tone_pipeline = VulkanGraphicsPipeline::new(
        &tone_pipeline_layout,
        &tone_vertex_shader,
        &tone_fragment_shader,
        &tone_render_pass,
        &GraphicsPipelineConfig::default(),
    )
    .expect("create tone-map graphics pipeline");

    let mut readback = VulkanBuffer::new(
        device,
        BufferConfig {
            size: u64::from(WIDTH) * u64::from(HEIGHT) * 4,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create scene readback buffer");
    let command_pool = VulkanCommandPool::new(device, CommandPoolConfig::default())
        .expect("create scene command pool");
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate scene command buffer");
    let clear = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin scene command recording");
    gpu_scene
        .record_upload_barriers(&mut recording)
        .expect("synchronize uploaded scene mesh buffers");
    recording
        .begin_render_pass(&hdr_render_pass, &hdr_framebuffer, &clear)
        .expect("begin scene render pass");
    recording
        .bind_graphics_pipeline(&pipeline)
        .expect("bind scene graphics pipeline");
    recording
        .set_viewport(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .expect("set scene viewport");
    recording
        .set_scissor(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            },
        })
        .expect("set scene scissor");
    let draw_count = gpu_scene
        .record_draws_with_materials(
            scene,
            &mut recording,
            &pipeline_layout,
            |material, recording, layout| {
                if let Some(material) = material {
                    assert_eq!(material.base_color_factor, material_factor);
                }
                recording.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[descriptor_lease.as_ref()],
                    &[],
                )
            },
        )
        .expect("record runtime scene draws with per-primitive materials");
    recording
        .end_render_pass()
        .expect("end HDR scene render pass");
    recording
        .begin_render_pass(&tone_render_pass, &tone_framebuffer, &clear)
        .expect("begin tone-mapping pass");
    recording
        .bind_graphics_pipeline(&tone_pipeline)
        .expect("bind tone-map pipeline");
    recording
        .bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            &tone_pipeline_layout,
            0,
            &[tone_descriptor_lease.as_ref()],
            &[],
        )
        .expect("bind HDR image for tone mapping");
    recording
        .set_viewport(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .expect("set tone-map viewport");
    recording
        .set_scissor(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            },
        })
        .expect("set tone-map scissor");
    recording
        .draw(3, 1, 0, 0)
        .expect("draw full-screen tone-map triangle");
    recording.end_render_pass().expect("end tone-mapping pass");
    let copy_region = [vk::BufferImageCopy::default()
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
            &output_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &copy_region,
        )
        .expect("copy rendered scene to readback buffer");
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
        .expect("make rendered pixels visible to the host");
    let executable = recording.end().expect("finish scene command buffer");
    let queue = VulkanQueue::graphics(device);
    let mut submission = queue
        .submit(executable, &[], &[], &[])
        .expect("submit runtime scene draws");
    assert_eq!(
        submission.wait(u64::MAX).expect("wait for scene draw"),
        FenceWaitResult::Signaled
    );
    drop(submission);

    let mut pixels = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    readback
        .read(0, &mut pixels)
        .expect("read rendered scene pixels");
    (draw_count, pixels)
}

#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run with --ignored on a Vulkan host"]
fn gltf_pbr_material_descriptors_and_shader_pipeline_record_validation_clean() {
    let source_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_scene.gltf");
    let imported = GltfImporter::import(&source_path).expect("import glTF PBR fixture");
    let source_uri = AssetUri::new("fixtures/sample_scene.gltf").expect("valid source URI");
    let material = imported.materials.first().expect("fixture material");
    let mut cache = AssetCache::new();
    cache
        .insert(
            AssetId::from_uri(AssetKind::Mesh, &source_uri),
            imported.clone(),
        )
        .expect("cache imported PBR asset");
    let scene = snapshot(source_uri.clone(), 0);

    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal glTF PBR descriptor validation".to_owned(),
            enable_validation: true,
        })
        .expect("create validation-enabled Vulkan instance"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan device");

    {
        let gpu_scene =
            VulkanSceneMeshes::upload(&device, &cache, &scene).expect("upload PBR scene mesh");
        let hdr_environment = HdrEnvironmentMap {
            source_path: "pbr-test-environment.hdr".into(),
            width: 4,
            height: 2,
            pixels: vec![[0.5, 0.4, 0.3]; 8],
        };
        let lighting_maps =
            EnvironmentLightingMaps::from_equirectangular(&hdr_environment, 2, 4, 3, 64)
                .expect("generate deterministic PBR IBL test maps");
        let irradiance_cube = VulkanEnvironmentCube::from_cube_mips(
            &device,
            std::slice::from_ref(lighting_maps.irradiance()),
        )
        .expect("create diffuse irradiance cube");
        let specular_cube =
            VulkanEnvironmentCube::from_cube_mips(&device, lighting_maps.specular_mips())
                .expect("create GGX-prefiltered specular cube");

        let material_textures = GltfMaterialTextureSet::from_import(&device, &imported, material)
            .expect("create semantic material textures and neutral fallbacks");
        let mut untextured_material = material.clone();
        untextured_material.base_color_texture = None;
        untextured_material.metallic_roughness_texture = None;
        untextured_material.normal_texture = None;
        untextured_material.occlusion_texture = None;
        untextured_material.emissive_texture = None;
        let fallback_textures =
            GltfMaterialTextureSet::from_import(&device, &imported, &untextured_material)
                .expect("create all five neutral fallback textures");
        let upload_pool = VulkanCommandPool::new(&device, CommandPoolConfig::default())
            .expect("create material texture upload pool");
        let mut upload_buffer = upload_pool
            .allocate_primary()
            .expect("allocate material texture upload command buffer");
        let mut upload_recording = upload_buffer
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .expect("begin material texture upload");
        material_textures
            .record_upload(&mut upload_recording)
            .expect("record all five semantic texture uploads");
        fallback_textures
            .record_upload(&mut upload_recording)
            .expect("record all five neutral fallback texture uploads");
        irradiance_cube
            .record_upload(&mut upload_recording)
            .expect("record diffuse irradiance cube upload");
        specular_cube
            .record_upload(&mut upload_recording)
            .expect("record GGX specular mip-chain upload");
        let upload_executable = upload_recording.end().expect("finish material uploads");
        let mut upload_submission = VulkanQueue::graphics(&device)
            .submit(upload_executable, &[], &[], &[])
            .expect("submit material texture uploads");
        assert_eq!(
            upload_submission
                .wait(u64::MAX)
                .expect("wait for texture uploads"),
            FenceWaitResult::Signaled
        );
        drop(upload_submission);
        let texture_views = material_textures
            .create_views()
            .expect("create all five material texture views");
        let _fallback_views = fallback_textures
            .create_views()
            .expect("create all five neutral fallback texture views");
        let irradiance_view = irradiance_cube
            .create_view()
            .expect("create diffuse irradiance cube view");
        let specular_view = specular_cube
            .create_view()
            .expect("create prefiltered specular cube view");
        let shadow_image = VulkanImage::new(
            &device,
            ImageConfig {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: 32,
                    height: 32,
                    depth: 1,
                },
                format: vk::Format::D32_SFLOAT,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
        )
        .expect("create directional shadow depth image");
        let shadow_view = shadow_image
            .create_view(ImageViewConfig {
                aspect: vk::ImageAspectFlags::DEPTH,
                ..ImageViewConfig::default()
            })
            .expect("create directional shadow depth view");
        let ao_image = VulkanImage::new(
            &device,
            ImageConfig {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: 32,
                    height: 32,
                    depth: 1,
                },
                format: vk::Format::R32_SFLOAT,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
        )
        .expect("create neutral screen-space AO image");
        let ao_view = ao_image
            .create_view(ImageViewConfig::default())
            .expect("create screen-space AO view");
        let ao_render_pass = VulkanRenderPass::new(
            &device,
            RenderPassConfig {
                attachments: vec![RenderPassAttachmentConfig {
                    format: vk::Format::R32_SFLOAT,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
                    destination_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    source_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    destination_access_mask: vk::AccessFlags::SHADER_READ,
                    flags: vk::DependencyFlags::BY_REGION,
                }],
            },
        )
        .expect("create neutral AO clear render pass");
        let ao_framebuffer = VulkanFramebuffer::new(
            &ao_render_pass,
            &[&ao_view],
            vk::Extent2D {
                width: 32,
                height: 32,
            },
            1,
        )
        .expect("create neutral AO framebuffer");
        let shadow_sampler = VulkanSampler::new(
            &device,
            SamplerConfig {
                mag_filter: vk::Filter::NEAREST,
                min_filter: vk::Filter::NEAREST,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                compare_op: Some(vk::CompareOp::LESS_OR_EQUAL),
                max_lod: 0.0,
                ..SamplerConfig::default()
            },
        )
        .expect("create directional shadow comparison sampler");
        let ao_sampler = VulkanSampler::new(
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
        .expect("create non-comparison AO sampler");
        let shadow_render_pass = VulkanRenderPass::new(
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
                    destination_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    source_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    destination_access_mask: vk::AccessFlags::SHADER_READ,
                    flags: vk::DependencyFlags::BY_REGION,
                }],
            },
        )
        .expect("create directional shadow clear render pass");
        let shadow_framebuffer = VulkanFramebuffer::new(
            &shadow_render_pass,
            &[&shadow_view],
            vk::Extent2D {
                width: 32,
                height: 32,
            },
            1,
        )
        .expect("create directional shadow framebuffer");
        let (shadow_vertex_artifact, shadow_fragment_artifact) =
            compile_shadow_shaders().expect("compile directional shadow shaders");
        let shadow_vertex = VulkanShaderModule::new(&device, shadow_vertex_artifact)
            .expect("create directional shadow vertex shader");
        let shadow_fragment = VulkanShaderModule::new(&device, shadow_fragment_artifact)
            .expect("create directional shadow fragment shader");
        let shadow_push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64)];
        let shadow_layout = VulkanPipelineLayout::new(&device, &[], &shadow_push_constants)
            .expect("create directional shadow pipeline layout");
        let shadow_pipeline_config = GraphicsPipelineConfig {
            vertex_bindings: vec![z_ideal_renderer::MeshGpu::vertex_binding()],
            vertex_attributes: vec![z_ideal_renderer::MeshGpu::vertex_attributes()[0]],
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            ..GraphicsPipelineConfig::default()
        };
        let shadow_renderer = VulkanShadowRenderer::new(
            &shadow_layout,
            &shadow_vertex,
            &shadow_fragment,
            &shadow_render_pass,
            &shadow_pipeline_config,
        )
        .expect("create directional shadow renderer");

        let material_layout = VulkanDescriptorSetLayout::new(
            &device,
            DescriptorSetLayoutConfig {
                bindings: std::iter::once(DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                })
                .chain((1..=5).map(|binding| DescriptorBinding {
                    binding,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                }))
                .collect(),
            },
        )
        .expect("create material descriptor layout");
        let scene_layout = VulkanDescriptorSetLayout::new(
            &device,
            DescriptorSetLayoutConfig {
                bindings: vec![
                    DescriptorBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                    DescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                    DescriptorBinding {
                        binding: 2,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                    DescriptorBinding {
                        binding: 3,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                    DescriptorBinding {
                        binding: 4,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                ],
            },
        )
        .expect("create scene descriptor layout");
        let descriptor_pool = VulkanDescriptorPool::new(
            &device,
            DescriptorPoolConfig {
                max_sets: 2,
                pool_sizes: vec![
                    DescriptorPoolSize {
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 2,
                    },
                    DescriptorPoolSize {
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 9,
                    },
                ],
            },
        )
        .expect("create PBR descriptor pool");
        let mut material_set = descriptor_pool
            .allocate_set(&material_layout)
            .expect("allocate material descriptor set");
        let mut scene_set = descriptor_pool
            .allocate_set(&scene_layout)
            .expect("allocate scene descriptor set");
        let mut material_buffer = VulkanBuffer::new(
            &device,
            BufferConfig {
                size: 96,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                memory: BufferMemoryLocation::Upload,
            },
        )
        .expect("create material uniform buffer");
        let mut scene_buffer = VulkanBuffer::new(
            &device,
            BufferConfig {
                size: 432,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                memory: BufferMemoryLocation::Upload,
            },
        )
        .expect("create scene uniform buffer");
        let material_lease = write_gltf_material_descriptors(
            &mut material_set,
            &mut material_buffer,
            material,
            texture_views.descriptor_resources(&material_textures),
        )
        .expect("write complete glTF material descriptor set");
        let scene_lease = write_gltf_pbr_scene_descriptor(
            &mut scene_set,
            &mut scene_buffer,
            GltfPbrSceneUniform {
                camera_position: [0.0, 0.0, 5.0],
                light_direction: [0.0, 0.0, -1.0],
                light_intensity: 1.0,
                light_color: [1.0, 1.0, 1.0],
                ambient_color: [0.0, 0.0, 0.0],
                environment_intensity: 1.0,
                environment_max_lod: (specular_cube.mip_levels() - 1) as f32,
                light_view_projection: (scene.projection * scene.view).to_cols_array_2d(),
                shadow_parameters: [1.0, 0.001, 1.0, 1.0],
                local_lights: [
                    GltfPbrLocalLight {
                        position: [0.0, 0.0, 2.0],
                        range: 10.0,
                        color: [0.2, 0.3, 0.4],
                        intensity: 0.5,
                        direction: [0.0, 0.0, -1.0],
                        inner_cone_cos: 0.9,
                        outer_cone_cos: 0.8,
                        kind: GltfPbrLocalLightKind::Spot,
                    },
                    GltfPbrLocalLight::default(),
                    GltfPbrLocalLight::default(),
                    GltfPbrLocalLight::default(),
                ],
                local_light_count: 1,
            },
            GltfPbrEnvironmentResources {
                diffuse_irradiance: z_ideal_runtime::GltfMaterialTextureResource {
                    view: &irradiance_view,
                    sampler: irradiance_cube.sampler(),
                },
                prefiltered_specular: z_ideal_runtime::GltfMaterialTextureResource {
                    view: &specular_view,
                    sampler: specular_cube.sampler(),
                },
                directional_shadow: z_ideal_runtime::GltfMaterialTextureResource {
                    view: &shadow_view,
                    sampler: &shadow_sampler,
                },
                screen_space_ao: z_ideal_runtime::GltfMaterialTextureResource {
                    view: &ao_view,
                    sampler: &ao_sampler,
                },
            },
        )
        .expect("write scene lighting descriptor");

        let (vertex_artifact, fragment_artifact) =
            compile_gltf_pbr_shaders().expect("compile standard glTF PBR shader pair");
        let vertex_shader =
            VulkanShaderModule::new(&device, vertex_artifact).expect("create PBR vertex module");
        let fragment_shader = VulkanShaderModule::new(&device, fragment_artifact)
            .expect("create PBR fragment module");
        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(128)];
        let pipeline_layout =
            VulkanPipelineLayout::new(&device, &[&material_layout, &scene_layout], &push_constants)
                .expect("create reflected PBR pipeline layout");
        let color_config = ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: 32,
                height: 32,
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
        let color_image =
            VulkanImage::new(&device, color_config).expect("create PBR readback color image");
        let color_view = color_image
            .create_view(ImageViewConfig::default())
            .expect("create PBR readback color view");
        let render_pass = VulkanRenderPass::new(
            &device,
            RenderPassConfig {
                attachments: vec![RenderPassAttachmentConfig {
                    format: color_config.format,
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
        .expect("create PBR render pass");
        let framebuffer = VulkanFramebuffer::new(
            &render_pass,
            &[&color_view],
            vk::Extent2D {
                width: 32,
                height: 32,
            },
            1,
        )
        .expect("create PBR framebuffer");
        let pipeline_config = GraphicsPipelineConfig {
            vertex_bindings: vec![z_ideal_renderer::MeshGpu::vertex_binding()],
            vertex_attributes: z_ideal_renderer::MeshGpu::vertex_attributes().to_vec(),
            ..GraphicsPipelineConfig::default()
        };
        let scene_renderer = VulkanSceneRenderer::new(
            &pipeline_layout,
            &vertex_shader,
            &fragment_shader,
            &render_pass,
            &pipeline_config,
        )
        .expect("create four-variant runtime PBR scene renderer");

        let mut readback = VulkanBuffer::new(
            &device,
            BufferConfig {
                size: 32 * 32 * 4,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
                memory: BufferMemoryLocation::Readback,
            },
        )
        .expect("create PBR pixel readback buffer");
        let command_pool = VulkanCommandPool::new(&device, CommandPoolConfig::default())
            .expect("create PBR draw command pool");
        let mut command_buffer = command_pool
            .allocate_primary()
            .expect("allocate PBR draw command buffer");
        let clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let mut recording = command_buffer
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .expect("begin PBR command recording");
        gpu_scene
            .record_upload_barriers(&mut recording)
            .expect("synchronize PBR mesh uploads");
        let shadow_clear = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];
        recording
            .begin_render_pass(&shadow_render_pass, &shadow_framebuffer, &shadow_clear)
            .expect("begin directional shadow-map pass");
        let shadow_draws = shadow_renderer
            .record(
                &gpu_scene,
                &scene,
                &mut recording,
                VulkanSceneViewport {
                    viewport: vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: 32.0,
                        height: 32.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    },
                    scissor: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: 32,
                            height: 32,
                        },
                    },
                },
            )
            .expect("record directional shadow mesh draws");
        assert_eq!(shadow_draws, 1);
        recording
            .end_render_pass()
            .expect("finish directional shadow-map pass");
        let ao_clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.5, 0.5, 0.5, 1.0],
            },
        }];
        recording
            .begin_render_pass(&ao_render_pass, &ao_framebuffer, &ao_clear)
            .expect("begin neutral AO clear pass");
        recording
            .end_render_pass()
            .expect("finish neutral AO clear pass");
        recording
            .begin_render_pass(&render_pass, &framebuffer, &clear)
            .expect("begin PBR render pass");
        let draws = scene_renderer
            .record(
                &gpu_scene,
                &scene,
                &mut recording,
                scene_lease.as_ref(),
                VulkanSceneViewport {
                    viewport: vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: 32.0,
                        height: 32.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    },
                    scissor: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: 32,
                            height: 32,
                        },
                    },
                },
                |_, recording, layout| {
                    recording.bind_descriptor_sets(
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        0,
                        &[material_lease.as_ref()],
                        &[],
                    )
                },
            )
            .expect("record material-bound runtime PBR scene");
        assert_eq!(draws, 1);
        recording.end_render_pass().expect("end PBR render pass");
        let copy_region = [vk::BufferImageCopy::default()
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
            .image_extent(color_config.extent)];
        recording
            .copy_image_to_buffer(
                &color_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                &readback,
                &copy_region,
            )
            .expect("copy PBR output to host readback");
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
            .expect("make PBR pixels visible to host");
        let executable = recording.end().expect("finish PBR draw command buffer");
        let mut submission = VulkanQueue::graphics(&device)
            .submit(executable, &[], &[], &[])
            .expect("submit PBR material draw");
        assert_eq!(
            submission.wait(u64::MAX).expect("wait for PBR draw"),
            FenceWaitResult::Signaled
        );
        drop(submission);
        let mut pixels = vec![0_u8; 32 * 32 * 4];
        readback
            .read(0, &mut pixels)
            .expect("read PBR output pixels");
        std::fs::write("/tmp/zideal-pbr-reference.rgba8", &pixels)
            .expect("write temporary AO PBR reference");
        let center = &pixels[(16 * 32 + 16) * 4..][..4];
        let clear = &pixels[(31 * 32 + 31) * 4..][..4];
        assert!(
            center[0] > 25 || center[1] > 51 || center[2] > 76,
            "PBR environment lighting did not contribute beyond emissive: {center:?}"
        );
        assert_eq!(clear, &[0, 0, 0, 255], "PBR clear pixel changed: {clear:?}");

        // The golden is a 32x32 RGBA8 capture from the NVIDIA reference device.
        // Per-channel tolerance is 16/255; up to 16 edge pixels may exceed it to
        // accommodate rasterization and floating-point variation across Vulkan drivers.
        const REFERENCE: &[u8; 4096] = include_bytes!("fixtures/pbr_reference.rgba8");
        const CHANNEL_TOLERANCE: u8 = 16;
        const MAX_OUTLIER_PIXELS: usize = 16;
        let mut outlier_pixels = 0;
        for (pixel_index, (actual, expected)) in pixels
            .chunks_exact(4)
            .zip(REFERENCE.chunks_exact(4))
            .enumerate()
        {
            assert_eq!(
                actual[3], expected[3],
                "reference alpha mismatch at pixel {pixel_index}"
            );
            let channel_error = (0..3)
                .map(|channel| actual[channel].abs_diff(expected[channel]))
                .max()
                .expect("three color channels");
            if channel_error > CHANNEL_TOLERANCE {
                outlier_pixels += 1;
            }
        }
        assert!(
            outlier_pixels <= MAX_OUTLIER_PIXELS,
            "PBR reference image exceeded tolerance at {outlier_pixels} pixels (limit {MAX_OUTLIER_PIXELS})"
        );

        drop(material_lease);
        drop(scene_lease);
    } // All device-owned resources are dropped before the logical device.
    drop(device);
    let errors: Vec<_> = instance
        .validation_messages()
        .expect("read PBR validation messages")
        .into_iter()
        .filter(|message| {
            message
                .severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        })
        .collect();
    assert!(errors.is_empty(), "Vulkan validation errors: {errors:#?}");
}

#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run with --ignored on a Vulkan host"]
fn snapshot_mesh_references_resolve_upload_and_record_draws_validation_clean() {
    let uri = AssetUri::new("models/triangle.gltf").expect("valid source URI");
    let asset_id = AssetId::from_uri(AssetKind::Mesh, &uri);
    let mut cache = AssetCache::new();
    cache
        .insert(asset_id, triangle_import(PrimitiveTopology::Triangles))
        .expect("cache imported glTF mesh");

    let mut scene_snapshot = snapshot(uri.clone(), 0);
    scene_snapshot.meshes.push(scene_snapshot.meshes[0].clone());

    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal runtime scene mesh adapter validation".to_owned(),
            enable_validation: true,
        })
        .expect("create Vulkan instance with validation enabled"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan device");

    let empty_cache: AssetCache<GltfImport> = AssetCache::new();
    assert!(matches!(
        VulkanSceneMeshes::upload(&device, &empty_cache, &scene_snapshot),
        Err(VulkanSceneMeshesError::MissingAsset { .. })
    ));
    let missing_mesh_snapshot = snapshot(uri.clone(), 99);
    assert!(matches!(
        VulkanSceneMeshes::upload(&device, &cache, &missing_mesh_snapshot),
        Err(VulkanSceneMeshesError::MissingMesh { mesh_index: 99, .. })
    ));

    let unsupported_uri = AssetUri::new("models/unsupported.gltf").expect("valid URI");
    let mut unsupported_cache = AssetCache::new();
    unsupported_cache
        .insert(
            AssetId::from_uri(AssetKind::Mesh, &unsupported_uri),
            triangle_import(PrimitiveTopology::Lines),
        )
        .expect("cache unsupported topology fixture");
    let unsupported_snapshot = snapshot(unsupported_uri, 0);
    assert!(matches!(
        VulkanSceneMeshes::upload(&device, &unsupported_cache, &unsupported_snapshot),
        Err(VulkanSceneMeshesError::PrimitiveUpload {
            source: MeshUploadError::UnsupportedTopology(PrimitiveTopology::Lines),
            ..
        })
    ));

    let gpu_scene = VulkanSceneMeshes::upload(&device, &cache, &scene_snapshot)
        .expect("resolve and upload snapshot");
    assert_eq!(gpu_scene.asset_count(), 1);
    assert!(
        gpu_scene
            .material_for_primitive(&uri, 0, 0)
            .expect("resolve unmaterialed primitive")
            .is_none()
    );
    assert!(matches!(
        gpu_scene.material_for_primitive(&uri, 0, 1),
        Err(VulkanSceneMeshesError::MissingPrimitive {
            primitive_index: 1,
            ..
        })
    ));
    assert_eq!(
        gpu_scene.primitive_count(),
        1,
        "shared mesh uploaded more than once"
    );
    let (draw_count, pixels) = record_scene_draws(
        &device,
        &gpu_scene,
        &scene_snapshot,
        None,
        [1.0, 0.2, 0.0, 1.0],
    );
    assert_eq!(
        draw_count, 2,
        "one draw per primitive instance should be recorded"
    );
    let center = &pixels[((16 * 32 + 16) * 4)..][..4];
    assert_eq!(
        center,
        &[231, 178, 0, 255],
        "unexpected scene pixel: {center:?}"
    );
    assert_eq!(&pixels[..4], &[0, 0, 0, 255], "unexpected clear pixel");
    assert!(cache.remove(asset_id).is_some());
    assert_eq!(
        gpu_scene.asset_count(),
        1,
        "adapter lost its pinned asset handle"
    );

    drop(gpu_scene);
    drop(device);

    let validation_errors: Vec<_> = instance
        .validation_messages()
        .expect("read Vulkan validation diagnostics")
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

#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run with --ignored on a Vulkan host"]
fn imported_gltf_scene_instantiates_extracts_and_renders_expected_pixels() {
    let source_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_scene.gltf");
    let imported = GltfImporter::import(&source_path).expect("import deterministic glTF fixture");
    let source_uri = AssetUri::new("fixtures/sample_scene.gltf").expect("valid project asset URI");

    let mut world = World::new();
    let instance = instantiate_default_gltf_scene(&mut world, &imported, source_uri.clone())
        .expect("instantiate imported default scene");
    assert_eq!(instance.nodes.len(), 1);
    world
        .register_component::<PerspectiveCamera>()
        .expect("register perspective camera component");
    let camera_entity = world.spawn().expect("spawn camera entity");
    world
        .insert(
            camera_entity,
            PerspectiveCamera::new(1.0, 1.0, 0.1, 100.0).expect("valid perspective camera"),
        )
        .expect("attach perspective camera");
    world
        .insert(
            camera_entity,
            Transform::new(Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY, Vec3::ONE)
                .expect("valid camera transform"),
        )
        .expect("attach camera transform");
    let scene_snapshot =
        extract_render_scene(&world, camera_entity).expect("extract imported scene snapshot");
    assert_eq!(scene_snapshot.meshes.len(), 1);

    let asset_id = AssetId::from_uri(AssetKind::Mesh, &source_uri);
    let mut cache = AssetCache::new();
    assert_eq!(imported.textures.len(), 1);
    assert_eq!(imported.images.len(), 1);
    cache
        .insert(asset_id, imported.clone())
        .expect("cache imported sample scene");
    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal imported glTF scene render regression".to_owned(),
            enable_validation: true,
        })
        .expect("create Vulkan instance with validation enabled"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan device");
    let mip_image = ImportedImage {
        width: 7,
        height: 3,
        format: z_ideal_assets::ImagePixelFormat::Rgba8,
        pixels: vec![127; 7 * 3 * 4],
    };
    let mip_sampler = ImportedTexture {
        source_image: 0,
        mag_filter: None,
        min_filter: None,
        wrap_s: TextureWrapMode::Repeat,
        wrap_t: TextureWrapMode::MirroredRepeat,
    };
    let mip_texture =
        VulkanTexture2D::new(&device, &mip_image, &mip_sampler, TextureColorSpace::Linear)
            .expect("create 7x3 texture with a complete mip chain");
    assert_eq!(mip_texture.image().config().mip_levels, 3);
    record_texture_upload(&device, &mip_texture);
    let gpu_scene = VulkanSceneMeshes::upload(&device, &cache, &scene_snapshot)
        .expect("upload imported scene meshes");
    let material = gpu_scene
        .material_for_primitive(&source_uri, 0, 0)
        .expect("resolve source primitive material")
        .expect("sample primitive declares a material");
    assert_eq!(material.base_color_texture, Some(0));
    assert_eq!(material.base_color_factor, [1.0, 0.2, 0.0, 1.0]);
    assert_eq!(
        gpu_scene
            .material_texture_bindings_for_primitive(&source_uri, 0, 0)
            .expect("resolve standard material texture semantics"),
        vec![
            GltfMaterialTextureBinding {
                slot: GltfMaterialTextureSlot::BaseColor,
                texture_index: 0,
                color_space: TextureColorSpace::Srgb,
                tex_coord: 0,
            },
            GltfMaterialTextureBinding {
                slot: GltfMaterialTextureSlot::MetallicRoughness,
                texture_index: 0,
                color_space: TextureColorSpace::Linear,
                tex_coord: 0,
            },
            GltfMaterialTextureBinding {
                slot: GltfMaterialTextureSlot::Normal,
                texture_index: 0,
                color_space: TextureColorSpace::Linear,
                tex_coord: 0,
            },
            GltfMaterialTextureBinding {
                slot: GltfMaterialTextureSlot::Occlusion,
                texture_index: 0,
                color_space: TextureColorSpace::Linear,
                tex_coord: 0,
            },
            GltfMaterialTextureBinding {
                slot: GltfMaterialTextureSlot::Emissive,
                texture_index: 0,
                color_space: TextureColorSpace::Srgb,
                tex_coord: 0,
            },
        ]
    );
    let pinned_import = gpu_scene
        .imported_asset(&source_uri)
        .expect("scene adapter pins imported asset");
    let texture_index = material
        .base_color_texture
        .expect("base-color texture reference");
    let texture = VulkanTexture2D::from_import(
        &device,
        pinned_import,
        texture_index,
        TextureColorSpace::Srgb,
    )
    .expect("create Vulkan image and sampler from imported base-color texture");
    record_texture_upload(&device, &texture);
    let (draw_count, pixels) = record_scene_draws(
        &device,
        &gpu_scene,
        &scene_snapshot,
        Some(&texture),
        material.base_color_factor,
    );
    assert_eq!(draw_count, 1, "expected one imported primitive draw");
    let center = &pixels[((16 * 32 + 16) * 4)..][..4];
    assert_eq!(
        center,
        &[231, 178, 0, 255],
        "unexpected rendered sample pixel: {center:?}"
    );
    assert_eq!(&pixels[..4], &[0, 0, 0, 255], "unexpected clear pixel");

    drop(gpu_scene);
    drop(mip_texture);
    drop(texture);
    drop(device);
    let validation_errors: Vec<_> = instance
        .validation_messages()
        .expect("read Vulkan validation diagnostics")
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

#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run with --ignored on a Vulkan host"]
fn depth_only_shadow_pipeline_creation_is_validation_clean() {
    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal depth-only shadow pipeline validation".to_owned(),
            enable_validation: true,
        })
        .expect("create validation-enabled Vulkan instance"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan device");
    let (vertex_artifact, fragment_artifact) =
        compile_shadow_shaders().expect("compile depth-only shadow shaders");
    let vertex_shader =
        VulkanShaderModule::new(&device, vertex_artifact).expect("create shadow vertex module");
    let fragment_shader =
        VulkanShaderModule::new(&device, fragment_artifact).expect("create shadow fragment module");
    let push_constants = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64)];
    let pipeline_layout = VulkanPipelineLayout::new(&device, &[], &push_constants)
        .expect("create depth-only shadow pipeline layout");
    let depth_format = vk::Format::D32_SFLOAT;
    let render_pass = VulkanRenderPass::new(
        &device,
        RenderPassConfig {
            attachments: vec![RenderPassAttachmentConfig {
                format: depth_format,
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
            dependencies: Vec::new(),
        },
    )
    .expect("create depth-only shadow render pass");
    let pipeline_config = GraphicsPipelineConfig {
        vertex_bindings: vec![z_ideal_renderer::MeshGpu::vertex_binding()],
        vertex_attributes: vec![z_ideal_renderer::MeshGpu::vertex_attributes()[0]],
        depth_test_enable: true,
        depth_write_enable: true,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        ..GraphicsPipelineConfig::default()
    };
    let _shadow_renderer = VulkanShadowRenderer::new(
        &pipeline_layout,
        &vertex_shader,
        &fragment_shader,
        &render_pass,
        &pipeline_config,
    )
    .expect("create depth-only shadow pipeline");
    let errors: Vec<_> = instance
        .validation_messages()
        .expect("read shadow validation messages")
        .into_iter()
        .filter(|message| {
            message
                .severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        })
        .collect();
    assert!(errors.is_empty(), "Vulkan validation errors: {errors:#?}");
}
