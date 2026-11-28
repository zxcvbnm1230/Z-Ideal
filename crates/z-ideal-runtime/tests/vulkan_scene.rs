use std::sync::Arc;

use ash::vk;
use z_ideal_assets::{
    AssetCache, AssetId, AssetKind, AssetUri, GltfImport, ImportedMesh, ImportedPrimitive,
    PrimitiveTopology,
};
use z_ideal_math::Mat4;
use z_ideal_renderer::{
    CommandPoolConfig, FenceWaitResult, GraphicsPipelineConfig, ImageConfig, ImageViewConfig,
    MeshUploadError, RenderPassAttachmentConfig, RenderPassAttachmentReference, RenderPassConfig,
    RenderPassSubpassConfig, ShaderCompileConfig, ShaderStage, VulkanCommandPool, VulkanConfig,
    VulkanDevice, VulkanFramebuffer, VulkanGraphicsPipeline, VulkanImage, VulkanInstance,
    VulkanPipelineLayout, VulkanQueue, VulkanRenderPass, VulkanShaderModule, compile_glsl,
};
use z_ideal_runtime::{
    GltfMeshInstance, RenderMeshInstance, RenderSceneSnapshot, VulkanSceneMeshes,
    VulkanSceneMeshesError,
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
                tex_coords_0: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
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

fn record_scene_draws(
    device: &VulkanDevice,
    gpu_scene: &VulkanSceneMeshes<'_>,
    scene: &RenderSceneSnapshot,
) -> usize {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
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
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        flags: vk::ImageCreateFlags::empty(),
    };
    let image = VulkanImage::new(device, image_config).expect("create scene color target");
    let image_view = image
        .create_view(ImageViewConfig::default())
        .expect("create scene color target view");
    let render_pass = VulkanRenderPass::new(
        device,
        RenderPassConfig {
            attachments: vec![RenderPassAttachmentConfig {
                format: image_config.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![RenderPassAttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: Vec::new(),
        },
    )
    .expect("create scene render pass");
    let framebuffer = VulkanFramebuffer::new(
        &render_pass,
        &[&image_view],
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
        1,
    )
    .expect("create scene framebuffer");

    let vertex_shader = VulkanShaderModule::new(
        device,
        compile_glsl(
            "#version 450\nlayout(location=0) in vec3 position;\nlayout(location=1) in vec2 tex_coord;\nlayout(push_constant) uniform Draw { mat4 mvp; } draw_data;\nvoid main() { gl_Position = draw_data.mvp * vec4(position, 1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        )
        .expect("compile scene vertex shader"),
    )
    .expect("create scene vertex shader module");
    let fragment_shader = VulkanShaderModule::new(
        device,
        compile_glsl(
            "#version 450\nlayout(location=0) out vec4 output_color;\nvoid main() { output_color = vec4(1.0, 0.25, 0.0, 1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Fragment),
        )
        .expect("compile scene fragment shader"),
    )
    .expect("create scene fragment shader module");
    let push_constants = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64)];
    let pipeline_layout = VulkanPipelineLayout::new(device, &[], &push_constants)
        .expect("create scene pipeline layout");
    let pipeline = VulkanGraphicsPipeline::new(
        &pipeline_layout,
        &vertex_shader,
        &fragment_shader,
        &render_pass,
        &GraphicsPipelineConfig {
            vertex_bindings: vec![z_ideal_renderer::MeshGpu::vertex_binding()],
            vertex_attributes: z_ideal_renderer::MeshGpu::vertex_attributes().to_vec(),
            ..GraphicsPipelineConfig::default()
        },
    )
    .expect("create scene graphics pipeline");

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
        .begin_render_pass(&render_pass, &framebuffer, &clear)
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
        .record_draws(scene, &mut recording, &pipeline_layout)
        .expect("record runtime scene draws");
    recording.end_render_pass().expect("end scene render pass");
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

    draw_count
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
    assert_eq!(
        gpu_scene.primitive_count(),
        1,
        "shared mesh uploaded more than once"
    );
    assert_eq!(
        record_scene_draws(&device, &gpu_scene, &scene_snapshot),
        2,
        "one draw per primitive instance should be recorded"
    );
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
