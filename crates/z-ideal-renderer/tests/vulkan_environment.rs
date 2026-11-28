use std::sync::Arc;

use ash::vk;
use z_ideal_assets::HdrEnvironmentMap;
use z_ideal_renderer::{
    BufferBarrier, BufferConfig, BufferMemoryLocation, CommandPoolConfig, EnvironmentCubeFace,
    EnvironmentCubeMap, EnvironmentLightingMaps, FenceWaitResult, ImageBarrier, TextureUploadError,
    VulkanBuffer, VulkanCommandPool, VulkanConfig, VulkanDevice, VulkanEnvironmentCube,
    VulkanInstance, VulkanQueue, VulkanTexture2D,
};

/// Verifies float HDRI upload, image transitions, mip generation, and staging lifetime.
#[test]
#[ignore = "requires a Vulkan ICD and Khronos validation layer; run with --ignored on a Vulkan host"]
fn hdr_environment_upload_preserves_linear_radiance_and_is_validation_clean() {
    let instance = Arc::new(
        VulkanInstance::new(&VulkanConfig {
            application_name: "Z Ideal HDR environment upload validation".to_owned(),
            enable_validation: true,
        })
        .expect("create Vulkan instance with validation enabled"),
    );
    let device = VulkanDevice::new(Arc::clone(&instance)).expect("create Vulkan graphics device");
    let environment = HdrEnvironmentMap {
        source_path: "validation-fixture.hdr".into(),
        width: 2,
        height: 1,
        pixels: vec![[1.0, 0.5, 0.25], [4.0, 2.0, 8.0]],
    };
    let texture = match VulkanTexture2D::from_hdr_environment(&device, &environment) {
        Ok(texture) => texture,
        Err(TextureUploadError::UnsupportedFormatFeatures { .. })
        | Err(TextureUploadError::UnsupportedMipFeatures { .. }) => {
            panic!(
                "selected Vulkan device must support linear-filtered RGBA32F environment textures"
            );
        }
        Err(error) => panic!("create HDR environment texture: {error}"),
    };
    let _view = texture
        .create_view()
        .expect("create environment image view");
    let cpu_cube = EnvironmentCubeMap::from_equirectangular(&environment, 1)
        .expect("convert HDRI to six CPU cube faces");
    let cube = VulkanEnvironmentCube::from_equirectangular(&device, &environment, 1)
        .expect("create Vulkan cube environment");
    let _cube_view = cube.create_view().expect("create cube image view");
    let lighting = EnvironmentLightingMaps::from_equirectangular(&environment, 1, 2, 2, 32)
        .expect("generate deterministic irradiance and GGX prefilter maps");
    let irradiance_cube =
        VulkanEnvironmentCube::from_cube_mips(&device, std::slice::from_ref(lighting.irradiance()))
            .expect("create irradiance cube image");
    let _irradiance_view = irradiance_cube
        .create_view()
        .expect("create irradiance cube view");
    let prefiltered_cube = VulkanEnvironmentCube::from_cube_mips(&device, lighting.specular_mips())
        .expect("create GGX-prefiltered specular cube mip chain");
    let _prefiltered_view = prefiltered_cube
        .create_view()
        .expect("create prefiltered cube view");
    assert_eq!(prefiltered_cube.mip_levels(), 2);
    let mut prefiltered_readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: 480,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create specular mip readback buffer");
    let mut cube_readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: 96,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create cube face readback buffer");
    let mut readback = VulkanBuffer::new(
        &device,
        BufferConfig {
            size: 32,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory: BufferMemoryLocation::Readback,
        },
    )
    .expect("create HDR image readback buffer");
    let queue = VulkanQueue::graphics(&device);
    let command_pool =
        VulkanCommandPool::new_for_queue(&device, &queue, CommandPoolConfig::default())
            .expect("create graphics command pool");
    let mut command_buffer = command_pool
        .allocate_primary()
        .expect("allocate primary command buffer");
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
            width: 2,
            height: 1,
            depth: 1,
        })];

    let cube_regions: Vec<_> = (0..6_u32)
        .map(|face| {
            vk::BufferImageCopy::default()
                .buffer_offset(u64::from(face) * 16)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(face)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
        })
        .collect();
    let prefiltered_regions: Vec<_> = (0..2_u32)
        .flat_map(|mip| {
            let size = (2_u32 >> mip).max(1);
            let face_bytes = u64::from(size) * u64::from(size) * 16;
            (0..6_u32).map(move |face| {
                vk::BufferImageCopy::default()
                    .buffer_offset(if mip == 0 {
                        u64::from(face) * face_bytes
                    } else {
                        384 + u64::from(face) * face_bytes
                    })
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .base_array_layer(face)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: size,
                        height: size,
                        depth: 1,
                    })
            })
        })
        .collect();
    let mut recording = command_buffer
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .expect("begin HDR environment upload recording");
    texture
        .record_upload(&mut recording)
        .expect("record HDR environment upload and mip generation");
    cube.record_upload(&mut recording)
        .expect("record six-face cube upload");
    irradiance_cube
        .record_upload(&mut recording)
        .expect("record diffuse irradiance cube upload");
    prefiltered_cube
        .record_upload(&mut recording)
        .expect("record all GGX-prefiltered specular mip uploads");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: texture.image(),
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                source_access: vk::AccessFlags::SHADER_READ,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )
        .expect("transition uploaded HDR image for readback");
    recording
        .copy_image_to_buffer(
            texture.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &region,
        )
        .expect("copy uploaded HDR image to host readback");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: cube.image(),
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                source_access: vk::AccessFlags::SHADER_READ,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            }],
        )
        .expect("transition cube environment for readback");
    recording
        .copy_image_to_buffer(
            cube.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &cube_readback,
            &cube_regions,
        )
        .expect("copy all six cube faces to host readback");
    recording
        .pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: prefiltered_cube.image(),
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                source_access: vk::AccessFlags::SHADER_READ,
                destination_access: vk::AccessFlags::TRANSFER_READ,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 2,
                base_array_layer: 0,
                layer_count: 6,
            }],
        )
        .expect("transition prefiltered mip chain for readback");
    recording
        .copy_image_to_buffer(
            prefiltered_cube.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            &prefiltered_readback,
            &prefiltered_regions,
        )
        .expect("copy both prefiltered specular mip levels");
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
                    buffer: &cube_readback,
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::HOST_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                },
                BufferBarrier {
                    buffer: &prefiltered_readback,
                    source_access: vk::AccessFlags::TRANSFER_WRITE,
                    destination_access: vk::AccessFlags::HOST_READ,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                },
            ],
            &[],
        )
        .expect("make HDR readback visible to host");
    let executable = recording
        .end()
        .expect("finish HDR environment command buffer");
    let mut submission = queue
        .submit(executable, &[], &[], &[])
        .expect("submit HDR environment upload");
    assert_eq!(
        submission
            .wait(u64::MAX)
            .expect("wait for HDR environment upload"),
        FenceWaitResult::Signaled
    );
    drop(submission);

    let mut bytes = vec![0_u8; 32];
    readback
        .read(0, &mut bytes)
        .expect("read back uploaded HDR radiance");
    let samples: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect();
    assert_eq!(samples, [1.0, 0.5, 0.25, 1.0, 4.0, 2.0, 8.0, 1.0]);
    let mut cube_bytes = vec![0_u8; 96];
    cube_readback
        .read(0, &mut cube_bytes)
        .expect("read back all six cube faces");
    let cube_samples: Vec<f32> = cube_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect();
    let expected_cube_samples: Vec<f32> = EnvironmentCubeFace::ALL
        .into_iter()
        .flat_map(|face| cpu_cube.face(face)[0])
        .collect();
    assert_eq!(cube_samples, expected_cube_samples);
    let mut prefiltered_bytes = vec![0_u8; 480];
    prefiltered_readback
        .read(0, &mut prefiltered_bytes)
        .expect("read back GGX-prefiltered mip chain");
    let prefiltered_samples: Vec<f32> = prefiltered_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect();
    let expected_prefiltered: Vec<f32> = lighting
        .specular_mips()
        .iter()
        .flat_map(|mip| {
            EnvironmentCubeFace::ALL
                .into_iter()
                .flat_map(|face| mip.face(face).iter().flatten().copied())
        })
        .collect();
    assert_eq!(prefiltered_samples, expected_prefiltered);
    assert!(
        instance
            .validation_messages()
            .expect("read Vulkan validation diagnostics")
            .iter()
            .all(|message| {
                !message
                    .severity
                    .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            }),
        "HDR environment upload emitted Vulkan validation errors"
    );
}
