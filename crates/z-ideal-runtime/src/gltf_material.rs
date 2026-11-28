//! Runtime-side descriptor contract for standard glTF metallic-roughness materials.
use ash::vk;
use thiserror::Error;
use z_ideal_assets::{
    GltfImport, ImagePixelFormat, ImportedImage, ImportedMaterial, ImportedTexture,
    MaterialAlphaMode, TextureWrapMode,
};
use z_ideal_renderer::{
    BufferDescriptorWrite, CommandBufferRecording, DescriptorResourceLease,
    DescriptorResourceWrite, GraphicsPipelineConfig, ImageDescriptorWrite, TextureColorSpace,
    TextureUploadError, VulkanBuffer, VulkanDescriptorSet, VulkanDevice, VulkanError,
    VulkanImageView, VulkanSampler, VulkanTexture2D,
};

/// The exact std140-compatible material payload consumed by the glTF PBR shaders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GltfMaterialUniform {
    /// Base color RGBA multiplier.
    pub base_color_factor: [f32; 4],
    /// Emissive RGB multiplier.
    pub emissive_factor: [f32; 3],
    /// Alpha cutoff used by MASK materials.
    pub alpha_cutoff: f32,
    /// Metallic multiplier.
    pub metallic_factor: f32,
    /// Roughness multiplier.
    pub roughness_factor: f32,
    /// Normal-map XY scale.
    pub normal_scale: f32,
    /// Occlusion strength.
    pub occlusion_strength: f32,
    /// OPAQUE=0, MASK=1, BLEND=2.
    pub alpha_mode: u32,
    /// Whether the material is double-sided.
    pub double_sided: u32,
    /// UV set selected by the base-color texture.
    pub base_color_tex_coord: u32,
    /// UV set selected by the metallic-roughness texture.
    pub metallic_roughness_tex_coord: u32,
    /// UV set selected by the normal texture.
    pub normal_tex_coord: u32,
    /// UV set selected by the occlusion texture.
    pub occlusion_tex_coord: u32,
    /// UV set selected by the emissive texture.
    pub emissive_tex_coord: u32,
}

impl From<&ImportedMaterial> for GltfMaterialUniform {
    fn from(material: &ImportedMaterial) -> Self {
        Self {
            base_color_factor: material.base_color_factor,
            emissive_factor: material.emissive_factor,
            alpha_cutoff: material.alpha_cutoff,
            metallic_factor: material.metallic_factor,
            roughness_factor: material.roughness_factor,
            normal_scale: material.normal_scale,
            occlusion_strength: material.occlusion_strength,
            alpha_mode: match material.alpha_mode {
                MaterialAlphaMode::Opaque => 0,
                MaterialAlphaMode::Mask => 1,
                MaterialAlphaMode::Blend => 2,
            },
            double_sided: u32::from(material.double_sided),
            base_color_tex_coord: material.base_color_tex_coord,
            metallic_roughness_tex_coord: material.metallic_roughness_tex_coord,
            normal_tex_coord: material.normal_tex_coord,
            occlusion_tex_coord: material.occlusion_tex_coord,
            emissive_tex_coord: material.emissive_tex_coord,
        }
    }
}

fn validate_imported_material(material: &ImportedMaterial) -> Result<(), GltfMaterialError> {
    let finite = material
        .base_color_factor
        .iter()
        .chain(material.emissive_factor.iter())
        .chain([
            &material.metallic_factor,
            &material.roughness_factor,
            &material.normal_scale,
            &material.occlusion_strength,
            &material.alpha_cutoff,
        ])
        .all(|value| value.is_finite());
    let base_color_valid = material
        .base_color_factor
        .iter()
        .all(|value| (0.0..=1.0).contains(value));
    let scalar_ranges_valid = (0.0..=1.0).contains(&material.metallic_factor)
        && (0.0..=1.0).contains(&material.roughness_factor)
        && material.normal_scale >= 0.0
        && (0.0..=1.0).contains(&material.occlusion_strength)
        && (0.0..=1.0).contains(&material.alpha_cutoff);
    let emissive_valid = material.emissive_factor.iter().all(|value| *value >= 0.0);
    let texture_coordinates_valid = [
        material.base_color_tex_coord,
        material.metallic_roughness_tex_coord,
        material.normal_tex_coord,
        material.occlusion_tex_coord,
        material.emissive_tex_coord,
    ]
    .into_iter()
    .all(|set| set <= 1);
    if !finite
        || !base_color_valid
        || !scalar_ranges_valid
        || !emissive_valid
        || !texture_coordinates_valid
    {
        return Err(GltfMaterialError::InvalidMaterialValues);
    }
    Ok(())
}

fn validate_scene_uniform(scene: GltfPbrSceneUniform) -> Result<(), GltfMaterialError> {
    let values = scene
        .camera_position
        .into_iter()
        .chain(scene.light_direction)
        .chain([scene.light_intensity])
        .chain(scene.light_color)
        .chain(scene.ambient_color)
        .chain([scene.environment_intensity, scene.environment_max_lod]);
    if values.clone().any(|value| !value.is_finite())
        || scene.light_direction.iter().all(|value| *value == 0.0)
        || scene.light_intensity < 0.0
        || scene.light_color.iter().any(|value| *value < 0.0)
        || scene.ambient_color.iter().any(|value| *value < 0.0)
        || scene.environment_intensity < 0.0
        || scene.environment_max_lod < 0.0
    {
        return Err(GltfMaterialError::InvalidSceneValues);
    }
    Ok(())
}

impl GltfMaterialUniform {
    /// Encodes six 16-byte std140 vectors (96 bytes), without relying on Rust struct padding.
    #[must_use]
    pub fn to_std140_bytes(self) -> [u8; 96] {
        let vectors = [
            self.base_color_factor,
            [
                self.emissive_factor[0],
                self.emissive_factor[1],
                self.emissive_factor[2],
                self.alpha_cutoff,
            ],
            [
                self.metallic_factor,
                self.roughness_factor,
                self.normal_scale,
                self.occlusion_strength,
            ],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
        ];
        let mut bytes = [0_u8; 96];
        for (index, value) in vectors.into_iter().flatten().enumerate() {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        bytes[48..52].copy_from_slice(&self.alpha_mode.to_ne_bytes());
        bytes[52..56].copy_from_slice(&self.double_sided.to_ne_bytes());
        for (index, value) in [
            self.base_color_tex_coord,
            self.metallic_roughness_tex_coord,
            self.normal_tex_coord,
            self.occlusion_tex_coord,
        ]
        .into_iter()
        .enumerate()
        {
            let offset = 64 + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        bytes[80..84].copy_from_slice(&self.emissive_tex_coord.to_ne_bytes());
        bytes
    }
}

/// Scene-wide values consumed by the PBR fragment shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GltfPbrSceneUniform {
    /// World-space camera position.
    pub camera_position: [f32; 3],
    /// Direction rays travel from the directional light.
    pub light_direction: [f32; 3],
    /// Directional-light radiant intensity multiplier.
    pub light_intensity: f32,
    /// Linear-light RGB color of the directional light.
    pub light_color: [f32; 3],
    /// Linear-light ambient irradiance approximation retained as an optional fill term.
    pub ambient_color: [f32; 3],
    /// Non-negative multiplier applied to diffuse and specular image-based lighting.
    pub environment_intensity: f32,
    /// Maximum roughness-prefiltered specular mip level.
    pub environment_max_lod: f32,
}

impl GltfPbrSceneUniform {
    /// Encodes five std140 vec4 values (80 bytes).
    #[must_use]
    pub fn to_std140_bytes(self) -> [u8; 80] {
        let vectors = [
            [
                self.camera_position[0],
                self.camera_position[1],
                self.camera_position[2],
                1.0,
            ],
            [
                self.light_direction[0],
                self.light_direction[1],
                self.light_direction[2],
                self.light_intensity,
            ],
            [
                self.light_color[0],
                self.light_color[1],
                self.light_color[2],
                1.0,
            ],
            [
                self.ambient_color[0],
                self.ambient_color[1],
                self.ambient_color[2],
                1.0,
            ],
            [
                self.environment_intensity,
                self.environment_max_lod,
                0.0,
                0.0,
            ],
        ];
        let mut bytes = [0_u8; 80];
        for (index, value) in vectors.into_iter().flatten().enumerate() {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        bytes
    }
}

/// Validates and applies the fixed-function state required by one glTF material.
///
/// Alpha-blended instances must be submitted back-to-front by the caller. The
/// function disables depth writes for BLEND materials and configures conventional
/// source-alpha blending; it does not reorder scene instances.
pub fn apply_gltf_material_pipeline_state(
    config: &mut GraphicsPipelineConfig,
    material: &ImportedMaterial,
    color_attachment_count: usize,
) -> Result<(), GltfMaterialError> {
    validate_imported_material(material)?;
    apply_gltf_material_pipeline_variant(
        config,
        material.alpha_mode == MaterialAlphaMode::Blend,
        material.double_sided,
        color_attachment_count,
    )
}

pub(crate) fn apply_gltf_material_pipeline_variant(
    config: &mut GraphicsPipelineConfig,
    alpha_blend: bool,
    double_sided: bool,
    color_attachment_count: usize,
) -> Result<(), GltfMaterialError> {
    if color_attachment_count == 0 {
        return Err(GltfMaterialError::MissingColorAttachment);
    }
    // glTF uses counter-clockwise front faces in right-handed object space. With
    // the engine's positive-height Vulkan viewport and non-Y-flipped projection,
    // framebuffer winding is clockwise.
    config.front_face = vk::FrontFace::CLOCKWISE;
    config.cull_mode = if double_sided {
        vk::CullModeFlags::NONE
    } else {
        vk::CullModeFlags::BACK
    };
    if alpha_blend {
        config.depth_write_enable = false;
        let blend = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        config.color_blend_attachments = vec![blend; color_attachment_count];
    } else {
        config.color_blend_attachments.clear();
    }
    Ok(())
}

/// Writes the PBR scene uniform and image-based lighting resources to descriptor set 1.
///
/// Binding 0 is the 80-byte std140 scene block; bindings 1 and 2 are combined
/// cube samplers for diffuse irradiance and roughness-prefiltered specular radiance.
/// The caller owns the views, images, samplers, buffer, and descriptor set. The
/// returned lease borrows all referenced resources for the duration of GPU use.
pub fn write_gltf_pbr_scene_descriptor<'set, 'resource, 'image, 'pool, 'layout, 'device>(
    descriptor_set: &'set mut VulkanDescriptorSet<'pool, 'layout, 'device>,
    uniform_buffer: &'resource mut VulkanBuffer<'device>,
    scene: GltfPbrSceneUniform,
    irradiance_view: &'image VulkanImageView<'_, 'device>,
    irradiance_sampler: &'resource VulkanSampler<'device>,
    specular_view: &'image VulkanImageView<'_, 'device>,
    specular_sampler: &'resource VulkanSampler<'device>,
) -> Result<
    DescriptorResourceLease<'set, 'resource, 'image, 'pool, 'layout, 'device>,
    GltfMaterialError,
> {
    validate_scene_uniform(scene)?;
    let bindings = &descriptor_set.layout().config().bindings;
    if bindings.len() != 3
        || bindings.iter().any(|binding| {
            let expected_type = if binding.binding == 0 {
                vk::DescriptorType::UNIFORM_BUFFER
            } else {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            };
            binding.binding > 2
                || binding.descriptor_type != expected_type
                || binding.descriptor_count != 1
                || !binding.stage_flags.contains(vk::ShaderStageFlags::FRAGMENT)
        })
    {
        return Err(GltfMaterialError::DescriptorLayoutMismatch);
    }
    if uniform_buffer.size() < 80 {
        return Err(GltfMaterialError::UniformBufferTooSmall);
    }
    uniform_buffer.write(0, &scene.to_std140_bytes())?;
    Ok(descriptor_set.write_descriptors(&[
        DescriptorResourceWrite::Buffer(BufferDescriptorWrite {
            binding: 0,
            array_element: 0,
            buffer: uniform_buffer,
            offset: 0,
            range: 80,
        }),
        DescriptorResourceWrite::Image(ImageDescriptorWrite {
            binding: 1,
            array_element: 0,
            view: irradiance_view,
            sampler: Some(irradiance_sampler),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }),
        DescriptorResourceWrite::Image(ImageDescriptorWrite {
            binding: 2,
            array_element: 0,
            view: specular_view,
            sampler: Some(specular_sampler),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }),
    ])?)
}

/// Owns the five GPU texture resources used by one glTF material.
///
/// Textures are uploaded independently by semantic role. If one source texture is
/// referenced by both sRGB and linear slots, it is decoded/uploaded twice with the
/// required Vulkan formats. Missing slots receive explicit glTF-neutral 1x1 images.
pub struct GltfMaterialTextureSet<'device> {
    base_color: VulkanTexture2D<'device>,
    metallic_roughness: VulkanTexture2D<'device>,
    normal: VulkanTexture2D<'device>,
    occlusion: VulkanTexture2D<'device>,
    emissive: VulkanTexture2D<'device>,
}

impl<'device> GltfMaterialTextureSet<'device> {
    /// Creates the five semantic GPU textures for one imported material.
    pub fn from_import(
        device: &'device VulkanDevice,
        import: &GltfImport,
        material: &ImportedMaterial,
    ) -> Result<Self, GltfMaterialError> {
        validate_imported_material(material)?;
        for tex_coord in [
            material.base_color_tex_coord,
            material.metallic_roughness_tex_coord,
            material.normal_tex_coord,
            material.occlusion_tex_coord,
            material.emissive_tex_coord,
        ] {
            if tex_coord > 1 {
                return Err(GltfMaterialError::UnsupportedTextureCoordinateSet(
                    tex_coord,
                ));
            }
        }
        Ok(Self {
            base_color: semantic_texture(
                device,
                import,
                material.base_color_texture,
                TextureColorSpace::Srgb,
                [255, 255, 255, 255],
            )?,
            metallic_roughness: semantic_texture(
                device,
                import,
                material.metallic_roughness_texture,
                TextureColorSpace::Linear,
                [255, 255, 255, 255],
            )?,
            normal: semantic_texture(
                device,
                import,
                material.normal_texture,
                TextureColorSpace::Linear,
                [128, 128, 255, 255],
            )?,
            occlusion: semantic_texture(
                device,
                import,
                material.occlusion_texture,
                TextureColorSpace::Linear,
                [255, 255, 255, 255],
            )?,
            emissive: semantic_texture(
                device,
                import,
                material.emissive_texture,
                TextureColorSpace::Srgb,
                [0, 0, 0, 255],
            )?,
        })
    }

    /// Records all five image uploads and shader-read transitions.
    ///
    /// If any recording operation fails, discard the command buffer and this texture
    /// set; a partially recorded upload cannot safely be retried.
    pub fn record_upload<'recording>(
        &'recording self,
        recording: &mut CommandBufferRecording<'recording, '_, 'device>,
    ) -> Result<(), GltfMaterialError> {
        self.base_color.record_upload(recording)?;
        self.metallic_roughness.record_upload(recording)?;
        self.normal.record_upload(recording)?;
        self.occlusion.record_upload(recording)?;
        self.emissive.record_upload(recording)?;
        Ok(())
    }

    /// Creates borrowed image views for all five semantic textures.
    pub fn create_views<'resource>(
        &'resource self,
    ) -> Result<GltfMaterialTextureViews<'resource, 'device>, GltfMaterialError> {
        Ok(GltfMaterialTextureViews {
            base_color: self.base_color.create_view()?,
            metallic_roughness: self.metallic_roughness.create_view()?,
            normal: self.normal.create_view()?,
            occlusion: self.occlusion.create_view()?,
            emissive: self.emissive.create_view()?,
        })
    }
}

fn semantic_texture<'device>(
    device: &'device VulkanDevice,
    import: &GltfImport,
    texture_index: Option<usize>,
    color_space: TextureColorSpace,
    fallback_pixel: [u8; 4],
) -> Result<VulkanTexture2D<'device>, GltfMaterialError> {
    if let Some(texture_index) = texture_index {
        return Ok(VulkanTexture2D::from_import(
            device,
            import,
            texture_index,
            color_space,
        )?);
    }
    let image = ImportedImage {
        width: 1,
        height: 1,
        format: ImagePixelFormat::Rgba8,
        pixels: fallback_pixel.to_vec(),
    };
    let sampler = ImportedTexture {
        source_image: 0,
        mag_filter: None,
        min_filter: None,
        wrap_s: TextureWrapMode::Repeat,
        wrap_t: TextureWrapMode::Repeat,
    };
    Ok(VulkanTexture2D::new(device, &image, &sampler, color_space)?)
}

/// Borrowed views for the five textures owned by GltfMaterialTextureSet.
pub struct GltfMaterialTextureViews<'resource, 'device> {
    base_color: VulkanImageView<'resource, 'device>,
    metallic_roughness: VulkanImageView<'resource, 'device>,
    normal: VulkanImageView<'resource, 'device>,
    occlusion: VulkanImageView<'resource, 'device>,
    emissive: VulkanImageView<'resource, 'device>,
}

impl<'resource, 'device> GltfMaterialTextureViews<'resource, 'device> {
    /// Returns descriptor resources while borrowing both views and owning texture set.
    pub fn descriptor_resources<'borrow>(
        &'borrow self,
        textures: &'borrow GltfMaterialTextureSet<'device>,
    ) -> GltfMaterialTextureResources<'borrow, 'resource, 'device> {
        GltfMaterialTextureResources {
            base_color: GltfMaterialTextureResource {
                view: &self.base_color,
                sampler: textures.base_color.sampler(),
            },
            metallic_roughness: GltfMaterialTextureResource {
                view: &self.metallic_roughness,
                sampler: textures.metallic_roughness.sampler(),
            },
            normal: GltfMaterialTextureResource {
                view: &self.normal,
                sampler: textures.normal.sampler(),
            },
            occlusion: GltfMaterialTextureResource {
                view: &self.occlusion,
                sampler: textures.occlusion.sampler(),
            },
            emissive: GltfMaterialTextureResource {
                view: &self.emissive,
                sampler: textures.emissive.sampler(),
            },
        }
    }
}

/// One already-created texture view and sampler for a material texture semantic.
///
/// The caller supplies all five resources, including semantically neutral fallback
/// images for absent glTF slots. Fallback pixel values and their color spaces are
/// explicit asset/resource policy, not hidden renderer behavior.
#[derive(Clone, Copy)]
pub struct GltfMaterialTextureResource<'resource, 'image, 'device> {
    /// Image view sampled by the material shader.
    pub view: &'resource VulkanImageView<'image, 'device>,
    /// Sampler configured from the corresponding glTF texture or chosen fallback.
    pub sampler: &'resource VulkanSampler<'device>,
}

/// The five texture descriptors in fixed shader binding order.
#[derive(Clone, Copy)]
pub struct GltfMaterialTextureResources<'resource, 'image, 'device> {
    /// Binding 1: base color, sRGB.
    pub base_color: GltfMaterialTextureResource<'resource, 'image, 'device>,
    /// Binding 2: metallic (B) and roughness (G), linear.
    pub metallic_roughness: GltfMaterialTextureResource<'resource, 'image, 'device>,
    /// Binding 3: tangent-space normal, linear.
    pub normal: GltfMaterialTextureResource<'resource, 'image, 'device>,
    /// Binding 4: occlusion, linear.
    pub occlusion: GltfMaterialTextureResource<'resource, 'image, 'device>,
    /// Binding 5: emissive, sRGB.
    pub emissive: GltfMaterialTextureResource<'resource, 'image, 'device>,
}

impl<'resource, 'image, 'device> GltfMaterialTextureResources<'resource, 'image, 'device> {
    fn ordered(self) -> [GltfMaterialTextureResource<'resource, 'image, 'device>; 5] {
        [
            self.base_color,
            self.metallic_roughness,
            self.normal,
            self.occlusion,
            self.emissive,
        ]
    }
}

/// Material descriptor creation and update failures.
#[derive(Debug, Error)]
pub enum GltfMaterialError {
    /// A descriptor layout does not match the documented material or PBR scene contract.
    #[error(
        "glTF material descriptor layout must contain binding 0 uniform-buffer and bindings 1-5 combined-image-sampler, all fragment-visible"
    )]
    DescriptorLayoutMismatch,
    /// A supplied uniform buffer is smaller than its std140 payload.
    #[error(
        "glTF material and scene uniform buffers must contain at least their std140 payload sizes"
    )]
    UniformBufferTooSmall,
    /// Material factors, ranges, or texture-coordinate selections are invalid.
    #[error("glTF material factors or texture-coordinate selections are invalid")]
    InvalidMaterialValues,
    /// Scene camera or lighting values are non-finite or outside supported ranges.
    #[error("glTF PBR scene camera and lighting values are invalid")]
    InvalidSceneValues,
    /// A material pipeline requires at least one color attachment.
    #[error("glTF material pipeline requires at least one color attachment")]
    MissingColorAttachment,
    /// A material texture selects a UV set beyond the uploaded UV0/UV1 attributes.
    #[error(
        "glTF material texture coordinate set {0} is unsupported; only UV0 and UV1 are uploaded"
    )]
    UnsupportedTextureCoordinateSet(u32),
    /// Imported texture decoding or Vulkan texture allocation/upload failed.
    #[error(transparent)]
    TextureUpload(#[from] TextureUploadError),
    /// Vulkan buffer update or descriptor update failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

/// Writes one glTF material's factors and five explicit texture resources to a descriptor set.
///
/// Binding contract:
/// - binding 0: one fragment-visible UNIFORM_BUFFER containing the 96-byte GltfMaterialUniform.
/// - bindings 1..=5: one fragment-visible COMBINED_IMAGE_SAMPLER each, ordered
///   base-color, metallic-roughness, normal, occlusion, emissive.
///
/// The caller owns the descriptor pool, set layout, descriptor set, uniform buffer,
/// texture images, views, and samplers. The returned lease borrows the set and all
/// referenced resources. Keep it alive through command submission; renderer command
/// submissions retain its lease through fence completion.
pub fn write_gltf_material_descriptors<'set, 'resource, 'image, 'pool, 'layout, 'device>(
    descriptor_set: &'set mut VulkanDescriptorSet<'pool, 'layout, 'device>,
    uniform_buffer: &'resource mut VulkanBuffer<'device>,
    material: &ImportedMaterial,
    textures: GltfMaterialTextureResources<'resource, 'image, 'device>,
) -> Result<
    DescriptorResourceLease<'set, 'resource, 'image, 'pool, 'layout, 'device>,
    GltfMaterialError,
> {
    validate_imported_material(material)?;
    let layout = descriptor_set.layout().config();
    if layout.bindings.len() != 6
        || layout.bindings.iter().any(|binding| {
            let expected_type = if binding.binding == 0 {
                vk::DescriptorType::UNIFORM_BUFFER
            } else {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            };
            binding.binding > 5
                || binding.descriptor_type != expected_type
                || binding.descriptor_count != 1
                || !binding.stage_flags.contains(vk::ShaderStageFlags::FRAGMENT)
        })
    {
        return Err(GltfMaterialError::DescriptorLayoutMismatch);
    }
    if uniform_buffer.size() < 80 {
        return Err(GltfMaterialError::UniformBufferTooSmall);
    }

    let uniform = GltfMaterialUniform::from(material).to_std140_bytes();
    uniform_buffer.write(0, &uniform)?;

    let mut writes = Vec::with_capacity(6);
    writes.push(DescriptorResourceWrite::Buffer(BufferDescriptorWrite {
        binding: 0,
        array_element: 0,
        buffer: uniform_buffer,
        offset: 0,
        range: 80,
    }));
    for (index, texture) in textures.ordered().into_iter().enumerate() {
        writes.push(DescriptorResourceWrite::Image(ImageDescriptorWrite {
            binding: u32::try_from(index + 1)
                .map_err(|_| GltfMaterialError::DescriptorLayoutMismatch)?,
            array_element: 0,
            view: texture.view,
            sampler: Some(texture.sampler),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }));
    }
    Ok(descriptor_set.write_descriptors(&writes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use z_ideal_assets::MaterialAlphaMode;

    fn material() -> ImportedMaterial {
        ImportedMaterial {
            name: None,
            base_color_factor: [0.1, 0.2, 0.3, 0.4],
            metallic_factor: 0.5,
            roughness_factor: 0.6,
            base_color_texture: None,
            base_color_tex_coord: 0,
            metallic_roughness_texture: None,
            metallic_roughness_tex_coord: 0,
            normal_texture: None,
            normal_tex_coord: 0,
            normal_scale: 0.7,
            occlusion_texture: None,
            occlusion_tex_coord: 0,
            occlusion_strength: 0.8,
            emissive_texture: None,
            emissive_tex_coord: 0,
            emissive_factor: [0.9, 1.0, 0.11],
            alpha_mode: MaterialAlphaMode::Mask,
            alpha_cutoff: 0.33,
            double_sided: true,
        }
    }

    #[test]
    fn std140_payload_is_exactly_six_aligned_vectors() {
        let uniform = GltfMaterialUniform::from(&material());
        let bytes = uniform.to_std140_bytes();
        assert_eq!(bytes.len(), 96);
        assert_eq!(
            f32::from_ne_bytes(bytes[0..4].try_into().expect("float")),
            0.1
        );
        assert_eq!(
            f32::from_ne_bytes(bytes[16..20].try_into().expect("float")),
            0.9
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[48..52].try_into().expect("uint")),
            1
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[52..56].try_into().expect("uint")),
            1
        );
    }

    #[test]
    fn rejects_invalid_material_and_scene_uniform_values() {
        let mut source = material();
        source.metallic_factor = 1.5;
        assert!(matches!(
            validate_imported_material(&source),
            Err(GltfMaterialError::InvalidMaterialValues)
        ));
        source = material();
        source.normal_tex_coord = 2;
        assert!(matches!(
            validate_imported_material(&source),
            Err(GltfMaterialError::InvalidMaterialValues)
        ));
        let mut scene = GltfPbrSceneUniform {
            camera_position: [0.0, 0.0, 1.0],
            light_direction: [0.0, 0.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0; 3],
            ambient_color: [0.1; 3],
            environment_intensity: 1.0,
            environment_max_lod: 2.0,
        };
        assert!(matches!(
            validate_scene_uniform(scene),
            Err(GltfMaterialError::InvalidSceneValues)
        ));
        scene.light_direction = [0.0, -1.0, 0.0];
        scene.light_intensity = f32::NAN;
        assert!(matches!(
            validate_scene_uniform(scene),
            Err(GltfMaterialError::InvalidSceneValues)
        ));
    }

    #[test]
    fn scene_uniform_is_exactly_five_std140_vectors() {
        let scene = GltfPbrSceneUniform {
            camera_position: [1.0, 2.0, 3.0],
            light_direction: [0.0, -1.0, 0.0],
            light_intensity: 4.0,
            light_color: [5.0, 6.0, 7.0],
            ambient_color: [0.1, 0.2, 0.3],
            environment_intensity: 0.75,
            environment_max_lod: 4.0,
        };
        let bytes = scene.to_std140_bytes();
        assert_eq!(bytes.len(), 80);
        assert_eq!(
            f32::from_ne_bytes(bytes[28..32].try_into().expect("float")),
            4.0
        );
        assert_eq!(
            f32::from_ne_bytes(bytes[48..52].try_into().expect("float")),
            0.1
        );
        assert_eq!(
            f32::from_ne_bytes(bytes[64..68].try_into().expect("float")),
            0.75
        );
        assert_eq!(
            f32::from_ne_bytes(bytes[68..72].try_into().expect("float")),
            4.0
        );
    }

    #[test]
    fn material_pipeline_state_respects_alpha_and_double_sided_flags() {
        let mut source = material();
        let mut config = GraphicsPipelineConfig::default();
        apply_gltf_material_pipeline_state(&mut config, &source, 1).expect("mask state");
        assert_eq!(config.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(config.front_face, vk::FrontFace::CLOCKWISE);
        assert!(config.color_blend_attachments.is_empty());
        source.alpha_mode = MaterialAlphaMode::Blend;
        config.depth_write_enable = true;
        apply_gltf_material_pipeline_state(&mut config, &source, 1).expect("blend state");
        assert!(!config.depth_write_enable);
        assert!(config.color_blend_attachments[0].blend_enable == vk::TRUE);
        source.double_sided = false;
        apply_gltf_material_pipeline_state(&mut config, &source, 1).expect("single-sided state");
        assert_eq!(config.cull_mode, vk::CullModeFlags::BACK);
        assert!(matches!(
            apply_gltf_material_pipeline_state(&mut config, &source, 0),
            Err(GltfMaterialError::MissingColorAttachment)
        ));
    }

    #[test]
    fn maps_all_material_factors_and_alpha_modes() {
        let mut source = material();
        assert_eq!(GltfMaterialUniform::from(&source).alpha_mode, 1);
        source.alpha_mode = MaterialAlphaMode::Opaque;
        assert_eq!(GltfMaterialUniform::from(&source).alpha_mode, 0);
        source.alpha_mode = MaterialAlphaMode::Blend;
        assert_eq!(GltfMaterialUniform::from(&source).alpha_mode, 2);
        assert_eq!(GltfMaterialUniform::from(&source).double_sided, 1);
    }
}
