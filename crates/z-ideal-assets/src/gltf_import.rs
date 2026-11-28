//! glTF 2.0 / GLB import into renderer-independent, validated CPU asset data.
use gltf::{buffer, image as gltf_image};
use image::{DynamicImage, ImageDecoder as _, ImageReader, Limits};

/// Engine-facing primitive topology, independent of the glTF parser representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

/// Engine-facing material alpha mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialAlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Engine-facing decoded image channel layout and component precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePixelFormat {
    R8,
    Rg8,
    Rgb8,
    Rgba8,
    R16,
    Rg16,
    Rgb16,
    Rgba16,
    Rgb32Float,
    Rgba32Float,
}

/// Texture magnification sampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureMagFilter {
    Nearest,
    Linear,
}

/// Texture minification and mipmap sampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureMinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

/// Texture coordinate wrapping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureWrapMode {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
}
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use crate::AssetUri;

/// Complete CPU-side import result for one glTF 2.0 document.
#[derive(Debug, Clone)]
pub struct GltfImport {
    /// Source document path.
    pub source_path: PathBuf,
    /// Imported mesh definitions and primitives.
    pub meshes: Vec<ImportedMesh>,
    /// All source nodes in document-index order, including nodes not selected by a scene.
    pub nodes: Vec<ImportedNode>,
    /// Scene roots referencing indices in `nodes`.
    pub scenes: Vec<ImportedScene>,
    /// Index of the document's default scene, when one is declared.
    pub default_scene: Option<usize>,
    /// Explicitly defined materials.
    pub materials: Vec<ImportedMaterial>,
    /// Sampler/texture-to-image mappings.
    pub textures: Vec<ImportedTexture>,
    /// Decoded source images in their original channel/bit-depth representation.
    pub images: Vec<ImportedImage>,
    /// External buffer and image source URIs referenced by the document.
    pub dependency_uris: Vec<String>,
}
/// Imported mesh, preserving its primitive boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMesh {
    /// Source mesh index.
    pub source_index: usize,
    /// Optional source name.
    pub name: Option<String>,
    /// Imported primitives.
    pub primitives: Vec<ImportedPrimitive>,
}
/// Imported glTF node with a validated local TRS transform and source mesh reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedNode {
    /// Source node index; stable within this source document.
    pub source_index: usize,
    /// Optional source name.
    pub name: Option<String>,
    /// Local translation in engine coordinates.
    pub translation: [f32; 3],
    /// Local rotation as an XYZW quaternion.
    pub rotation_xyzw: [f32; 4],
    /// Local scale.
    pub scale: [f32; 3],
    /// Referenced imported mesh index.
    pub mesh_index: Option<usize>,
    /// Child node indices in source order.
    pub children: Vec<usize>,
}

/// Imported glTF scene and its root node indices.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedScene {
    /// Source scene index.
    pub source_index: usize,
    /// Optional source name.
    pub name: Option<String>,
    /// Root nodes in source order.
    pub root_nodes: Vec<usize>,
}

/// Imported mesh primitive data.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedPrimitive {
    /// glTF primitive topology.
    pub mode: PrimitiveTopology,
    /// Position attribute in engine coordinates (glTF and engine both use right-handed +Y-up).
    pub positions: Vec<[f32; 3]>,
    /// Optional vertex normals.
    pub normals: Option<Vec<[f32; 3]>>,
    /// Optional first texture coordinate set.
    pub tex_coords_0: Option<Vec<[f32; 2]>>,
    /// Triangle/line/point indices; non-indexed primitives receive sequential indices.
    pub indices: Vec<u32>,
    /// Explicit material index, if supplied by the document.
    pub material_index: Option<usize>,
}
/// Imported metallic-roughness material and its texture references.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMaterial {
    /// Optional source material name.
    pub name: Option<String>,
    /// Base color RGBA factor.
    pub base_color_factor: [f32; 4],
    /// Metallic factor.
    pub metallic_factor: f32,
    /// Roughness factor.
    pub roughness_factor: f32,
    /// Base color texture index.
    pub base_color_texture: Option<usize>,
    /// Texture coordinate set used by the base color texture.
    pub base_color_tex_coord: u32,
    /// Metallic-roughness texture index.
    pub metallic_roughness_texture: Option<usize>,
    /// Texture coordinate set used by the metallic-roughness texture.
    pub metallic_roughness_tex_coord: u32,
    /// Normal texture index.
    pub normal_texture: Option<usize>,
    /// Texture coordinate set used by the normal texture.
    pub normal_tex_coord: u32,
    /// Scalar applied to sampled tangent-space normal X and Y components.
    pub normal_scale: f32,
    /// Occlusion texture index.
    pub occlusion_texture: Option<usize>,
    /// Texture coordinate set used by the occlusion texture.
    pub occlusion_tex_coord: u32,
    /// Strength of the occlusion effect.
    pub occlusion_strength: f32,
    /// Emissive texture index.
    pub emissive_texture: Option<usize>,
    /// Texture coordinate set used by the emissive texture.
    pub emissive_tex_coord: u32,
    /// Emissive RGB factor.
    pub emissive_factor: [f32; 3],
    /// Alpha rendering mode.
    pub alpha_mode: MaterialAlphaMode,
    /// Alpha cutoff used by mask materials.
    pub alpha_cutoff: f32,
    /// Whether both triangle sides should be rendered.
    pub double_sided: bool,
}
/// Imported glTF texture sampling state and source image index.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedTexture {
    /// Source image index.
    pub source_image: usize,
    /// Magnification filter.
    pub mag_filter: Option<TextureMagFilter>,
    /// Minification filter.
    pub min_filter: Option<TextureMinFilter>,
    /// Horizontal wrapping mode.
    pub wrap_s: TextureWrapMode,
    /// Vertical wrapping mode.
    pub wrap_t: TextureWrapMode,
}
/// Decoded image pixels. Pixel channel layout is described by format.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedImage {
    /// Image width in texels.
    pub width: u32,
    /// Image height in texels.
    pub height: u32,
    /// Original decoded pixel representation.
    pub format: ImagePixelFormat,
    /// Contiguous row-major pixel bytes.
    pub pixels: Vec<u8>,
}
/// glTF import failures.
#[derive(Debug, thiserror::Error)]
pub enum GltfImportError {
    /// File access or glTF parsing/importing failed.
    #[error("could not import glTF asset {path}: {message}")]
    Import { path: PathBuf, message: String },
    /// An external resource URI is absolute, escapes the source directory, or cannot be resolved.
    #[error("glTF {path} has unsafe external resource URI {uri:?}: {reason}")]
    InvalidExternalUri {
        path: PathBuf,
        uri: String,
        reason: String,
    },
    /// The imported document references a missing buffer/image.
    #[error("glTF {path} references missing {kind} index {index}")]
    MissingResource {
        path: PathBuf,
        kind: &'static str,
        index: usize,
    },
    /// Imported image metadata does not match its decoded pixel storage.
    #[error("glTF {path} image {index} has invalid decoded pixel storage")]
    InvalidImageData { path: PathBuf, index: usize },
    /// A mesh primitive is malformed.
    #[error("glTF {path} mesh {mesh} primitive {primitive}: {reason}")]
    InvalidPrimitive {
        path: PathBuf,
        mesh: usize,
        primitive: usize,
        reason: String,
    },
    /// An imported material contains non-finite or out-of-range core glTF factors.
    #[error("glTF {path} material {material} has invalid material factors: {reason}")]
    InvalidMaterial {
        path: PathBuf,
        material: usize,
        reason: String,
    },
    /// A material refers to a texture outside the imported texture table.
    #[error("glTF {path} material {material} refers to missing texture {texture}")]
    MissingTexture {
        path: PathBuf,
        material: usize,
        texture: usize,
    },
    /// A node has an invalid reference, transform, parent count, or cyclic relationship.
    #[error("glTF {path} node {node}: {reason}")]
    InvalidNode {
        path: PathBuf,
        node: usize,
        reason: String,
    },
    /// A scene refers to a node outside the imported node table.
    #[error("glTF {path} scene {scene} refers to missing root node {node}")]
    MissingSceneNode {
        path: PathBuf,
        scene: usize,
        node: usize,
    },
    /// The default scene index is outside the imported scene table.
    #[error("glTF {path} declares missing default scene {scene}")]
    MissingDefaultScene { path: PathBuf, scene: usize },
    /// A source or external resource exceeded the configured import byte budget.
    #[error(
        "glTF {path} resource {resource:?} is {actual_bytes} bytes; configured limit is {limit_bytes} bytes"
    )]
    ResourceLimitExceeded {
        path: PathBuf,
        resource: String,
        actual_bytes: u64,
        limit_bytes: u64,
    },
    /// An image exceeded the configured strict width or height limit.
    #[error("glTF {path} image {index} is {width}x{height}; maximum is {max_width}x{max_height}")]
    ImageDimensionsExceeded {
        path: PathBuf,
        index: usize,
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },
    /// Encoded image data could not be read or decoded.
    #[error("glTF {path} image {index} could not be decoded: {message}")]
    ImageDecode {
        path: PathBuf,
        index: usize,
        message: String,
    },
}

/// Resource limits for glTF source documents, external resources, and decoded images.
///
/// Source and external resource reads are bounded, and decoded image output size is
/// checked before allocating pixel storage. Canonical path validation still has a
/// symlink/file-replacement time-of-check/time-of-use race. Embedded data URIs and
/// GLB binary chunks are bounded by max_source_file_bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GltfImportOptions {
    /// Maximum source glTF or GLB file size in bytes.
    pub max_source_file_bytes: u64,
    /// Maximum size of any one external buffer or image file in bytes.
    pub max_external_resource_bytes: u64,
    /// Maximum combined size of distinct external buffer and image files in bytes.
    pub max_total_external_resource_bytes: u64,
    /// Maximum decoded image width in pixels.
    pub max_image_width: u32,
    /// Maximum decoded image height in pixels.
    pub max_image_height: u32,
    /// Maximum decoded pixel storage for one image in bytes.
    pub max_decoded_image_bytes: u64,
    /// Maximum combined decoded pixel storage across all images in bytes.
    pub max_total_decoded_image_bytes: u64,
}

impl Default for GltfImportOptions {
    fn default() -> Self {
        Self {
            max_source_file_bytes: 64 * 1024 * 1024,
            max_external_resource_bytes: 256 * 1024 * 1024,
            max_total_external_resource_bytes: 512 * 1024 * 1024,
            max_image_width: 16_384,
            max_image_height: 16_384,
            max_decoded_image_bytes: 256 * 1024 * 1024,
            max_total_decoded_image_bytes: 512 * 1024 * 1024,
        }
    }
}

/// glTF 2.0 importer.
#[derive(Debug, Default, Clone, Copy)]
pub struct GltfImporter;
impl GltfImporter {
    /// Imports a .gltf or .glb file, including referenced buffers and decoded images,
    /// using the default GltfImportOptions byte budgets.
    pub fn import(path: impl AsRef<Path>) -> Result<GltfImport, GltfImportError> {
        Self::import_with_options(path, GltfImportOptions::default())
    }

    /// Imports a glTF/GLB document while enforcing source and external-resource byte budgets.
    pub fn import_with_options(
        path: impl AsRef<Path>,
        options: GltfImportOptions,
    ) -> Result<GltfImport, GltfImportError> {
        let path = path.as_ref();
        let source_path = path.to_path_buf();
        let source_file = std::fs::File::open(path).map_err(|error| GltfImportError::Import {
            path: source_path.clone(),
            message: format!("could not open source document: {error}"),
        })?;
        let mut bounded_reader = source_file.take(options.max_source_file_bytes.saturating_add(1));
        let mut source_data = Vec::new();
        bounded_reader
            .read_to_end(&mut source_data)
            .map_err(|error| GltfImportError::Import {
                path: source_path.clone(),
                message: format!("could not read source document: {error}"),
            })?;
        let source_bytes = u64::try_from(source_data.len()).map_err(|_| {
            GltfImportError::ResourceLimitExceeded {
                path: source_path.clone(),
                resource: "source document".to_owned(),
                actual_bytes: u64::MAX,
                limit_bytes: options.max_source_file_bytes,
            }
        })?;
        if source_bytes > options.max_source_file_bytes {
            return Err(GltfImportError::ResourceLimitExceeded {
                path: source_path.clone(),
                resource: "source document".to_owned(),
                actual_bytes: source_bytes,
                limit_bytes: options.max_source_file_bytes,
            });
        }
        let parsed =
            gltf::Gltf::from_slice(&source_data).map_err(|error| GltfImportError::Import {
                path: source_path.clone(),
                message: error.to_string(),
            })?;
        let source_directory = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()
            .map_err(|error| GltfImportError::Import {
                path: source_path.clone(),
                message: format!("could not resolve source directory: {error}"),
            })?;
        let gltf::Gltf { document, blob } = parsed;
        let mut external_dependencies = BTreeSet::new();
        let mut external_resources = BTreeMap::new();
        for buffer in document.buffers() {
            if let buffer::Source::Uri(uri) = buffer.source() {
                if let Some((uri, resolved)) =
                    validate_external_uri(&source_directory, &source_path, uri)?
                {
                    external_dependencies.insert(uri.clone());
                    external_resources.insert(resolved, uri);
                }
            }
        }
        for image in document.images() {
            if let gltf_image::Source::Uri { uri, .. } = image.source() {
                if let Some((uri, resolved)) =
                    validate_external_uri(&source_directory, &source_path, uri)?
                {
                    external_dependencies.insert(uri.clone());
                    external_resources.insert(resolved, uri);
                }
            }
        }
        validate_external_resource_budgets(&source_path, &external_resources, options)?;
        let mut external_read_budget = ExternalReadBudget::default();
        let buffers = import_buffers_with_limits(
            &document,
            &source_directory,
            &source_path,
            blob,
            options,
            &mut external_read_budget,
        )?;
        let imported_images = import_images_with_limits(
            &document,
            &source_directory,
            &source_path,
            &buffers,
            options,
            &mut external_read_budget,
        )?;

        let imported_textures = document
            .textures()
            .map(|texture| {
                let source_image = texture.source().index();
                if source_image >= imported_images.len() {
                    return Err(GltfImportError::MissingResource {
                        path: source_path.clone(),
                        kind: "image",
                        index: source_image,
                    });
                }
                let sampler = texture.sampler();
                Ok(ImportedTexture {
                    source_image,
                    mag_filter: sampler.mag_filter().map(|filter| match filter {
                        gltf::texture::MagFilter::Nearest => TextureMagFilter::Nearest,
                        gltf::texture::MagFilter::Linear => TextureMagFilter::Linear,
                    }),
                    min_filter: sampler.min_filter().map(map_min_filter),
                    wrap_s: map_wrap_mode(sampler.wrap_s()),
                    wrap_t: map_wrap_mode(sampler.wrap_t()),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let imported_materials = document
            .materials()
            .map(|material| {
                let pbr = material.pbr_metallic_roughness();
                let base_color_texture = pbr.base_color_texture();
                let metallic_roughness_texture = pbr.metallic_roughness_texture();
                let normal_texture = material.normal_texture();
                let occlusion_texture = material.occlusion_texture();
                let emissive_texture = material.emissive_texture();
                let texture_indices = [
                    base_color_texture
                        .as_ref()
                        .map(|info| info.texture().index()),
                    metallic_roughness_texture
                        .as_ref()
                        .map(|info| info.texture().index()),
                    normal_texture.as_ref().map(|info| info.texture().index()),
                    occlusion_texture
                        .as_ref()
                        .map(|info| info.texture().index()),
                    emissive_texture.as_ref().map(|info| info.texture().index()),
                ];
                for texture in texture_indices.into_iter().flatten() {
                    if texture >= imported_textures.len() {
                        return Err(GltfImportError::MissingTexture {
                            path: source_path.clone(),
                            material: material.index().unwrap_or(usize::MAX),
                            texture,
                        });
                    }
                }
                let imported = ImportedMaterial {
                    name: material.name().map(str::to_owned),
                    base_color_factor: pbr.base_color_factor(),
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    base_color_texture: texture_indices[0],
                    base_color_tex_coord: base_color_texture
                        .as_ref()
                        .map_or(0, |info| info.tex_coord()),
                    metallic_roughness_texture: texture_indices[1],
                    metallic_roughness_tex_coord: metallic_roughness_texture
                        .map_or(0, |info| info.tex_coord()),
                    normal_texture: texture_indices[2],
                    normal_tex_coord: normal_texture.as_ref().map_or(0, |info| info.tex_coord()),
                    normal_scale: normal_texture.as_ref().map_or(1.0, |info| info.scale()),
                    occlusion_texture: texture_indices[3],
                    occlusion_tex_coord: occlusion_texture
                        .as_ref()
                        .map_or(0, |info| info.tex_coord()),
                    occlusion_strength: occlusion_texture
                        .as_ref()
                        .map_or(1.0, |info| info.strength()),
                    emissive_texture: texture_indices[4],
                    emissive_tex_coord: emissive_texture
                        .as_ref()
                        .map_or(0, |info| info.tex_coord()),
                    emissive_factor: material.emissive_factor(),
                    alpha_mode: match material.alpha_mode() {
                        gltf::material::AlphaMode::Opaque => MaterialAlphaMode::Opaque,
                        gltf::material::AlphaMode::Mask => MaterialAlphaMode::Mask,
                        gltf::material::AlphaMode::Blend => MaterialAlphaMode::Blend,
                    },
                    alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
                    double_sided: material.double_sided(),
                };
                validate_imported_material(
                    &source_path,
                    material.index().unwrap_or(usize::MAX),
                    &imported,
                )?;
                Ok(imported)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut imported_meshes = Vec::new();
        for source_mesh in document.meshes() {
            let mesh_index = source_mesh.index();
            let mut primitives = Vec::new();
            for (primitive_index, primitive) in source_mesh.primitives().enumerate() {
                let reader = primitive
                    .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
                let positions = reader
                    .read_positions()
                    .ok_or_else(|| {
                        invalid_primitive(
                            &source_path,
                            mesh_index,
                            primitive_index,
                            "POSITION attribute is required",
                        )
                    })?
                    .collect::<Vec<_>>();
                if positions.is_empty()
                    || positions.iter().flatten().any(|value| !value.is_finite())
                {
                    return Err(invalid_primitive(
                        &source_path,
                        mesh_index,
                        primitive_index,
                        "positions must be non-empty and finite",
                    ));
                }
                let normals = reader
                    .read_normals()
                    .map(|values| values.collect::<Vec<_>>());
                if normals.as_ref().is_some_and(|values| {
                    values.len() != positions.len()
                        || values.iter().flatten().any(|value| !value.is_finite())
                }) {
                    return Err(invalid_primitive(
                        &source_path,
                        mesh_index,
                        primitive_index,
                        "normals must be finite and match the position count",
                    ));
                }
                let tex_coords_0 = reader
                    .read_tex_coords(0)
                    .map(|values| values.into_f32().collect::<Vec<_>>());
                if tex_coords_0.as_ref().is_some_and(|values| {
                    values.len() != positions.len()
                        || values.iter().flatten().any(|value| !value.is_finite())
                }) {
                    return Err(invalid_primitive(
                        &source_path,
                        mesh_index,
                        primitive_index,
                        "texture coordinates must be finite and match the position count",
                    ));
                }
                let mode = map_primitive_mode(primitive.mode());
                let indices = if let Some(values) = reader.read_indices() {
                    values.into_u32().collect::<Vec<_>>()
                } else {
                    let vertex_count = u32::try_from(positions.len()).map_err(|_| {
                        invalid_primitive(
                            &source_path,
                            mesh_index,
                            primitive_index,
                            "non-indexed primitive exceeds the 32-bit vertex index range",
                        )
                    })?;
                    (0..vertex_count).collect()
                };
                if indices
                    .iter()
                    .any(|index| *index as usize >= positions.len())
                {
                    return Err(invalid_primitive(
                        &source_path,
                        mesh_index,
                        primitive_index,
                        "an index is outside the vertex array",
                    ));
                }
                if !valid_index_count(mode, indices.len()) {
                    return Err(invalid_primitive(
                        &source_path,
                        mesh_index,
                        primitive_index,
                        "index count is invalid for the primitive topology",
                    ));
                }
                let material_index = primitive.material().index();
                if material_index.is_some_and(|index| index >= imported_materials.len()) {
                    return Err(invalid_primitive(
                        &source_path,
                        mesh_index,
                        primitive_index,
                        "material index is outside the imported material table",
                    ));
                }
                primitives.push(ImportedPrimitive {
                    mode,
                    positions,
                    normals,
                    tex_coords_0,
                    indices,
                    material_index,
                });
            }
            imported_meshes.push(ImportedMesh {
                source_index: mesh_index,
                name: source_mesh.name().map(str::to_owned),
                primitives,
            });
        }

        let imported_nodes = document
            .nodes()
            .map(|node| {
                let (translation, rotation_xyzw, scale) = node.transform().decomposed();
                let mesh_index = node.mesh().map(|mesh| mesh.index());
                if mesh_index.is_some_and(|index| index >= imported_meshes.len()) {
                    return Err(GltfImportError::InvalidNode {
                        path: source_path.clone(),
                        node: node.index(),
                        reason: "mesh reference is outside the imported mesh table".to_owned(),
                    });
                }
                let children = node
                    .children()
                    .map(|child| child.index())
                    .collect::<Vec<_>>();
                if translation
                    .iter()
                    .chain(rotation_xyzw.iter())
                    .chain(scale.iter())
                    .any(|value| !value.is_finite())
                {
                    return Err(GltfImportError::InvalidNode {
                        path: source_path.clone(),
                        node: node.index(),
                        reason: "transform components must be finite".to_owned(),
                    });
                }
                let rotation_length = rotation_xyzw.iter().map(|value| value * value).sum::<f32>();
                if !rotation_length.is_finite() || rotation_length <= f32::EPSILON {
                    return Err(GltfImportError::InvalidNode {
                        path: source_path.clone(),
                        node: node.index(),
                        reason: "rotation quaternion must have non-zero finite length".to_owned(),
                    });
                }
                Ok(ImportedNode {
                    source_index: node.index(),
                    name: node.name().map(str::to_owned),
                    translation,
                    rotation_xyzw,
                    scale,
                    mesh_index,
                    children,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let node_parents = validate_node_graph(&source_path, &imported_nodes)?;
        let imported_scenes = document
            .scenes()
            .map(|scene| {
                let root_nodes = scene.nodes().map(|node| node.index()).collect::<Vec<_>>();
                for node in &root_nodes {
                    if *node >= imported_nodes.len() {
                        return Err(GltfImportError::MissingSceneNode {
                            path: source_path.clone(),
                            scene: scene.index(),
                            node: *node,
                        });
                    }
                    if node_parents[*node].is_some() {
                        return Err(GltfImportError::InvalidNode {
                            path: source_path.clone(),
                            node: *node,
                            reason: "scene root is also a child of another node".to_owned(),
                        });
                    }
                }
                Ok(ImportedScene {
                    source_index: scene.index(),
                    name: scene.name().map(str::to_owned),
                    root_nodes,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let default_scene = document.default_scene().map(|scene| scene.index());
        if default_scene.is_some_and(|index| index >= imported_scenes.len()) {
            return Err(GltfImportError::MissingDefaultScene {
                path: source_path.clone(),
                scene: default_scene.unwrap_or(usize::MAX),
            });
        }

        let dependency_uris = external_dependencies.into_iter().collect();

        Ok(GltfImport {
            source_path,
            meshes: imported_meshes,
            nodes: imported_nodes,
            scenes: imported_scenes,
            default_scene,
            materials: imported_materials,
            textures: imported_textures,
            images: imported_images,
            dependency_uris,
        })
    }
}

fn validate_node_graph(
    path: &Path,
    nodes: &[ImportedNode],
) -> Result<Vec<Option<usize>>, GltfImportError> {
    let mut parent_of = vec![None; nodes.len()];
    for node in nodes {
        for child in &node.children {
            if *child >= nodes.len() {
                return Err(GltfImportError::InvalidNode {
                    path: path.to_path_buf(),
                    node: node.source_index,
                    reason: format!("child node index {child} is outside the node table"),
                });
            }
            if parent_of[*child].replace(node.source_index).is_some() {
                return Err(GltfImportError::InvalidNode {
                    path: path.to_path_buf(),
                    node: *child,
                    reason: "node has more than one parent".to_owned(),
                });
            }
        }
    }

    // Follow the single-parent links iteratively so adversarially deep input cannot
    // exhaust the call stack during cycle validation.
    let mut state = vec![0_u8; nodes.len()];
    for start in 0..nodes.len() {
        let mut path_nodes = Vec::new();
        let mut cursor = Some(start);
        while let Some(index) = cursor {
            match state[index] {
                1 => {
                    return Err(GltfImportError::InvalidNode {
                        path: path.to_path_buf(),
                        node: index,
                        reason: "node hierarchy contains a cycle".to_owned(),
                    });
                }
                2 => break,
                _ => {}
            }
            state[index] = 1;
            path_nodes.push(index);
            cursor = parent_of[index];
        }
        for index in path_nodes {
            state[index] = 2;
        }
    }
    Ok(parent_of)
}

fn valid_index_count(mode: PrimitiveTopology, count: usize) -> bool {
    match mode {
        PrimitiveTopology::Points => count >= 1,
        PrimitiveTopology::Lines => count >= 2 && count % 2 == 0,
        PrimitiveTopology::LineLoop | PrimitiveTopology::LineStrip => count >= 2,
        PrimitiveTopology::Triangles => count >= 3 && count % 3 == 0,
        PrimitiveTopology::TriangleStrip | PrimitiveTopology::TriangleFan => count >= 3,
    }
}

#[derive(Default)]
struct ExternalReadBudget {
    resource_sizes: BTreeMap<PathBuf, u64>,
    total_bytes: u64,
}

fn import_buffers_with_limits(
    document: &gltf::Document,
    source_directory: &Path,
    source_path: &Path,
    blob: Option<Vec<u8>>,
    options: GltfImportOptions,
    external_budget: &mut ExternalReadBudget,
) -> Result<Vec<buffer::Data>, GltfImportError> {
    let mut blob = blob;
    let mut imported_buffers = Vec::new();
    for source_buffer in document.buffers() {
        let mut data = match source_buffer.source() {
            buffer::Source::Bin => {
                buffer::Data(blob.take().ok_or_else(|| GltfImportError::Import {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "buffer {} requires a missing GLB BIN chunk",
                        source_buffer.index()
                    ),
                })?)
            }
            buffer::Source::Uri(uri) if uri.starts_with("data:") => {
                buffer::Data::from_source(buffer::Source::Uri(uri), None).map_err(|error| {
                    GltfImportError::Import {
                        path: source_path.to_path_buf(),
                        message: format!(
                            "could not decode buffer {} data URI: {error}",
                            source_buffer.index()
                        ),
                    }
                })?
            }
            buffer::Source::Uri(uri) => {
                let (_, resolved) = validate_external_uri(source_directory, source_path, uri)?
                    .ok_or_else(|| GltfImportError::InvalidExternalUri {
                        path: source_path.to_path_buf(),
                        uri: uri.to_owned(),
                        reason: "external buffer URI resolved as embedded data".to_owned(),
                    })?;
                buffer::Data(read_external_resource(
                    &resolved,
                    uri,
                    source_path,
                    options,
                    external_budget,
                )?)
            }
        };
        let actual_length = data.0.len();
        if actual_length < source_buffer.length() {
            return Err(GltfImportError::Import {
                path: source_path.to_path_buf(),
                message: format!(
                    "buffer {} contains {} bytes but declares {}",
                    source_buffer.index(),
                    actual_length,
                    source_buffer.length()
                ),
            });
        }
        while data.0.len() % 4 != 0 {
            data.0.push(0);
        }
        imported_buffers.push(data);
    }
    Ok(imported_buffers)
}

fn import_images_with_limits(
    document: &gltf::Document,
    source_directory: &Path,
    source_path: &Path,
    buffers: &[buffer::Data],
    options: GltfImportOptions,
    external_budget: &mut ExternalReadBudget,
) -> Result<Vec<ImportedImage>, GltfImportError> {
    let mut imported_images = Vec::new();
    let mut total_decoded_bytes = 0_u64;
    for source_image in document.images() {
        let index = source_image.index();
        let encoded = match source_image.source() {
            gltf_image::Source::Uri { uri, .. } if uri.starts_with("data:") => {
                decode_image_data_uri(uri, source_path, index)?
            }
            gltf_image::Source::Uri { uri, .. } => {
                let (_, resolved) = validate_external_uri(source_directory, source_path, uri)?
                    .ok_or_else(|| GltfImportError::ImageDecode {
                        path: source_path.to_path_buf(),
                        index,
                        message: "external image URI resolved as embedded data".to_owned(),
                    })?;
                read_external_resource(&resolved, uri, source_path, options, external_budget)?
            }
            gltf_image::Source::View { view, .. } => {
                let buffer_index = view.buffer().index();
                let data =
                    buffers
                        .get(buffer_index)
                        .ok_or_else(|| GltfImportError::MissingResource {
                            path: source_path.to_path_buf(),
                            kind: "image buffer",
                            index: buffer_index,
                        })?;
                let begin = view.offset();
                let end = begin.checked_add(view.length()).ok_or_else(|| {
                    GltfImportError::ImageDecode {
                        path: source_path.to_path_buf(),
                        index,
                        message: "image buffer-view range overflowed".to_owned(),
                    }
                })?;
                data.0
                    .get(begin..end)
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| GltfImportError::ImageDecode {
                        path: source_path.to_path_buf(),
                        index,
                        message: "image buffer-view range is outside its buffer".to_owned(),
                    })?
            }
        };

        let reader = ImageReader::new(Cursor::new(encoded))
            .with_guessed_format()
            .map_err(|error| GltfImportError::ImageDecode {
                path: source_path.to_path_buf(),
                index,
                message: error.to_string(),
            })?;
        let mut decoder = reader
            .into_decoder()
            .map_err(|error| GltfImportError::ImageDecode {
                path: source_path.to_path_buf(),
                index,
                message: error.to_string(),
            })?;
        let (width, height) = decoder.dimensions();
        if width == 0
            || height == 0
            || width > options.max_image_width
            || height > options.max_image_height
        {
            return Err(GltfImportError::ImageDimensionsExceeded {
                path: source_path.to_path_buf(),
                index,
                width,
                height,
                max_width: options.max_image_width,
                max_height: options.max_image_height,
            });
        }
        let decoded_bytes = decoder.total_bytes();
        if decoded_bytes > options.max_decoded_image_bytes {
            return Err(GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: format!("decoded image {index}"),
                actual_bytes: decoded_bytes,
                limit_bytes: options.max_decoded_image_bytes,
            });
        }
        total_decoded_bytes = total_decoded_bytes
            .checked_add(decoded_bytes)
            .ok_or_else(|| GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: "combined decoded images".to_owned(),
                actual_bytes: u64::MAX,
                limit_bytes: options.max_total_decoded_image_bytes,
            })?;
        if total_decoded_bytes > options.max_total_decoded_image_bytes {
            return Err(GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: "combined decoded images".to_owned(),
                actual_bytes: total_decoded_bytes,
                limit_bytes: options.max_total_decoded_image_bytes,
            });
        }

        let mut decoder_limits = Limits::default();
        decoder_limits.max_image_width = Some(options.max_image_width);
        decoder_limits.max_image_height = Some(options.max_image_height);
        decoder_limits.max_alloc = Some(options.max_decoded_image_bytes);
        decoder
            .set_limits(decoder_limits)
            .map_err(|error| GltfImportError::ImageDecode {
                path: source_path.to_path_buf(),
                index,
                message: error.to_string(),
            })?;
        let decoded =
            DynamicImage::from_decoder(decoder).map_err(|error| GltfImportError::ImageDecode {
                path: source_path.to_path_buf(),
                index,
                message: error.to_string(),
            })?;
        let format =
            map_image_color_type(decoded.color()).ok_or_else(|| GltfImportError::ImageDecode {
                path: source_path.to_path_buf(),
                index,
                message: "decoded image channel format is unsupported".to_owned(),
            })?;
        let pixels = decoded.into_bytes();
        if expected_image_bytes(format, width, height) != Some(pixels.len()) {
            return Err(GltfImportError::InvalidImageData {
                path: source_path.to_path_buf(),
                index,
            });
        }
        imported_images.push(ImportedImage {
            width,
            height,
            format,
            pixels,
        });
    }
    Ok(imported_images)
}

fn decode_image_data_uri(
    uri: &str,
    source_path: &Path,
    image_index: usize,
) -> Result<Vec<u8>, GltfImportError> {
    let (metadata, encoded) = uri
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
        .ok_or_else(|| GltfImportError::ImageDecode {
            path: source_path.to_path_buf(),
            index: image_index,
            message: "malformed image data URI".to_owned(),
        })?;
    if !metadata.split(';').any(|part| part == "base64") {
        return Err(GltfImportError::ImageDecode {
            path: source_path.to_path_buf(),
            index: image_index,
            message: "only base64-encoded image data URIs are supported".to_owned(),
        });
    }
    base64::decode(encoded).map_err(|error| GltfImportError::ImageDecode {
        path: source_path.to_path_buf(),
        index: image_index,
        message: format!("invalid base64 image data URI: {error}"),
    })
}

fn read_external_resource(
    path: &Path,
    uri: &str,
    source_path: &Path,
    options: GltfImportOptions,
    budget: &mut ExternalReadBudget,
) -> Result<Vec<u8>, GltfImportError> {
    let file = std::fs::File::open(path).map_err(|error| GltfImportError::InvalidExternalUri {
        path: source_path.to_path_buf(),
        uri: uri.to_owned(),
        reason: format!("could not open external resource: {error}"),
    })?;
    let mut reader = file.take(options.max_external_resource_bytes.saturating_add(1));
    let mut encoded = Vec::new();
    reader
        .read_to_end(&mut encoded)
        .map_err(|error| GltfImportError::InvalidExternalUri {
            path: source_path.to_path_buf(),
            uri: uri.to_owned(),
            reason: format!("could not read external resource: {error}"),
        })?;
    let actual_bytes =
        u64::try_from(encoded.len()).map_err(|_| GltfImportError::ResourceLimitExceeded {
            path: source_path.to_path_buf(),
            resource: uri.to_owned(),
            actual_bytes: u64::MAX,
            limit_bytes: options.max_external_resource_bytes,
        })?;
    if actual_bytes > options.max_external_resource_bytes {
        return Err(GltfImportError::ResourceLimitExceeded {
            path: source_path.to_path_buf(),
            resource: uri.to_owned(),
            actual_bytes,
            limit_bytes: options.max_external_resource_bytes,
        });
    }

    if let Some(previous_size) = budget.resource_sizes.get(path) {
        if *previous_size != actual_bytes {
            return Err(GltfImportError::InvalidExternalUri {
                path: source_path.to_path_buf(),
                uri: uri.to_owned(),
                reason: "external resource changed size during import".to_owned(),
            });
        }
    } else {
        budget.total_bytes = budget
            .total_bytes
            .checked_add(actual_bytes)
            .ok_or_else(|| GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: "combined external resources".to_owned(),
                actual_bytes: u64::MAX,
                limit_bytes: options.max_total_external_resource_bytes,
            })?;
        if budget.total_bytes > options.max_total_external_resource_bytes {
            return Err(GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: "combined external resources".to_owned(),
                actual_bytes: budget.total_bytes,
                limit_bytes: options.max_total_external_resource_bytes,
            });
        }
        budget
            .resource_sizes
            .insert(path.to_path_buf(), actual_bytes);
    }
    Ok(encoded)
}

fn map_image_color_type(color: image::ColorType) -> Option<ImagePixelFormat> {
    Some(match color {
        image::ColorType::L8 => ImagePixelFormat::R8,
        image::ColorType::La8 => ImagePixelFormat::Rg8,
        image::ColorType::Rgb8 => ImagePixelFormat::Rgb8,
        image::ColorType::Rgba8 => ImagePixelFormat::Rgba8,
        image::ColorType::L16 => ImagePixelFormat::R16,
        image::ColorType::La16 => ImagePixelFormat::Rg16,
        image::ColorType::Rgb16 => ImagePixelFormat::Rgb16,
        image::ColorType::Rgba16 => ImagePixelFormat::Rgba16,
        image::ColorType::Rgb32F => ImagePixelFormat::Rgb32Float,
        image::ColorType::Rgba32F => ImagePixelFormat::Rgba32Float,
        _ => return None,
    })
}

fn validate_external_uri(
    source_directory: &Path,
    source_path: &Path,
    uri: &str,
) -> Result<Option<(String, PathBuf)>, GltfImportError> {
    if uri
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
    {
        return Ok(None);
    }
    let decoded = urlencoding::decode(uri)
        .map_err(|error| GltfImportError::InvalidExternalUri {
            path: source_path.to_path_buf(),
            uri: uri.to_owned(),
            reason: format!("invalid percent encoding: {error}"),
        })?
        .into_owned();
    if decoded.contains(':') || decoded.contains('?') || decoded.contains('#') {
        return Err(GltfImportError::InvalidExternalUri {
            path: source_path.to_path_buf(),
            uri: uri.to_owned(),
            reason: "absolute paths, URI schemes, queries, and fragments are not supported"
                .to_owned(),
        });
    }
    let asset_uri =
        AssetUri::new(decoded).map_err(|error| GltfImportError::InvalidExternalUri {
            path: source_path.to_path_buf(),
            uri: uri.to_owned(),
            reason: error.to_string(),
        })?;
    let resolved = source_directory
        .join(asset_uri.as_str())
        .canonicalize()
        .map_err(|error| GltfImportError::InvalidExternalUri {
            path: source_path.to_path_buf(),
            uri: uri.to_owned(),
            reason: format!("external resource cannot be resolved: {error}"),
        })?;
    if !resolved.starts_with(source_directory) {
        return Err(GltfImportError::InvalidExternalUri {
            path: source_path.to_path_buf(),
            uri: uri.to_owned(),
            reason: "resolved external resource escapes the source directory".to_owned(),
        });
    }
    Ok(Some((asset_uri.to_string(), resolved)))
}

fn validate_external_resource_budgets(
    source_path: &Path,
    resources: &BTreeMap<PathBuf, String>,
    options: GltfImportOptions,
) -> Result<(), GltfImportError> {
    let mut total_bytes = 0_u64;
    for (path, uri) in resources {
        let actual_bytes = std::fs::metadata(path)
            .map_err(|error| GltfImportError::InvalidExternalUri {
                path: source_path.to_path_buf(),
                uri: uri.clone(),
                reason: format!("external resource cannot be inspected: {error}"),
            })?
            .len();
        if actual_bytes > options.max_external_resource_bytes {
            return Err(GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: uri.clone(),
                actual_bytes,
                limit_bytes: options.max_external_resource_bytes,
            });
        }
        total_bytes = total_bytes.checked_add(actual_bytes).ok_or_else(|| {
            GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: "combined external resources".to_owned(),
                actual_bytes: u64::MAX,
                limit_bytes: options.max_total_external_resource_bytes,
            }
        })?;
        if total_bytes > options.max_total_external_resource_bytes {
            return Err(GltfImportError::ResourceLimitExceeded {
                path: source_path.to_path_buf(),
                resource: "combined external resources".to_owned(),
                actual_bytes: total_bytes,
                limit_bytes: options.max_total_external_resource_bytes,
            });
        }
    }
    Ok(())
}

fn map_primitive_mode(mode: gltf::mesh::Mode) -> PrimitiveTopology {
    match mode {
        gltf::mesh::Mode::Points => PrimitiveTopology::Points,
        gltf::mesh::Mode::Lines => PrimitiveTopology::Lines,
        gltf::mesh::Mode::LineLoop => PrimitiveTopology::LineLoop,
        gltf::mesh::Mode::LineStrip => PrimitiveTopology::LineStrip,
        gltf::mesh::Mode::Triangles => PrimitiveTopology::Triangles,
        gltf::mesh::Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
        gltf::mesh::Mode::TriangleFan => PrimitiveTopology::TriangleFan,
    }
}

fn validate_imported_material(
    path: &Path,
    material: usize,
    value: &ImportedMaterial,
) -> Result<(), GltfImportError> {
    let in_unit_interval = |factor: f32| factor.is_finite() && (0.0..=1.0).contains(&factor);
    if !value.base_color_factor.into_iter().all(in_unit_interval) {
        return Err(GltfImportError::InvalidMaterial {
            path: path.to_path_buf(),
            material,
            reason: "base color factor components must be finite and within [0, 1]".to_owned(),
        });
    }
    if !in_unit_interval(value.metallic_factor) || !in_unit_interval(value.roughness_factor) {
        return Err(GltfImportError::InvalidMaterial {
            path: path.to_path_buf(),
            material,
            reason: "metallic and roughness factors must be finite and within [0, 1]".to_owned(),
        });
    }
    if !value.emissive_factor.into_iter().all(in_unit_interval) {
        return Err(GltfImportError::InvalidMaterial {
            path: path.to_path_buf(),
            material,
            reason: "emissive factor components must be finite and within [0, 1]".to_owned(),
        });
    }
    if !value.normal_scale.is_finite()
        || !value.occlusion_strength.is_finite()
        || !(0.0..=1.0).contains(&value.occlusion_strength)
        || !value.alpha_cutoff.is_finite()
        || value.alpha_cutoff < 0.0
    {
        return Err(GltfImportError::InvalidMaterial {
            path: path.to_path_buf(),
            material,
            reason: "normal scale, occlusion strength, or alpha cutoff is invalid".to_owned(),
        });
    }
    Ok(())
}

fn map_min_filter(filter: gltf::texture::MinFilter) -> TextureMinFilter {
    match filter {
        gltf::texture::MinFilter::Nearest => TextureMinFilter::Nearest,
        gltf::texture::MinFilter::Linear => TextureMinFilter::Linear,
        gltf::texture::MinFilter::NearestMipmapNearest => TextureMinFilter::NearestMipmapNearest,
        gltf::texture::MinFilter::LinearMipmapNearest => TextureMinFilter::LinearMipmapNearest,
        gltf::texture::MinFilter::NearestMipmapLinear => TextureMinFilter::NearestMipmapLinear,
        gltf::texture::MinFilter::LinearMipmapLinear => TextureMinFilter::LinearMipmapLinear,
    }
}

fn map_wrap_mode(mode: gltf::texture::WrappingMode) -> TextureWrapMode {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => TextureWrapMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => TextureWrapMode::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => TextureWrapMode::Repeat,
    }
}

fn invalid_primitive(path: &Path, mesh: usize, primitive: usize, reason: &str) -> GltfImportError {
    GltfImportError::InvalidPrimitive {
        path: path.to_path_buf(),
        mesh,
        primitive,
        reason: reason.to_owned(),
    }
}
fn expected_image_bytes(format: ImagePixelFormat, width: u32, height: u32) -> Option<usize> {
    let (channels, bytes_per_channel) = match format {
        ImagePixelFormat::R8 => (1_u64, 1_u64),
        ImagePixelFormat::Rg8 => (2, 1),
        ImagePixelFormat::Rgb8 => (3, 1),
        ImagePixelFormat::Rgba8 => (4, 1),
        ImagePixelFormat::R16 => (1, 2),
        ImagePixelFormat::Rg16 => (2, 2),
        ImagePixelFormat::Rgb16 => (3, 2),
        ImagePixelFormat::Rgba16 => (4, 2),
        ImagePixelFormat::Rgb32Float => (3, 4),
        ImagePixelFormat::Rgba32Float => (4, 4),
    };
    u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(channels)?
        .checked_mul(bytes_per_channel)?
        .try_into()
        .ok()
}
#[cfg(test)]
mod tests {
    use super::{ImagePixelFormat, PrimitiveTopology, expected_image_bytes, valid_index_count};
    #[test]
    fn primitive_index_counts_match_each_topology() {
        assert!(valid_index_count(PrimitiveTopology::Points, 1));
        assert!(!valid_index_count(PrimitiveTopology::Points, 0));
        assert!(valid_index_count(PrimitiveTopology::Lines, 2));
        assert!(!valid_index_count(PrimitiveTopology::Lines, 3));
        assert!(valid_index_count(PrimitiveTopology::LineStrip, 2));
        assert!(valid_index_count(PrimitiveTopology::Triangles, 3));
        assert!(!valid_index_count(PrimitiveTopology::Triangles, 4));
        assert!(valid_index_count(PrimitiveTopology::TriangleStrip, 3));
        assert!(valid_index_count(PrimitiveTopology::TriangleFan, 4));
    }

    #[test]
    fn image_byte_validation_uses_channel_count_and_bit_depth() {
        assert_eq!(expected_image_bytes(ImagePixelFormat::R8, 2, 3), Some(6));
        assert_eq!(
            expected_image_bytes(ImagePixelFormat::Rgba8, 2, 3),
            Some(24)
        );
        assert_eq!(
            expected_image_bytes(ImagePixelFormat::Rgb16, 2, 3),
            Some(36)
        );
        assert_eq!(
            expected_image_bytes(ImagePixelFormat::Rgba32Float, 2, 3),
            Some(96)
        );
        assert_eq!(
            expected_image_bytes(ImagePixelFormat::Rgba32Float, u32::MAX, u32::MAX),
            None
        );
    }

    #[test]
    fn bounded_external_reads_enforce_actual_bytes_and_aggregate_budget() {
        use super::{
            ExternalReadBudget, GltfImportError, GltfImportOptions, read_external_resource,
        };
        use std::{
            fs,
            path::Path,
            sync::atomic::{AtomicU64, Ordering},
        };

        static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "z-ideal-resource-budget-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create isolated fixture directory");
        let oversized = directory.join("oversized.bin");
        fs::write(&oversized, [1_u8, 2, 3, 4]).expect("write oversized resource");
        let options = GltfImportOptions {
            max_external_resource_bytes: 3,
            max_total_external_resource_bytes: 10,
            ..GltfImportOptions::default()
        };
        let error = read_external_resource(
            &oversized,
            "oversized.bin",
            Path::new("scene.gltf"),
            options,
            &mut ExternalReadBudget::default(),
        )
        .expect_err("actual external read must enforce the per-file limit");
        assert!(matches!(
            error,
            GltfImportError::ResourceLimitExceeded {
                actual_bytes: 4,
                limit_bytes: 3,
                ..
            }
        ));
        fs::remove_dir_all(directory).expect("remove isolated fixture directory");
    }

    #[test]
    fn bounded_external_reads_count_distinct_canonical_paths_once() {
        use super::{ExternalReadBudget, GltfImportOptions, read_external_resource};
        use std::{
            fs,
            path::Path,
            sync::atomic::{AtomicU64, Ordering},
        };

        static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "z-ideal-resource-dedup-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create isolated fixture directory");
        let shared = directory.join("shared.bin");
        fs::write(&shared, [1_u8, 2, 3]).expect("write shared resource");
        let options = GltfImportOptions {
            max_external_resource_bytes: 3,
            max_total_external_resource_bytes: 3,
            ..GltfImportOptions::default()
        };
        let mut budget = ExternalReadBudget::default();
        for uri in ["shared.bin", "alias/shared.bin"] {
            read_external_resource(&shared, uri, Path::new("scene.gltf"), options, &mut budget)
                .expect("re-reading one canonical file does not consume aggregate budget twice");
        }
        assert_eq!(budget.total_bytes, 3);
        fs::remove_dir_all(directory).expect("remove isolated fixture directory");
    }
}
