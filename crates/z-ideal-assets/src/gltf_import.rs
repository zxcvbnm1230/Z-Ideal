//! glTF 2.0 / GLB import into renderer-independent, validated CPU asset data.
use gltf::{buffer, image};

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
    collections::BTreeSet,
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
    /// Metallic-roughness texture index.
    pub metallic_roughness_texture: Option<usize>,
    /// Normal texture index.
    pub normal_texture: Option<usize>,
    /// Occlusion texture index.
    pub occlusion_texture: Option<usize>,
    /// Emissive texture index.
    pub emissive_texture: Option<usize>,
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
    /// A material refers to a texture outside the imported texture table.
    #[error("glTF {path} material {material} refers to missing texture {texture}")]
    MissingTexture {
        path: PathBuf,
        material: usize,
        texture: usize,
    },
}

/// glTF 2.0 importer.
#[derive(Debug, Default, Clone, Copy)]
pub struct GltfImporter;
impl GltfImporter {
    /// Imports a .gltf or .glb file, including referenced buffers and decoded images.
    ///
    /// External URIs are resolved relative to the source file. This convenience API
    /// delegates external-file loading and image decoding to gltf/image without a
    /// configurable pre-allocation byte or pixel budget; import only trusted project assets.
    pub fn import(path: impl AsRef<Path>) -> Result<GltfImport, GltfImportError> {
        let path = path.as_ref();
        let source_path = path.to_path_buf();
        let parsed = gltf::Gltf::open(path).map_err(|error| GltfImportError::Import {
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
        for buffer in document.buffers() {
            if let buffer::Source::Uri(uri) = buffer.source() {
                if let Some(uri) = validate_external_uri(&source_directory, &source_path, uri)? {
                    external_dependencies.insert(uri);
                }
            }
        }
        for image in document.images() {
            if let image::Source::Uri { uri, .. } = image.source() {
                if let Some(uri) = validate_external_uri(&source_directory, &source_path, uri)? {
                    external_dependencies.insert(uri);
                }
            }
        }
        let buffers =
            gltf::import_buffers(&document, Some(&source_directory), blob).map_err(|error| {
                GltfImportError::Import {
                    path: source_path.clone(),
                    message: error.to_string(),
                }
            })?;
        let images =
            gltf::import_images(&document, Some(&source_directory), &buffers).map_err(|error| {
                GltfImportError::Import {
                    path: source_path.clone(),
                    message: error.to_string(),
                }
            })?;
        let imported_images = images
            .into_iter()
            .enumerate()
            .map(|(index, image)| {
                let format = map_image_format(image.format);
                let expected = expected_image_bytes(format, image.width, image.height);
                if expected != Some(image.pixels.len()) || image.width == 0 || image.height == 0 {
                    return Err(GltfImportError::InvalidImageData {
                        path: source_path.clone(),
                        index,
                    });
                }
                Ok(ImportedImage {
                    width: image.width,
                    height: image.height,
                    format,
                    pixels: image.pixels,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

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
                let texture_indices = [
                    pbr.base_color_texture().map(|info| info.texture().index()),
                    pbr.metallic_roughness_texture()
                        .map(|info| info.texture().index()),
                    material.normal_texture().map(|info| info.texture().index()),
                    material
                        .occlusion_texture()
                        .map(|info| info.texture().index()),
                    material
                        .emissive_texture()
                        .map(|info| info.texture().index()),
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
                Ok(ImportedMaterial {
                    name: material.name().map(str::to_owned),
                    base_color_factor: pbr.base_color_factor(),
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    base_color_texture: texture_indices[0],
                    metallic_roughness_texture: texture_indices[1],
                    normal_texture: texture_indices[2],
                    occlusion_texture: texture_indices[3],
                    emissive_texture: texture_indices[4],
                    emissive_factor: material.emissive_factor(),
                    alpha_mode: match material.alpha_mode() {
                        gltf::material::AlphaMode::Opaque => MaterialAlphaMode::Opaque,
                        gltf::material::AlphaMode::Mask => MaterialAlphaMode::Mask,
                        gltf::material::AlphaMode::Blend => MaterialAlphaMode::Blend,
                    },
                    alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
                    double_sided: material.double_sided(),
                })
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

        let dependency_uris = external_dependencies.into_iter().collect();

        Ok(GltfImport {
            source_path,
            meshes: imported_meshes,
            materials: imported_materials,
            textures: imported_textures,
            images: imported_images,
            dependency_uris,
        })
    }
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

fn validate_external_uri(
    source_directory: &Path,
    source_path: &Path,
    uri: &str,
) -> Result<Option<String>, GltfImportError> {
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
    Ok(Some(asset_uri.to_string()))
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

fn map_image_format(format: image::Format) -> ImagePixelFormat {
    match format {
        image::Format::R8 => ImagePixelFormat::R8,
        image::Format::R8G8 => ImagePixelFormat::Rg8,
        image::Format::R8G8B8 => ImagePixelFormat::Rgb8,
        image::Format::R8G8B8A8 => ImagePixelFormat::Rgba8,
        image::Format::R16 => ImagePixelFormat::R16,
        image::Format::R16G16 => ImagePixelFormat::Rg16,
        image::Format::R16G16B16 => ImagePixelFormat::Rgb16,
        image::Format::R16G16B16A16 => ImagePixelFormat::Rgba16,
        image::Format::R32G32B32FLOAT => ImagePixelFormat::Rgb32Float,
        image::Format::R32G32B32A32FLOAT => ImagePixelFormat::Rgba32Float,
    }
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
}
