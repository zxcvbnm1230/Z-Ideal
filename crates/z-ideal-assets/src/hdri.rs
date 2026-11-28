//! Bounded import of scene-linear Radiance RGBE equirectangular environment maps.
use std::{
    fs::File,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use image::{DynamicImage, ImageDecoder as _, ImageReader, Limits};

/// Resource limits for Radiance HDR environment-map imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdrEnvironmentImportOptions {
    /// Maximum encoded source-file size in bytes.
    pub max_source_file_bytes: u64,
    /// Maximum decoded width in pixels.
    pub max_width: u32,
    /// Maximum decoded height in pixels.
    pub max_height: u32,
    /// Maximum decoded RGB32F pixel storage in bytes.
    pub max_decoded_bytes: u64,
}

impl Default for HdrEnvironmentImportOptions {
    fn default() -> Self {
        Self {
            max_source_file_bytes: 256 * 1024 * 1024,
            max_width: 16_384,
            max_height: 8_192,
            max_decoded_bytes: 512 * 1024 * 1024,
        }
    }
}

/// A decoded, scene-linear, equirectangular Radiance environment map.
///
/// Pixels are RGB32F, row-major, and preserve values above one. The map must have
/// a 2:1 width-to-height ratio; orientation follows the decoded Radiance image.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrEnvironmentMap {
    /// Original source path, retained for diagnostics and asset provenance.
    pub source_path: PathBuf,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Scene-linear RGB pixels in row-major order.
    pub pixels: Vec<[f32; 3]>,
}

impl HdrEnvironmentMap {
    /// Returns the number of RGB pixels in the environment map.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.pixels.len()
    }
}

/// Errors returned while importing a Radiance HDR environment map.
#[derive(Debug, thiserror::Error)]
pub enum HdrEnvironmentImportError {
    /// The source could not be opened or read.
    #[error("could not read HDR environment {path}: {message}")]
    Io { path: PathBuf, message: String },
    /// Encoded or decoded data exceeded the configured resource budget.
    #[error("HDR environment {path} {resource} is {actual}; configured limit is {limit}")]
    ResourceLimitExceeded {
        path: PathBuf,
        resource: &'static str,
        actual: u64,
        limit: u64,
    },
    /// Import options contain a zero resource limit.
    #[error("HDR environment import limits must all be non-zero")]
    InvalidOptions,
    /// The decoded image dimensions are zero or exceed configured limits.
    #[error(
        "HDR environment {path} dimensions {width}x{height} exceed valid limits {max_width}x{max_height}"
    )]
    InvalidDimensions {
        path: PathBuf,
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },
    /// The decoded image is not a 2:1 equirectangular projection.
    #[error(
        "HDR environment {path} is {width}x{height}; equirectangular maps must have a 2:1 ratio"
    )]
    InvalidProjection {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    /// The source is not a supported RGB32F Radiance HDR image or could not be decoded.
    #[error("could not decode HDR environment {path}: {message}")]
    Decode { path: PathBuf, message: String },
    /// A decoded sample is negative or non-finite and cannot represent scene radiance.
    #[error("HDR environment {path} contains invalid radiance at pixel {pixel}, channel {channel}")]
    InvalidRadiance {
        path: PathBuf,
        pixel: usize,
        channel: usize,
    },
}

/// Imports Radiance RGBE .hdr files as bounded scene-linear equirectangular maps.
#[derive(Debug, Default, Clone, Copy)]
pub struct HdrEnvironmentImporter;

impl HdrEnvironmentImporter {
    /// Imports an HDR environment using the default resource limits.
    pub fn import(path: impl AsRef<Path>) -> Result<HdrEnvironmentMap, HdrEnvironmentImportError> {
        Self::import_with_options(path, HdrEnvironmentImportOptions::default())
    }

    /// Imports an HDR environment while enforcing encoded and decoded resource limits.
    pub fn import_with_options(
        path: impl AsRef<Path>,
        options: HdrEnvironmentImportOptions,
    ) -> Result<HdrEnvironmentMap, HdrEnvironmentImportError> {
        let path = path.as_ref();
        if options.max_source_file_bytes == 0
            || options.max_width == 0
            || options.max_height == 0
            || options.max_decoded_bytes == 0
        {
            return Err(HdrEnvironmentImportError::InvalidOptions);
        }

        let file = File::open(path).map_err(|error| HdrEnvironmentImportError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let mut encoded = Vec::new();
        file.take(options.max_source_file_bytes.saturating_add(1))
            .read_to_end(&mut encoded)
            .map_err(|error| HdrEnvironmentImportError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let encoded_bytes = u64::try_from(encoded.len()).unwrap_or(u64::MAX);
        if encoded_bytes > options.max_source_file_bytes {
            return Err(HdrEnvironmentImportError::ResourceLimitExceeded {
                path: path.to_path_buf(),
                resource: "encoded source",
                actual: encoded_bytes,
                limit: options.max_source_file_bytes,
            });
        }

        let reader = ImageReader::new(Cursor::new(encoded))
            .with_guessed_format()
            .map_err(|error| HdrEnvironmentImportError::Decode {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let mut decoder =
            reader
                .into_decoder()
                .map_err(|error| HdrEnvironmentImportError::Decode {
                    path: path.to_path_buf(),
                    message: error.to_string(),
                })?;
        let (width, height) = decoder.dimensions();
        if width == 0 || height == 0 || width > options.max_width || height > options.max_height {
            return Err(HdrEnvironmentImportError::InvalidDimensions {
                path: path.to_path_buf(),
                width,
                height,
                max_width: options.max_width,
                max_height: options.max_height,
            });
        }
        if u64::from(width) != u64::from(height).saturating_mul(2) {
            return Err(HdrEnvironmentImportError::InvalidProjection {
                path: path.to_path_buf(),
                width,
                height,
            });
        }
        if decoder.color_type() != image::ColorType::Rgb32F {
            return Err(HdrEnvironmentImportError::Decode {
                path: path.to_path_buf(),
                message: format!("expected RGB32F radiance, got {:?}", decoder.color_type()),
            });
        }
        let decoded_bytes = decoder.total_bytes();
        if decoded_bytes > options.max_decoded_bytes {
            return Err(HdrEnvironmentImportError::ResourceLimitExceeded {
                path: path.to_path_buf(),
                resource: "decoded pixel storage",
                actual: decoded_bytes,
                limit: options.max_decoded_bytes,
            });
        }
        let mut limits = Limits::default();
        limits.max_image_width = Some(options.max_width);
        limits.max_image_height = Some(options.max_height);
        limits.max_alloc = Some(options.max_decoded_bytes);
        decoder
            .set_limits(limits)
            .map_err(|error| HdrEnvironmentImportError::Decode {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let decoded = DynamicImage::from_decoder(decoder).map_err(|error| {
            HdrEnvironmentImportError::Decode {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let raw_pixels = match decoded {
            DynamicImage::ImageRgb32F(image) => image.into_raw(),
            other => {
                return Err(HdrEnvironmentImportError::Decode {
                    path: path.to_path_buf(),
                    message: format!(
                        "expected RGB32F image after decode, got {:?}",
                        other.color()
                    ),
                });
            }
        };
        let expected_pixels =
            usize::try_from(u64::from(width) * u64::from(height)).map_err(|_| {
                HdrEnvironmentImportError::ResourceLimitExceeded {
                    path: path.to_path_buf(),
                    resource: "decoded pixel count",
                    actual: u64::MAX,
                    limit: options.max_decoded_bytes / 12,
                }
            })?;
        let expected_channels = expected_pixels.checked_mul(3).ok_or_else(|| {
            HdrEnvironmentImportError::ResourceLimitExceeded {
                path: path.to_path_buf(),
                resource: "decoded channel count",
                actual: u64::MAX,
                limit: options.max_decoded_bytes / 4,
            }
        })?;
        if raw_pixels.len() != expected_channels {
            return Err(HdrEnvironmentImportError::Decode {
                path: path.to_path_buf(),
                message: format!(
                    "decoded {} channels, expected {expected_channels}",
                    raw_pixels.len()
                ),
            });
        }
        let pixels: Vec<[f32; 3]> = raw_pixels
            .chunks_exact(3)
            .map(|channels| [channels[0], channels[1], channels[2]])
            .collect();
        for (pixel_index, pixel) in pixels.iter().enumerate() {
            for (channel_index, sample) in pixel.iter().enumerate() {
                if !sample.is_finite() || *sample < 0.0 {
                    return Err(HdrEnvironmentImportError::InvalidRadiance {
                        path: path.to_path_buf(),
                        pixel: pixel_index,
                        channel: channel_index,
                    });
                }
            }
        }
        Ok(HdrEnvironmentMap {
            source_path: path.to_path_buf(),
            width,
            height,
            pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{HdrEnvironmentImportError, HdrEnvironmentImportOptions, HdrEnvironmentImporter};
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn fixture_path(name: &str) -> std::path::PathBuf {
        static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("z-ideal-hdri-{}-{sequence}", std::process::id()));
        fs::create_dir_all(&directory).expect("create fixture directory");
        directory.join(name)
    }

    fn tiny_hdr() -> Vec<u8> {
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        // Widths below eight use the Radiance uncompressed RGBE scanline representation.
        bytes.extend_from_slice(&[128, 64, 32, 129, 64, 128, 255, 130]);
        bytes
    }

    #[test]
    fn imports_scene_linear_radiance_and_preserves_values_above_one() {
        let path = fixture_path("tiny.hdr");
        fs::write(&path, tiny_hdr()).expect("write Radiance fixture");
        let map = HdrEnvironmentImporter::import(&path).expect("import valid HDRI");
        assert_eq!((map.width, map.height), (2, 1));
        assert_eq!(map.pixel_count(), 2);
        assert_eq!(map.pixels[0], [1.0, 0.5, 0.25]);
        assert!(map.pixels[1][2] > 1.0);
        fs::remove_dir_all(path.parent().expect("fixture parent")).expect("clean fixture");
    }

    #[test]
    fn rejects_encoded_source_over_budget_before_decoding() {
        let path = fixture_path("oversized.hdr");
        fs::write(&path, tiny_hdr()).expect("write Radiance fixture");
        let error = HdrEnvironmentImporter::import_with_options(
            &path,
            HdrEnvironmentImportOptions {
                max_source_file_bytes: 8,
                ..HdrEnvironmentImportOptions::default()
            },
        )
        .expect_err("encoded source exceeds budget");
        assert!(matches!(
            error,
            HdrEnvironmentImportError::ResourceLimitExceeded {
                resource: "encoded source",
                ..
            }
        ));
        fs::remove_dir_all(path.parent().expect("fixture parent")).expect("clean fixture");
    }

    #[test]
    fn rejects_decoded_storage_over_budget_before_pixel_allocation() {
        let path = fixture_path("decoded-budget.hdr");
        fs::write(&path, tiny_hdr()).expect("write Radiance fixture");
        let error = HdrEnvironmentImporter::import_with_options(
            &path,
            HdrEnvironmentImportOptions {
                max_decoded_bytes: 23,
                ..HdrEnvironmentImportOptions::default()
            },
        )
        .expect_err("two RGB32F pixels require 24 bytes");
        assert!(matches!(
            error,
            HdrEnvironmentImportError::ResourceLimitExceeded {
                resource: "decoded pixel storage",
                actual: 24,
                limit: 23,
                ..
            }
        ));
        fs::remove_dir_all(path.parent().expect("fixture parent")).expect("clean fixture");
    }

    #[test]
    fn rejects_dimensions_over_configured_limits() {
        let path = fixture_path("dimension-budget.hdr");
        fs::write(&path, tiny_hdr()).expect("write Radiance fixture");
        let error = HdrEnvironmentImporter::import_with_options(
            &path,
            HdrEnvironmentImportOptions {
                max_width: 1,
                ..HdrEnvironmentImportOptions::default()
            },
        )
        .expect_err("width exceeds configured limit");
        assert!(matches!(
            error,
            HdrEnvironmentImportError::InvalidDimensions {
                width: 2,
                height: 1,
                max_width: 1,
                ..
            }
        ));
        fs::remove_dir_all(path.parent().expect("fixture parent")).expect("clean fixture");
    }

    #[test]
    fn rejects_non_equirectangular_projection() {
        let path = fixture_path("square.hdr");
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n".to_vec();
        bytes.extend_from_slice(&[128, 64, 32, 129]);
        fs::write(&path, bytes).expect("write Radiance fixture");
        let error = HdrEnvironmentImporter::import(&path).expect_err("square map is not 2:1");
        assert!(matches!(
            error,
            HdrEnvironmentImportError::InvalidProjection {
                width: 1,
                height: 1,
                ..
            }
        ));
        fs::remove_dir_all(path.parent().expect("fixture parent")).expect("clean fixture");
    }

    #[test]
    fn rejects_invalid_limits_and_malformed_headers() {
        let path = fixture_path("bad.hdr");
        fs::write(&path, b"not an HDR file").expect("write malformed fixture");
        let invalid_options = HdrEnvironmentImporter::import_with_options(
            &path,
            HdrEnvironmentImportOptions {
                max_width: 0,
                ..HdrEnvironmentImportOptions::default()
            },
        )
        .expect_err("zero limits are invalid");
        assert!(matches!(
            invalid_options,
            HdrEnvironmentImportError::InvalidOptions
        ));
        let malformed =
            HdrEnvironmentImporter::import(&path).expect_err("bad signature is rejected");
        assert!(matches!(
            malformed,
            HdrEnvironmentImportError::Decode { .. }
        ));
        fs::remove_dir_all(path.parent().expect("fixture parent")).expect("clean fixture");
    }
}
