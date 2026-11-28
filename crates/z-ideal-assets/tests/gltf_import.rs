use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

use z_ideal_assets::{GltfImportError, GltfImportOptions, GltfImporter};

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        loop {
            let sequence = FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "z-ideal-gltf-fixture-{}-{timestamp}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create unique fixture directory {path:?}: {error}"),
            }
        }
    }

    fn write_triangle(&self, invalid_index: bool) -> PathBuf {
        let mut binary = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        let indices: [u16; 3] = if invalid_index { [0, 1, 3] } else { [0, 1, 2] };
        for index in indices {
            binary.extend_from_slice(&index.to_le_bytes());
        }
        fs::write(self.0.join("mesh.bin"), binary).expect("write binary buffer");

        let document = r#"{
            "asset":{"version":"2.0","generator":"z-ideal-tests"},
            "scene":0,
            "scenes":[{"nodes":[0]}],
            "nodes":[{"mesh":0}],
            "buffers":[{"uri":"mesh.bin","byteLength":42}],
            "bufferViews":[
                {"buffer":0,"byteOffset":0,"byteLength":36,"target":34962},
                {"buffer":0,"byteOffset":36,"byteLength":6,"target":34963}
            ],
            "accessors":[
                {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
                {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}
            ],
            "images":[{"uri":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=="}],
            "textures":[{"source":0}],
            "materials":[{"name":"paint","pbrMetallicRoughness":{
                "baseColorFactor":[0.5,0.6,0.7,1.0],
                "baseColorTexture":{"index":0,"texCoord":0},
                "metallicRoughnessTexture":{"index":0,"texCoord":1},
                "metallicFactor":0.25,
                "roughnessFactor":0.75
            },
            "normalTexture":{"index":0,"texCoord":2,"scale":0.35},
            "occlusionTexture":{"index":0,"texCoord":3,"strength":0.4},
            "emissiveTexture":{"index":0,"texCoord":4},
            "emissiveFactor":[0.1,0.2,0.3],
            "alphaMode":"MASK",
            "alphaCutoff":0.37,
            "doubleSided":true}],
            "meshes":[{"name":"triangle","primitives":[{
                "attributes":{"POSITION":0},"indices":1,"material":0
            }]}]
        }"#;
        let path = self.0.join("triangle.gltf");
        fs::write(&path, document).expect("write glTF document");
        path
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn imports_gltf_mesh_indices_material_and_external_dependency() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let imported = GltfImporter::import(&path).expect("valid glTF fixture");

    assert_eq!(imported.meshes.len(), 1);
    assert_eq!(imported.meshes[0].name.as_deref(), Some("triangle"));
    let primitive = &imported.meshes[0].primitives[0];
    assert_eq!(primitive.positions.len(), 3);
    assert_eq!(primitive.indices, vec![0, 1, 2]);
    assert_eq!(primitive.material_index, Some(0));
    assert_eq!(imported.materials.len(), 1);
    assert_eq!(
        imported.materials[0].base_color_factor,
        [0.5, 0.6, 0.7, 1.0]
    );
    assert_eq!(imported.materials[0].metallic_factor, 0.25);
    assert_eq!(imported.materials[0].roughness_factor, 0.75);
    let material = &imported.materials[0];
    assert_eq!(material.base_color_texture, Some(0));
    assert_eq!(material.base_color_tex_coord, 0);
    assert_eq!(material.metallic_roughness_texture, Some(0));
    assert_eq!(material.metallic_roughness_tex_coord, 1);
    assert_eq!(material.normal_texture, Some(0));
    assert_eq!(material.normal_tex_coord, 2);
    assert_eq!(material.normal_scale, 0.35);
    assert_eq!(material.occlusion_texture, Some(0));
    assert_eq!(material.occlusion_tex_coord, 3);
    assert_eq!(material.occlusion_strength, 0.4);
    assert_eq!(material.emissive_texture, Some(0));
    assert_eq!(material.emissive_tex_coord, 4);
    assert_eq!(material.emissive_factor, [0.1, 0.2, 0.3]);
    assert_eq!(material.alpha_mode, z_ideal_assets::MaterialAlphaMode::Mask);
    assert_eq!(material.alpha_cutoff, 0.37);
    assert!(material.double_sided);
    assert_eq!(imported.textures.len(), 1);
    assert_eq!(imported.images.len(), 1);
    assert_eq!(imported.images[0].width, 1);
    assert_eq!(imported.images[0].height, 1);
    assert_eq!(imported.dependency_uris, vec!["mesh.bin"]);
}

#[test]
fn rejects_out_of_range_imported_material_factors() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read glTF fixture");
    fs::write(
        &path,
        document.replace("\"metallicFactor\":0.25", "\"metallicFactor\":1.5"),
    )
    .expect("write invalid material fixture");

    assert!(matches!(
        GltfImporter::import(&path),
        Err(GltfImportError::InvalidMaterial { material: 0, .. })
    ));
}

#[test]
fn imports_glb_with_embedded_binary_buffer() {
    let fixture = FixtureDir::new();
    let json_path = fixture.write_triangle(false);
    let mut document = fs::read_to_string(&json_path).expect("read glTF fixture");
    document = document.replace(r#""uri":"mesh.bin","#, "");
    let mut json = document.into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut binary = fs::read(fixture.0.join("mesh.bin")).expect("read binary fixture");
    while binary.len() % 4 != 0 {
        binary.push(0);
    }

    let total_length = 12 + 8 + json.len() + 8 + binary.len();
    let mut glb = Vec::with_capacity(total_length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(
        &u32::try_from(total_length)
            .expect("GLB length")
            .to_le_bytes(),
    );
    glb.extend_from_slice(
        &u32::try_from(json.len())
            .expect("JSON chunk length")
            .to_le_bytes(),
    );
    glb.extend_from_slice(&0x4E4F534A_u32.to_le_bytes());
    glb.extend_from_slice(&json);
    glb.extend_from_slice(
        &u32::try_from(binary.len())
            .expect("BIN chunk length")
            .to_le_bytes(),
    );
    glb.extend_from_slice(&0x004E4942_u32.to_le_bytes());
    glb.extend_from_slice(&binary);

    let glb_path = fixture.0.join("triangle.glb");
    fs::write(&glb_path, glb).expect("write GLB fixture");
    let imported = GltfImporter::import(&glb_path).expect("valid GLB fixture");
    assert_eq!(imported.meshes.len(), 1);
    assert_eq!(imported.meshes[0].primitives[0].indices, vec![0, 1, 2]);
    assert_eq!(imported.images.len(), 1);
    assert!(imported.dependency_uris.is_empty());
}

#[test]
fn rejects_external_uri_path_traversal_before_loading_resources() {
    for uri in ["../mesh.bin", "%2e%2e%2fmesh.bin"] {
        let fixture = FixtureDir::new();
        let path = fixture.write_triangle(false);
        let document = fs::read_to_string(&path).expect("read fixture");
        let document = document.replace("mesh.bin", uri);
        fs::write(&path, document).expect("write traversal fixture");
        assert!(matches!(
            GltfImporter::import(&path),
            Err(GltfImportError::InvalidExternalUri { .. })
        ));
    }
}

#[test]
fn rejects_mesh_indices_outside_the_vertex_array() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(true);
    assert!(matches!(
        GltfImporter::import(&path),
        Err(GltfImportError::InvalidPrimitive { .. })
    ));
}

#[test]
fn malformed_document_returns_an_error_instead_of_panicking() {
    let fixture = FixtureDir::new();
    let path = fixture.0.join("broken.gltf");
    fs::write(
        &path,
        r#"{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{}}]}]}"#,
    )
    .expect("write malformed fixture");
    assert!(GltfImporter::import(&path).is_err());
}

#[test]
fn imports_scene_nodes_transforms_mesh_instances_and_roots() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read fixture");
    let document = document
        .replace(
            r#""nodes":[{"mesh":0}]"#,
            r#""nodes":[
                {"name":"root","translation":[1.0,2.0,3.0],"children":[1]},
                {"name":"mesh-node","rotation":[0.0,0.0,0.7071068,0.7071068],"scale":[2.0,2.0,2.0],"mesh":0}
            ]"#,
        )
        .replace(
            r#""scenes":[{"nodes":[0]}]"#,
            r#""scenes":[{"name":"main","nodes":[0]}]"#,
        );
    fs::write(&path, document).expect("write hierarchy fixture");

    let imported = GltfImporter::import(&path).expect("valid node hierarchy");
    assert_eq!(imported.nodes.len(), 2);
    assert_eq!(imported.nodes[0].source_index, 0);
    assert_eq!(imported.nodes[0].name.as_deref(), Some("root"));
    assert_eq!(imported.nodes[0].translation, [1.0, 2.0, 3.0]);
    assert_eq!(imported.nodes[0].children, vec![1]);
    assert_eq!(imported.nodes[1].name.as_deref(), Some("mesh-node"));
    assert_eq!(imported.nodes[1].mesh_index, Some(0));
    assert_eq!(imported.nodes[1].scale, [2.0, 2.0, 2.0]);
    assert_eq!(imported.scenes.len(), 1);
    assert_eq!(imported.scenes[0].name.as_deref(), Some("main"));
    assert_eq!(imported.scenes[0].root_nodes, vec![0]);
    assert_eq!(imported.default_scene, Some(0));
}

#[test]
fn rejects_cyclic_node_hierarchies() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read fixture");
    let document = document.replace(
        r#""nodes":[{"mesh":0}]"#,
        r#""nodes":[{"children":[1]},{"children":[0],"mesh":0}]"#,
    );
    fs::write(&path, document).expect("write cyclic fixture");
    assert!(GltfImporter::import(&path).is_err());
}

#[test]
fn rejects_nodes_with_multiple_parents() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read fixture");
    let document = document.replace(
        r#""nodes":[{"mesh":0}]"#,
        r#""nodes":[{"children":[2]},{"children":[2]},{"mesh":0}]"#,
    );
    fs::write(&path, document).expect("write multi-parent fixture");
    assert!(GltfImporter::import(&path).is_err());
}

#[test]
fn enforces_source_document_byte_budget_before_parsing() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_source_file_bytes: 1,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("source document exceeds configured budget");
    assert!(matches!(
        error,
        GltfImportError::ResourceLimitExceeded {
            resource,
            actual_bytes,
            limit_bytes: 1,
            ..
        } if resource == "source document" && actual_bytes > 1
    ));
}

#[test]
fn enforces_per_file_external_resource_byte_budget_before_loading() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_external_resource_bytes: 41,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("external buffer exceeds configured budget");
    assert!(matches!(
        error,
        GltfImportError::ResourceLimitExceeded {
            resource,
            actual_bytes: 42,
            limit_bytes: 41,
            ..
        } if resource == "mesh.bin"
    ));
}

#[test]
fn enforces_aggregate_external_resource_byte_budget() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_total_external_resource_bytes: 41,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("combined external resources exceed configured budget");
    assert!(matches!(
        error,
        GltfImportError::ResourceLimitExceeded {
            resource,
            actual_bytes: 42,
            limit_bytes: 41,
            ..
        } if resource == "combined external resources"
    ));
}

#[test]
fn enforces_strict_image_dimensions_before_decoding_pixels() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_image_width: 0,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("one-pixel image exceeds zero-pixel width budget");
    assert!(matches!(
        error,
        GltfImportError::ImageDimensionsExceeded {
            index: 0,
            width: 1,
            height: 1,
            max_width: 0,
            ..
        }
    ));
}

#[test]
fn enforces_per_image_decoded_byte_budget_before_pixel_allocation() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_decoded_image_bytes: 3,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("four decoded RGBA bytes exceed configured per-image budget");
    assert!(matches!(
        error,
        GltfImportError::ResourceLimitExceeded {
            resource,
            actual_bytes: 4,
            limit_bytes: 3,
            ..
        } if resource == "decoded image 0"
    ));
}

#[test]
fn enforces_aggregate_decoded_image_byte_budget() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_total_decoded_image_bytes: 3,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("four decoded RGBA bytes exceed aggregate image budget");
    assert!(matches!(
        error,
        GltfImportError::ResourceLimitExceeded {
            resource,
            actual_bytes: 4,
            limit_bytes: 3,
            ..
        } if resource == "combined decoded images"
    ));
}

#[test]
fn malformed_embedded_image_returns_a_structured_decode_error() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read fixture");
    let document = document.replace(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg==",
        "AA==",
    );
    fs::write(&path, document).expect("write malformed image fixture");
    assert!(matches!(
        GltfImporter::import(&path),
        Err(GltfImportError::ImageDecode { index: 0, .. })
    ));
}

#[test]
fn enforces_external_image_file_byte_budget() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read fixture");
    let document = document.replace(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg==",
        "image.png",
    );
    fs::write(&path, document).expect("write external image fixture");
    fs::write(
        fixture.0.join("image.png"),
        base64::decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg==")
            .expect("decode PNG fixture"),
    )
    .expect("write external image");
    let error = GltfImporter::import_with_options(
        &path,
        GltfImportOptions {
            max_external_resource_bytes: 64,
            ..GltfImportOptions::default()
        },
    )
    .expect_err("external image exceeds configured per-file budget");
    assert!(matches!(
        error,
        GltfImportError::ResourceLimitExceeded {
            resource,
            actual_bytes: 70,
            limit_bytes: 64,
            ..
        } if resource == "image.png"
    ));
}

#[test]
fn truncated_external_buffer_returns_a_structured_import_error() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    fs::write(fixture.0.join("mesh.bin"), [0_u8; 41]).expect("truncate external buffer");
    let error = GltfImporter::import(&path).expect_err("truncated buffer must fail");
    println!("truncated buffer error: {error:?}");
    assert!(matches!(error, GltfImportError::Import { .. }));
}

#[test]
fn accessor_extending_beyond_its_buffer_view_is_rejected() {
    let fixture = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let document = fs::read_to_string(&path).expect("read glTF fixture");
    let malformed = document.replace(
        "\"byteOffset\":0,\"byteLength\":36",
        "\"byteOffset\":0,\"byteLength\":35",
    );
    assert_ne!(document, malformed, "fixture buffer view must be found");
    fs::write(&path, malformed).expect("write malformed accessor fixture");
    assert!(GltfImporter::import(&path).is_err());
}

#[cfg(unix)]
#[test]
fn rejects_external_resource_symlink_that_resolves_outside_project_directory() {
    use std::os::unix::fs::symlink;

    let fixture = FixtureDir::new();
    let outside = FixtureDir::new();
    let path = fixture.write_triangle(false);
    let external_target = outside.0.join("outside.bin");
    fs::write(&external_target, [0_u8; 42]).expect("write external target");
    fs::remove_file(fixture.0.join("mesh.bin")).expect("remove original resource");
    symlink(&external_target, fixture.0.join("mesh.bin")).expect("create escaping symlink");
    assert!(matches!(
        GltfImporter::import(&path),
        Err(GltfImportError::InvalidExternalUri { .. })
    ));
}
