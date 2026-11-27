use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use z_ideal_assets::{GltfImportError, GltfImporter};

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "z-ideal-gltf-fixture-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create fixture directory");
        Self(path)
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
                "baseColorTexture":{"index":0},
                "metallicFactor":0.25,
                "roughnessFactor":0.75
            }}],
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
    assert_eq!(imported.materials[0].base_color_texture, Some(0));
    assert_eq!(imported.textures.len(), 1);
    assert_eq!(imported.images.len(), 1);
    assert_eq!(imported.images[0].width, 1);
    assert_eq!(imported.images[0].height, 1);
    assert_eq!(imported.dependency_uris, vec!["mesh.bin"]);
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
