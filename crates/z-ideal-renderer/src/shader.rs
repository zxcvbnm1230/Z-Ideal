//! Deterministic GLSL-to-SPIR-V compilation and shader interface reflection.
use std::collections::{BTreeMap, HashMap};

use ash::vk;

use crate::{VulkanDevice, VulkanError, api_error};

/// Shader execution stage supported by the initial renderer pipeline API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex processing stage.
    Vertex,
    /// Fragment processing stage.
    Fragment,
    /// Compute processing stage.
    Compute,
}

impl ShaderStage {
    /// Returns the Vulkan shader-stage bit for this stage.
    #[must_use]
    pub const fn vulkan_flags(self) -> vk::ShaderStageFlags {
        match self {
            Self::Vertex => vk::ShaderStageFlags::VERTEX,
            Self::Fragment => vk::ShaderStageFlags::FRAGMENT,
            Self::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }

    const fn shaderc_kind(self) -> shaderc::ShaderKind {
        match self {
            Self::Vertex => shaderc::ShaderKind::Vertex,
            Self::Fragment => shaderc::ShaderKind::Fragment,
            Self::Compute => shaderc::ShaderKind::Compute,
        }
    }
}

/// Deterministic GLSL compilation settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderCompileConfig {
    /// Stage compiled from this GLSL source.
    pub stage: ShaderStage,
    /// Source name included in compiler diagnostics.
    pub source_name: String,
    /// Preprocessor definitions, applied in sorted key order.
    pub defines: BTreeMap<String, String>,
}

impl ShaderCompileConfig {
    /// Creates default compiler settings for one shader stage.
    #[must_use]
    pub fn new(stage: ShaderStage) -> Self {
        Self {
            stage,
            source_name: "shader.glsl".to_owned(),
            defines: BTreeMap::new(),
        }
    }
}

/// Descriptor binding reflected from one compiled shader entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShaderDescriptorBinding {
    /// Vulkan descriptor-set index (`layout(set = N)`).
    pub set: u32,
    /// Binding index within the descriptor set.
    pub binding: u32,
    /// Reflected Vulkan descriptor type.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors in this binding.
    pub descriptor_count: u32,
    /// Shader stage that references this binding.
    pub stage: vk::ShaderStageFlags,
}

/// Vertex-buffer input reflected from a vertex shader location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShaderVertexInput {
    /// Shader input location.
    pub location: u32,
    /// Vulkan vertex format required by the shader declaration.
    pub format: vk::Format,
}

/// Reflected resource interface for one shader entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderReflection {
    /// Stage of the reflected entry point.
    pub stage: ShaderStage,
    /// GLSL entry-point name; the initial compiler requires `main`.
    pub entry_point: String,
    /// Descriptor bindings sorted by set and binding.
    pub descriptors: Vec<ShaderDescriptorBinding>,
    /// Vertex input locations and formats; empty for non-vertex stages.
    pub vertex_inputs: Vec<ShaderVertexInput>,
    /// Push-constant block size in bytes, if declared.
    pub push_constant_size: Option<u32>,
}

/// Compiled SPIR-V words and their validated shader interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledShader {
    spirv: Vec<u32>,
    reflection: ShaderReflection,
    warnings: Vec<String>,
}

impl CompiledShader {
    /// Returns the compiled SPIR-V binary words.
    #[must_use]
    pub fn spirv(&self) -> &[u32] {
        &self.spirv
    }

    /// Returns the reflected shader interface.
    #[must_use]
    pub const fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }

    /// Returns non-fatal compiler diagnostics.
    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }
}

/// Owns a Vulkan shader module and its reflected entry-point interface.
pub struct VulkanShaderModule<'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::ShaderModule,
    reflection: ShaderReflection,
}

impl<'device> VulkanShaderModule<'device> {
    /// Creates a Vulkan shader module from a compiler-produced shader artifact.
    pub fn new(
        device: &'device VulkanDevice,
        compiled: CompiledShader,
    ) -> Result<Self, VulkanError> {
        Self::create(device, &compiled.spirv, compiled.reflection)
    }

    /// Creates a Vulkan shader module from precompiled SPIR-V words and reflects them.
    ///
    /// The input must be a valid Vulkan 1.0 / SPIR-V 1.0 module. This path checks
    /// the binary header and the resource/type structures needed for reflection;
    /// it does not replace the full shaderc validation performed by [`compile_glsl`].
    pub fn from_spirv(
        device: &'device VulkanDevice,
        spirv: &[u32],
        stage: ShaderStage,
    ) -> Result<Self, VulkanError> {
        let reflection = reflect_spirv(spirv, stage)?;
        Self::create(device, spirv, reflection)
    }

    fn create(
        device: &'device VulkanDevice,
        spirv: &[u32],
        reflection: ShaderReflection,
    ) -> Result<Self, VulkanError> {
        if spirv.len() < 5 || spirv.first().copied() != Some(SPIRV_MAGIC) {
            return Err(VulkanError::InvalidSpirv);
        }
        let info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: The SPIR-V word slice remains alive through creation, and the
        // returned wrapper retains the logical device that owns the module.
        let handle = unsafe {
            device
                .handle
                .create_shader_module(&info, None)
                .map_err(|error| api_error("create Vulkan shader module", error))?
        };
        Ok(Self {
            device,
            handle,
            reflection,
        })
    }

    /// Returns the raw Vulkan shader-module handle.
    #[must_use]
    pub const fn handle(&self) -> vk::ShaderModule {
        self.handle
    }

    /// Returns the reflected shader interface.
    #[must_use]
    pub const fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }
}

impl Drop for VulkanShaderModule<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before shader-module destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the shader module and retains its device.
        unsafe {
            self.device.handle.destroy_shader_module(self.handle, None);
        }
    }
}

/// Compiles GLSL source to Vulkan SPIR-V 1.0 and reflects the generated binary.
///
/// Shaderc/glslang is built from the pinned source dependency so compilation does
/// not depend on whichever shader compiler happens to be installed on the host.
/// The target environment is Vulkan 1.0 and optimization is disabled for stable,
/// debuggable output. Descriptor sets and bindings must be explicit in GLSL;
/// descriptor arrays must have compile-time constant lengths. Reflection covers
/// uniform/storage buffers, sampled/storage images, separate and combined samplers,
/// and push constants. Unsupported descriptor forms fail instead of being omitted.
pub fn compile_glsl(
    source: &str,
    config: &ShaderCompileConfig,
) -> Result<CompiledShader, VulkanError> {
    let compiler = shaderc::Compiler::new()
        .map_err(|error| VulkanError::ShaderCompilation(error.to_string()))?;
    let mut options = shaderc::CompileOptions::new()
        .map_err(|error| VulkanError::ShaderCompilation(error.to_string()))?;
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as u32,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_0);
    options.set_optimization_level(shaderc::OptimizationLevel::Zero);
    for (key, value) in &config.defines {
        options.add_macro_definition(key, Some(value));
    }
    let artifact = compiler
        .compile_into_spirv(
            source,
            config.stage.shaderc_kind(),
            &config.source_name,
            "main",
            Some(&options),
        )
        .map_err(|error| VulkanError::ShaderCompilation(error.to_string()))?;
    let spirv = artifact.as_binary().to_vec();
    if spirv.first().copied() != Some(SPIRV_MAGIC) {
        return Err(VulkanError::InvalidSpirv);
    }

    let reflection = reflect_spirv(&spirv, config.stage)?;
    let warning_text = artifact.get_warning_messages();
    let warnings = if warning_text.is_empty() {
        Vec::new()
    } else {
        warning_text.lines().map(str::to_owned).collect()
    };
    Ok(CompiledShader {
        spirv,
        reflection,
        warnings,
    })
}

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;

const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_ROW_MAJOR: u32 = 4;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_MATRIX_STRIDE: u32 = 7;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_UNIFORM: u32 = 2;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

#[derive(Debug, Clone)]
enum SpirvType {
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component: u32, count: u32 },
    Matrix { column: u32, count: u32 },
    Image { dimension: u32, sampled: u32 },
    Sampler,
    SampledImage,
    Array { element: u32, length_id: u32 },
    RuntimeArray,
    Struct { members: Vec<u32> },
    Pointer { storage_class: u32, pointee: u32 },
}

#[derive(Debug, Default, Clone, Copy)]
struct Decorations {
    descriptor_set: Option<u32>,
    binding: Option<u32>,
    block: bool,
    buffer_block: bool,
    array_stride: Option<u32>,
    location: Option<u32>,
    built_in: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct MemberDecorations {
    offset: Option<u32>,
    matrix_stride: Option<u32>,
    row_major: bool,
}

#[derive(Debug, Clone, Copy)]
struct Variable {
    pointer_type: u32,
    id: u32,
    storage_class: u32,
}

#[derive(Default)]
struct SpirvModule {
    types: HashMap<u32, SpirvType>,
    constants: HashMap<u32, u64>,
    decorations: HashMap<u32, Decorations>,
    member_decorations: HashMap<(u32, u32), MemberDecorations>,
    variables: Vec<Variable>,
}

fn reflect_spirv(words: &[u32], stage: ShaderStage) -> Result<ShaderReflection, VulkanError> {
    let module = parse_spirv(words)?;
    let mut descriptors = Vec::new();
    let mut vertex_inputs = Vec::new();
    let mut push_constant_size = None;

    for variable in &module.variables {
        let pointer = module
            .types
            .get(&variable.pointer_type)
            .ok_or(VulkanError::InvalidSpirv)?;
        let SpirvType::Pointer {
            storage_class,
            pointee,
        } = pointer
        else {
            return Err(VulkanError::InvalidSpirv);
        };
        if *storage_class != variable.storage_class {
            return Err(VulkanError::InvalidSpirv);
        }

        if variable.storage_class == STORAGE_CLASS_INPUT && stage == ShaderStage::Vertex {
            let decorations = module
                .decorations
                .get(&variable.id)
                .copied()
                .unwrap_or_default();
            if decorations.built_in {
                continue;
            }
            let location = decorations
                .location
                .ok_or(VulkanError::UnsupportedShaderResource)?;
            let format = module.vertex_input_format(*pointee)?;
            vertex_inputs.push(ShaderVertexInput { location, format });
            continue;
        }

        if variable.storage_class == STORAGE_CLASS_PUSH_CONSTANT {
            if push_constant_size.is_some() {
                return Err(VulkanError::InvalidShaderPushConstants);
            }
            let size = module.type_size(*pointee, None, false, 0)?;
            if size == 0 || size % 4 != 0 {
                return Err(VulkanError::InvalidShaderPushConstants);
            }
            push_constant_size = Some(size);
            continue;
        }

        if !matches!(
            variable.storage_class,
            STORAGE_CLASS_UNIFORM_CONSTANT | STORAGE_CLASS_UNIFORM | STORAGE_CLASS_STORAGE_BUFFER
        ) {
            continue;
        }

        let decorations = module
            .decorations
            .get(&variable.id)
            .copied()
            .unwrap_or_default();
        let (Some(set), Some(binding)) = (decorations.descriptor_set, decorations.binding) else {
            return Err(VulkanError::MissingShaderDescriptorBinding);
        };
        let (descriptor_type, descriptor_count) =
            module.descriptor_type(*pointee, variable.storage_class)?;
        descriptors.push(ShaderDescriptorBinding {
            set,
            binding,
            descriptor_type,
            descriptor_count,
            stage: stage.vulkan_flags(),
        });
    }

    descriptors.sort_unstable();
    vertex_inputs.sort_unstable();
    for pair in vertex_inputs.windows(2) {
        if pair[0].location == pair[1].location {
            return Err(VulkanError::UnsupportedShaderResource);
        }
    }
    for pair in descriptors.windows(2) {
        if pair[0].set == pair[1].set && pair[0].binding == pair[1].binding {
            return Err(VulkanError::DuplicateShaderDescriptorBinding);
        }
    }

    Ok(ShaderReflection {
        stage,
        entry_point: "main".to_owned(),
        descriptors,
        vertex_inputs,
        push_constant_size,
    })
}

fn parse_spirv(words: &[u32]) -> Result<SpirvModule, VulkanError> {
    if words.len() < 5
        || words[0] != SPIRV_MAGIC
        || words[1] != SPIRV_VERSION_1_0
        || words[3] == 0
        || words[4] != 0
    {
        return Err(VulkanError::InvalidSpirv);
    }
    let mut module = SpirvModule::default();
    let mut cursor = 5_usize;
    while cursor < words.len() {
        let instruction = words[cursor];
        let word_count = (instruction >> 16) as usize;
        let opcode = (instruction & 0xffff) as u16;
        if word_count == 0
            || cursor
                .checked_add(word_count)
                .is_none_or(|end| end > words.len())
        {
            return Err(VulkanError::InvalidSpirv);
        }
        let operands = &words[cursor + 1..cursor + word_count];
        match opcode {
            OP_TYPE_BOOL if !operands.is_empty() => {
                module.types.insert(operands[0], SpirvType::Bool);
            }
            OP_TYPE_INT if operands.len() >= 3 => {
                module.types.insert(
                    operands[0],
                    SpirvType::Int {
                        width: operands[1],
                        signed: operands[2] != 0,
                    },
                );
            }
            OP_TYPE_FLOAT if operands.len() >= 2 => {
                module
                    .types
                    .insert(operands[0], SpirvType::Float { width: operands[1] });
            }
            OP_TYPE_VECTOR if operands.len() >= 3 => {
                module.types.insert(
                    operands[0],
                    SpirvType::Vector {
                        component: operands[1],
                        count: operands[2],
                    },
                );
            }
            OP_TYPE_MATRIX if operands.len() >= 3 => {
                module.types.insert(
                    operands[0],
                    SpirvType::Matrix {
                        column: operands[1],
                        count: operands[2],
                    },
                );
            }
            OP_TYPE_IMAGE if operands.len() >= 8 => {
                module.types.insert(
                    operands[0],
                    SpirvType::Image {
                        dimension: operands[2],
                        sampled: operands[6],
                    },
                );
            }
            OP_TYPE_SAMPLER if !operands.is_empty() => {
                module.types.insert(operands[0], SpirvType::Sampler);
            }
            OP_TYPE_SAMPLED_IMAGE if operands.len() >= 2 => {
                module.types.insert(operands[0], SpirvType::SampledImage);
            }
            OP_TYPE_ARRAY if operands.len() >= 3 => {
                module.types.insert(
                    operands[0],
                    SpirvType::Array {
                        element: operands[1],
                        length_id: operands[2],
                    },
                );
            }
            OP_TYPE_RUNTIME_ARRAY if operands.len() >= 2 => {
                module.types.insert(operands[0], SpirvType::RuntimeArray);
            }
            OP_TYPE_STRUCT if !operands.is_empty() => {
                module.types.insert(
                    operands[0],
                    SpirvType::Struct {
                        members: operands[1..].to_vec(),
                    },
                );
            }
            OP_TYPE_POINTER if operands.len() >= 3 => {
                module.types.insert(
                    operands[0],
                    SpirvType::Pointer {
                        storage_class: operands[1],
                        pointee: operands[2],
                    },
                );
            }
            OP_CONSTANT if operands.len() >= 3 => {
                let value = if operands.len() >= 4 {
                    u64::from(operands[2]) | (u64::from(operands[3]) << 32)
                } else {
                    u64::from(operands[2])
                };
                module.constants.insert(operands[1], value);
            }
            OP_VARIABLE if operands.len() >= 3 => {
                module.variables.push(Variable {
                    pointer_type: operands[0],
                    id: operands[1],
                    storage_class: operands[2],
                });
            }
            OP_DECORATE if operands.len() >= 2 => {
                let decorations = module.decorations.entry(operands[0]).or_default();
                match operands[1] {
                    DECORATION_BLOCK => decorations.block = true,
                    DECORATION_BUFFER_BLOCK => decorations.buffer_block = true,
                    DECORATION_ARRAY_STRIDE if operands.len() >= 3 => {
                        decorations.array_stride = Some(operands[2]);
                    }
                    DECORATION_BUILT_IN => decorations.built_in = true,
                    DECORATION_LOCATION if operands.len() >= 3 => {
                        decorations.location = Some(operands[2]);
                    }
                    DECORATION_BINDING if operands.len() >= 3 => {
                        decorations.binding = Some(operands[2]);
                    }
                    DECORATION_DESCRIPTOR_SET if operands.len() >= 3 => {
                        decorations.descriptor_set = Some(operands[2]);
                    }
                    _ => {}
                }
            }
            OP_MEMBER_DECORATE if operands.len() >= 3 => {
                let decorations = module
                    .member_decorations
                    .entry((operands[0], operands[1]))
                    .or_default();
                match operands[2] {
                    DECORATION_ROW_MAJOR => decorations.row_major = true,
                    DECORATION_MATRIX_STRIDE if operands.len() >= 4 => {
                        decorations.matrix_stride = Some(operands[3]);
                    }
                    DECORATION_OFFSET if operands.len() >= 4 => {
                        decorations.offset = Some(operands[3]);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        cursor += word_count;
    }
    Ok(module)
}

impl SpirvModule {
    fn descriptor_type(
        &self,
        type_id: u32,
        storage_class: u32,
    ) -> Result<(vk::DescriptorType, u32), VulkanError> {
        let mut current = type_id;
        let mut count = 1_u32;
        loop {
            match self.types.get(&current).ok_or(VulkanError::InvalidSpirv)? {
                SpirvType::Array { element, length_id } => {
                    let length = self
                        .constants
                        .get(length_id)
                        .copied()
                        .filter(|length| *length > 0 && *length <= u64::from(u32::MAX))
                        .ok_or(VulkanError::UnsupportedShaderResource)?;
                    count = count
                        .checked_mul(length as u32)
                        .ok_or(VulkanError::UnsupportedShaderResource)?;
                    current = *element;
                }
                SpirvType::RuntimeArray => {
                    return Err(VulkanError::UnsupportedShaderResource);
                }
                _ => break,
            }
        }

        let ty = self.types.get(&current).ok_or(VulkanError::InvalidSpirv)?;
        let descriptor_type = match (storage_class, ty) {
            (STORAGE_CLASS_UNIFORM_CONSTANT, SpirvType::Sampler) => vk::DescriptorType::SAMPLER,
            (STORAGE_CLASS_UNIFORM_CONSTANT, SpirvType::SampledImage) => {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
            (
                STORAGE_CLASS_UNIFORM_CONSTANT,
                SpirvType::Image {
                    dimension: 0..=4,
                    sampled: 1,
                },
            ) => vk::DescriptorType::SAMPLED_IMAGE,
            (
                STORAGE_CLASS_UNIFORM_CONSTANT,
                SpirvType::Image {
                    dimension: 0..=4,
                    sampled: 2,
                },
            ) => vk::DescriptorType::STORAGE_IMAGE,
            (STORAGE_CLASS_UNIFORM, SpirvType::Struct { .. }) => {
                let decorations = self.decorations.get(&current).copied().unwrap_or_default();
                match (decorations.block, decorations.buffer_block) {
                    (true, false) => vk::DescriptorType::UNIFORM_BUFFER,
                    (false, true) => vk::DescriptorType::STORAGE_BUFFER,
                    _ => return Err(VulkanError::UnsupportedShaderResource),
                }
            }
            (STORAGE_CLASS_STORAGE_BUFFER, SpirvType::Struct { .. }) => {
                vk::DescriptorType::STORAGE_BUFFER
            }
            _ => return Err(VulkanError::UnsupportedShaderResource),
        };
        Ok((descriptor_type, count))
    }

    fn vertex_input_format(&self, type_id: u32) -> Result<vk::Format, VulkanError> {
        let (scalar_type, components) =
            match self.types.get(&type_id).ok_or(VulkanError::InvalidSpirv)? {
                SpirvType::Vector { component, count } => (*component, *count),
                _ => (type_id, 1),
            };
        let scalar = self
            .types
            .get(&scalar_type)
            .ok_or(VulkanError::InvalidSpirv)?;
        let format = match (scalar, components) {
            (SpirvType::Float { width: 32 }, 1) => vk::Format::R32_SFLOAT,
            (SpirvType::Float { width: 32 }, 2) => vk::Format::R32G32_SFLOAT,
            (SpirvType::Float { width: 32 }, 3) => vk::Format::R32G32B32_SFLOAT,
            (SpirvType::Float { width: 32 }, 4) => vk::Format::R32G32B32A32_SFLOAT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: true,
                },
                1,
            ) => vk::Format::R32_SINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: true,
                },
                2,
            ) => vk::Format::R32G32_SINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: true,
                },
                3,
            ) => vk::Format::R32G32B32_SINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: true,
                },
                4,
            ) => vk::Format::R32G32B32A32_SINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: false,
                },
                1,
            ) => vk::Format::R32_UINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: false,
                },
                2,
            ) => vk::Format::R32G32_UINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: false,
                },
                3,
            ) => vk::Format::R32G32B32_UINT,
            (
                SpirvType::Int {
                    width: 32,
                    signed: false,
                },
                4,
            ) => vk::Format::R32G32B32A32_UINT,
            _ => return Err(VulkanError::UnsupportedShaderResource),
        };
        Ok(format)
    }

    fn type_size(
        &self,
        type_id: u32,
        matrix_stride: Option<u32>,
        row_major: bool,
        depth: u32,
    ) -> Result<u32, VulkanError> {
        if depth > 64 {
            return Err(VulkanError::InvalidShaderPushConstants);
        }
        match self.types.get(&type_id).ok_or(VulkanError::InvalidSpirv)? {
            SpirvType::Bool => Ok(4),
            SpirvType::Int { width, .. } | SpirvType::Float { width }
                if *width > 0 && width % 8 == 0 =>
            {
                Ok(width / 8)
            }
            SpirvType::Vector { component, count } => self
                .type_size(*component, None, false, depth + 1)?
                .checked_mul(*count)
                .ok_or(VulkanError::InvalidShaderPushConstants),
            SpirvType::Matrix { column, count } => {
                let (rows, column_size) =
                    match self.types.get(column).ok_or(VulkanError::InvalidSpirv)? {
                        SpirvType::Vector { count, .. } => {
                            (*count, self.type_size(*column, None, false, depth + 1)?)
                        }
                        _ => return Err(VulkanError::InvalidShaderPushConstants),
                    };
                if let Some(stride) = matrix_stride {
                    stride
                        .checked_mul(if row_major { rows } else { *count })
                        .ok_or(VulkanError::InvalidShaderPushConstants)
                } else {
                    column_size
                        .checked_mul(*count)
                        .ok_or(VulkanError::InvalidShaderPushConstants)
                }
            }
            SpirvType::Array { element, length_id } => {
                let length =
                    self.constants
                        .get(length_id)
                        .copied()
                        .filter(|length| *length <= u64::from(u32::MAX))
                        .ok_or(VulkanError::InvalidShaderPushConstants)? as u32;
                let stride = self
                    .decorations
                    .get(&type_id)
                    .and_then(|decorations| decorations.array_stride)
                    .unwrap_or(self.type_size(*element, matrix_stride, row_major, depth + 1)?);
                stride
                    .checked_mul(length)
                    .ok_or(VulkanError::InvalidShaderPushConstants)
            }
            SpirvType::Struct { members } => {
                let mut end = 0_u32;
                for (index, member_type) in members.iter().enumerate() {
                    let decorations = self
                        .member_decorations
                        .get(&(type_id, index as u32))
                        .copied()
                        .unwrap_or_default();
                    let offset = decorations
                        .offset
                        .ok_or(VulkanError::InvalidShaderPushConstants)?;
                    let size = self.type_size(
                        *member_type,
                        decorations.matrix_stride,
                        decorations.row_major,
                        depth + 1,
                    )?;
                    end = end.max(
                        offset
                            .checked_add(size)
                            .ok_or(VulkanError::InvalidShaderPushConstants)?,
                    );
                }
                Ok(end)
            }
            SpirvType::Pointer { pointee, .. } => {
                self.type_size(*pointee, matrix_stride, row_major, depth + 1)
            }
            _ => Err(VulkanError::InvalidShaderPushConstants),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ShaderCompileConfig, ShaderDescriptorBinding, ShaderStage, ShaderVertexInput, compile_glsl,
        parse_spirv,
    };
    use ash::vk;

    #[test]
    fn rejects_malformed_spirv_headers_and_instruction_lengths() {
        assert!(parse_spirv(&[]).is_err());
        assert!(parse_spirv(&[0, 0, 0, 0, 0]).is_err());
        assert!(parse_spirv(&[0x0723_0203, 0x0001_0000, 0, 1, 0, 0]).is_err());
    }

    #[test]
    fn compiles_minimal_vertex_and_fragment_shaders_to_spirv_10() {
        let vertex = compile_glsl(
            "#version 450 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        )
        .expect("valid vertex shader compiles");
        let fragment = compile_glsl(
            "#version 450 core\nlayout(location = 0) out vec4 color;\nvoid main() { color = vec4(1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Fragment),
        )
        .expect("valid fragment shader compiles");
        assert_eq!(vertex.spirv[0], 0x0723_0203);
        assert_eq!(vertex.spirv[1], 0x0001_0000);
        assert_eq!(fragment.spirv[0], 0x0723_0203);
        assert_eq!(fragment.spirv[1], 0x0001_0000);
        assert_eq!(vertex.reflection.stage, ShaderStage::Vertex);
        assert_eq!(fragment.reflection.stage, ShaderStage::Fragment);
    }

    #[test]
    fn reflects_buffer_image_and_sampler_descriptor_types() {
        let source = r#"#version 450 core
layout(set = 0, binding = 0) uniform Camera { mat4 view_projection; } camera;
layout(set = 1, binding = 2) uniform sampler2D base_color;
layout(set = 1, binding = 3) uniform texture2D separate_texture;
layout(set = 1, binding = 4) uniform sampler separate_sampler;
layout(set = 2, binding = 0, rgba8) uniform image2D output_image;
layout(set = 2, binding = 1) buffer StorageData { uint values[]; } storage_data;
void main() { vec4 a = texture(base_color, vec2(0.5)); vec4 b = texture(sampler2D(separate_texture, separate_sampler), vec2(0.5)); imageStore(output_image, ivec2(0), a + b); storage_data.values[0] = 1u; }
"#;
        let shader = compile_glsl(source, &ShaderCompileConfig::new(ShaderStage::Compute))
            .expect("valid descriptor shader compiles");
        let actual: Vec<_> = shader
            .reflection
            .descriptors
            .iter()
            .map(|binding| (binding.set, binding.binding, binding.descriptor_type))
            .collect();
        assert_eq!(
            actual,
            vec![
                (0, 0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, 2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                (1, 3, vk::DescriptorType::SAMPLED_IMAGE),
                (1, 4, vk::DescriptorType::SAMPLER),
                (2, 0, vk::DescriptorType::STORAGE_IMAGE),
                (2, 1, vk::DescriptorType::STORAGE_BUFFER),
            ]
        );
    }

    #[test]
    fn reflects_constant_descriptor_array_lengths_and_push_constants() {
        let source = r#"#version 450 core
layout(set = 0, binding = 0) uniform sampler2D textures[4];
layout(push_constant) uniform PushData { mat4 transform; vec4 tint; } push_data;
void main() { gl_Position = push_data.transform * vec4(0.0, 0.0, 0.0, 1.0); }
"#;
        let shader = compile_glsl(source, &ShaderCompileConfig::new(ShaderStage::Vertex))
            .expect("valid array and push-constant shader compiles");
        assert_eq!(shader.reflection.descriptors.len(), 1);
        assert_eq!(shader.reflection.descriptors[0].descriptor_count, 4);
        assert_eq!(shader.reflection.push_constant_size, Some(80));
    }

    #[test]
    fn rejects_invalid_glsl_and_missing_entry_point() {
        assert!(
            compile_glsl(
                "#version 450 core\nvoid main( {",
                &ShaderCompileConfig::new(ShaderStage::Vertex),
            )
            .is_err()
        );
        assert!(
            compile_glsl(
                "#version 450 core\nvoid other() {}",
                &ShaderCompileConfig::new(ShaderStage::Vertex),
            )
            .is_err()
        );
    }

    #[test]
    fn applies_preprocessor_defines_deterministically() {
        let source = "#version 450 core\nvoid main() { gl_Position = vec4(VALUE, 0.0, 0.0, 1.0); }";
        let mut config = ShaderCompileConfig::new(ShaderStage::Vertex);
        config.defines.insert("VALUE".to_owned(), "0.5".to_owned());
        let shader = compile_glsl(source, &config).expect("defined shader compiles");
        assert_eq!(shader.spirv[0], 0x0723_0203);
    }

    #[test]
    fn reflects_vertex_input_locations_and_formats() {
        let shader = compile_glsl(
            "#version 450 core\nlayout(location = 0) in vec3 position;\nlayout(location = 1) in vec2 uv;\nvoid main() { gl_Position = vec4(position + vec3(uv, 0.0), 1.0); }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        ).expect("vertex inputs compile");
        assert_eq!(
            shader.reflection.vertex_inputs,
            vec![
                ShaderVertexInput {
                    location: 0,
                    format: vk::Format::R32G32B32_SFLOAT
                },
                ShaderVertexInput {
                    location: 1,
                    format: vk::Format::R32G32_SFLOAT
                },
            ]
        );
    }

    #[test]
    fn rejects_descriptor_resources_without_explicit_bindings() {
        assert!(compile_glsl(
            "#version 450 core\nuniform sampler2D missing_binding;\nvoid main() { gl_Position = vec4(0.0); }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        ).is_err());
    }

    #[test]
    fn reflected_descriptor_bindings_include_stage_flags() {
        let shader = compile_glsl(
            "#version 450 core\nlayout(set = 2, binding = 5) uniform Data { vec4 value; } data;\nvoid main() { gl_Position = data.value; }",
            &ShaderCompileConfig::new(ShaderStage::Vertex),
        )
        .expect("shader compiles");
        assert_eq!(
            shader.reflection.descriptors[0],
            ShaderDescriptorBinding {
                set: 2,
                binding: 5,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage: vk::ShaderStageFlags::VERTEX,
            }
        );
    }
}
