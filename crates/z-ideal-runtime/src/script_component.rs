//! Serializable scene component describing scripts attached by logical asset URI.
use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use z_ideal_assets::AssetUri;
use z_ideal_scene::Component;

/// Maximum script references accepted on one entity.
pub const MAX_SCRIPT_REFERENCES_PER_ENTITY: usize = 64;

/// One script asset reference on an entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptReference {
    /// Project-relative logical URI of the cooked script asset.
    pub uri: AssetUri,
    /// Whether the runtime should attach and execute this script.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl ScriptReference {
    /// Creates an enabled script reference.
    #[must_use]
    pub fn new(uri: AssetUri) -> Self {
        Self { uri, enabled: true }
    }

    /// Creates a script reference with an explicit enabled state.
    #[must_use]
    pub const fn with_enabled(uri: AssetUri, enabled: bool) -> Self {
        Self { uri, enabled }
    }
}

const fn default_enabled() -> bool {
    true
}

/// Scene component listing the scripts to attach to an entity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptSet {
    /// Script references in deterministic attachment order.
    pub scripts: Vec<ScriptReference>,
}

impl ScriptSet {
    /// Creates and validates an ordered set of script references.
    pub fn new(scripts: Vec<ScriptReference>) -> Result<Self, String> {
        let value = Self { scripts };
        value.validate()?;
        Ok(value)
    }
}

impl Component for ScriptSet {
    const ID: &'static str = "z-ideal.runtime.script_set";

    fn validate(&self) -> Result<(), String> {
        if self.scripts.len() > MAX_SCRIPT_REFERENCES_PER_ENTITY {
            return Err(format!(
                "script set contains {} references; maximum is {}",
                self.scripts.len(),
                MAX_SCRIPT_REFERENCES_PER_ENTITY
            ));
        }
        let mut unique = HashSet::with_capacity(self.scripts.len());
        for reference in &self.scripts {
            if !unique.insert(&reference.uri) {
                return Err(format!(
                    "script set contains duplicate URI {}",
                    reference.uri
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use z_ideal_scene::World;

    fn reference(uri: &str) -> ScriptReference {
        ScriptReference::new(AssetUri::new(uri).expect("valid URI"))
    }

    #[test]
    fn validates_unique_bounded_script_references() {
        assert!(ScriptSet::new(vec![reference("a.lua"), reference("b.lua")]).is_ok());
        assert!(ScriptSet::new(vec![reference("a.lua"), reference("a.lua")]).is_err());
        assert!(
            ScriptSet::new(
                (0..=MAX_SCRIPT_REFERENCES_PER_ENTITY)
                    .map(|index| reference(&format!("{index}.lua")))
                    .collect()
            )
            .is_err()
        );
    }

    #[test]
    fn omitted_enabled_field_defaults_to_true_for_older_scene_data() {
        let reference: ScriptReference =
            serde_json::from_str(r#"{"uri":"legacy.lua"}"#).expect("legacy reference");
        assert!(reference.enabled);
    }

    #[test]
    fn component_round_trips_through_scene_serialization() {
        let mut world = World::new();
        world.register_component::<ScriptSet>().expect("register");
        let entity = world.spawn().expect("spawn");
        let scripts = ScriptSet::new(vec![
            reference("player.lua"),
            ScriptReference::with_enabled(AssetUri::new("disabled.lua").expect("URI"), false),
        ])
        .expect("valid script set");
        world.insert(entity, scripts.clone()).expect("insert");

        let serialized = world.to_scene_json().expect("serialize");
        let mut restored = World::new();
        restored
            .register_component::<ScriptSet>()
            .expect("register");
        restored.load_scene_json(&serialized).expect("load");
        let restored_entity = restored.entities_with::<ScriptSet>()[0];
        assert_eq!(restored.get::<ScriptSet>(restored_entity), Some(&scripts));
    }
}
