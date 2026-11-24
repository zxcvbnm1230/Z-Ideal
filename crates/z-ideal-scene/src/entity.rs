//! Generation-safe runtime entity identifiers.

/// Runtime entity handle.
///
/// The index selects a world slot; the generation prevents a stale handle from
/// addressing a different entity after that slot is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    pub(crate) const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns the slot index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the slot generation.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) const fn from_parts_for_test(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}
