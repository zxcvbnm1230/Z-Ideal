//! Deterministic, dependency-aware system scheduling.

use std::collections::{BTreeSet, HashMap};

use crate::{Commands, World};

/// System scheduling and execution failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScheduleError {
    /// A system name was registered more than once.
    #[error("system name is already registered: {0}")]
    DuplicateSystemName(String),
    /// A declared dependency does not exist in this schedule.
    #[error("system {system} depends on missing system {dependency}")]
    MissingDependency {
        /// System declaring the dependency.
        system: String,
        /// Referenced system that is not registered.
        dependency: String,
    },
    /// System dependencies contain a cycle.
    #[error("system dependency graph contains a cycle: {0:?}")]
    DependencyCycle(Vec<String>),
    /// A system returned an application error.
    #[error("system {system} failed: {message}")]
    SystemFailed {
        /// Name of the failed system.
        system: String,
        /// Error returned by the system.
        message: String,
    },
    /// A deferred command could not be applied.
    #[error(transparent)]
    Command(#[from] crate::CommandError),
}

type SystemRunner = Box<dyn FnMut(&World, &mut Commands) -> Result<(), String> + Send + 'static>;

struct SystemNode {
    name: String,
    after: Vec<String>,
    run: SystemRunner,
}

/// Serial deterministic schedule.
///
/// Systems are ordered by explicit dependencies. When multiple systems are
/// ready at the same time, their registration order breaks ties. The schedule
/// does not run systems in parallel.
#[derive(Default)]
pub struct Schedule {
    systems: Vec<SystemNode>,
}

impl Schedule {
    /// Creates an empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a system that runs after every named dependency.
    ///
    /// Dependency names may refer to systems registered later. Duplicate names
    /// are rejected immediately; missing dependencies and cycles are reported
    /// when the schedule is executed.
    pub fn add_system<F>(
        &mut self,
        name: impl Into<String>,
        after: &[&str],
        run: F,
    ) -> Result<(), ScheduleError>
    where
        F: FnMut(&World, &mut Commands) -> Result<(), String> + Send + 'static,
    {
        let name = name.into();
        if self.systems.iter().any(|system| system.name == name) {
            return Err(ScheduleError::DuplicateSystemName(name));
        }
        self.systems.push(SystemNode {
            name,
            after: after
                .iter()
                .map(|dependency| (*dependency).to_owned())
                .collect(),
            run: Box::new(run),
        });
        Ok(())
    }

    /// Returns system names in their deterministic execution order.
    pub fn execution_order(&self) -> Result<Vec<String>, ScheduleError> {
        self.ordered_indices().map(|indices| {
            indices
                .into_iter()
                .map(|index| self.systems[index].name.clone())
                .collect()
        })
    }

    /// Executes each system serially and applies its command buffer before the
    /// next system begins.
    ///
    /// Commands from a system are discarded if that system returns an error.
    /// Command application itself is FIFO and not transactional; see Commands.
    pub fn run(&mut self, world: &mut World) -> Result<(), ScheduleError> {
        for index in self.ordered_indices()? {
            let system = &mut self.systems[index];
            let mut commands = Commands::new();
            (system.run)(world, &mut commands).map_err(|message| ScheduleError::SystemFailed {
                system: system.name.clone(),
                message,
            })?;
            commands.apply(world)?;
        }
        Ok(())
    }

    fn ordered_indices(&self) -> Result<Vec<usize>, ScheduleError> {
        let mut indices_by_name = HashMap::with_capacity(self.systems.len());
        for (index, system) in self.systems.iter().enumerate() {
            indices_by_name.insert(system.name.as_str(), index);
        }

        let mut outgoing = vec![Vec::new(); self.systems.len()];
        let mut indegree = vec![0_usize; self.systems.len()];
        for (index, system) in self.systems.iter().enumerate() {
            for dependency in &system.after {
                let Some(&dependency_index) = indices_by_name.get(dependency.as_str()) else {
                    return Err(ScheduleError::MissingDependency {
                        system: system.name.clone(),
                        dependency: dependency.clone(),
                    });
                };
                outgoing[dependency_index].push(index);
                indegree[index] += 1;
            }
        }

        let mut ready = BTreeSet::new();
        for (index, degree) in indegree.iter().enumerate() {
            if *degree == 0 {
                ready.insert(index);
            }
        }

        let mut ordered = Vec::with_capacity(self.systems.len());
        while let Some(index) = ready.pop_first() {
            ordered.push(index);
            for &dependent in &outgoing[index] {
                indegree[dependent] -= 1;
                if indegree[dependent] == 0 {
                    ready.insert(dependent);
                }
            }
        }

        if ordered.len() != self.systems.len() {
            let cycle = indegree
                .iter()
                .enumerate()
                .filter(|(_, degree)| **degree > 0)
                .map(|(index, _)| self.systems[index].name.clone())
                .collect();
            return Err(ScheduleError::DependencyCycle(cycle));
        }
        Ok(ordered)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{Schedule, ScheduleError};
    use crate::{Commands, Component, World};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct Marker(u32);

    impl Component for Marker {
        const ID: &'static str = "test.schedule_marker";
    }

    fn world_with_marker() -> World {
        let mut world = World::new();
        world
            .register_component::<Marker>()
            .expect("register marker");
        world
    }

    #[test]
    fn dependency_order_is_respected_even_when_registered_in_reverse() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();

        let render_events = Arc::clone(&events);
        schedule
            .add_system("render", &["update"], move |_, _| {
                render_events.lock().expect("event mutex").push("render");
                Ok(())
            })
            .expect("register render");

        let update_events = Arc::clone(&events);
        schedule
            .add_system("update", &[], move |_, _| {
                update_events.lock().expect("event mutex").push("update");
                Ok(())
            })
            .expect("register update");

        schedule
            .run(&mut World::new())
            .expect("run dependency schedule");
        assert_eq!(
            *events.lock().expect("event mutex"),
            vec!["update", "render"]
        );
    }

    #[test]
    fn independent_systems_keep_registration_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();

        for name in ["first", "second", "third"] {
            let events = Arc::clone(&events);
            schedule
                .add_system(name, &[], move |_, _| {
                    events.lock().expect("event mutex").push(name);
                    Ok(())
                })
                .expect("register independent system");
        }

        schedule
            .run(&mut World::new())
            .expect("run independent systems");
        assert_eq!(
            *events.lock().expect("event mutex"),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn duplicate_system_names_are_rejected() {
        let mut schedule = Schedule::new();
        schedule
            .add_system("update", &[], |_, _| Ok(()))
            .expect("register first update");
        assert_eq!(
            schedule.add_system("update", &[], |_, _| Ok(())),
            Err(ScheduleError::DuplicateSystemName("update".to_owned()))
        );
    }

    #[test]
    fn missing_dependencies_are_reported() {
        let mut schedule = Schedule::new();
        schedule
            .add_system("render", &["update"], |_, _| Ok(()))
            .expect("register render");
        assert_eq!(
            schedule.execution_order(),
            Err(ScheduleError::MissingDependency {
                system: "render".to_owned(),
                dependency: "update".to_owned()
            })
        );
    }

    #[test]
    fn dependency_cycles_are_reported() {
        let mut schedule = Schedule::new();
        schedule
            .add_system("first", &["second"], |_, _| Ok(()))
            .expect("register first");
        schedule
            .add_system("second", &["first"], |_, _| Ok(()))
            .expect("register second");
        assert!(matches!(
            schedule.execution_order(),
            Err(ScheduleError::DependencyCycle(_))
        ));
    }

    #[test]
    fn structural_changes_are_deferred_until_system_barrier() {
        let mut world = world_with_marker();
        let mut schedule = Schedule::new();

        schedule
            .add_system("create", &[], |world, commands| {
                assert!(world.entities().is_empty());
                let pending = commands.spawn().map_err(|error| error.to_string())?;
                commands.insert(pending, Marker(42));
                Ok(())
            })
            .expect("register create");

        schedule
            .add_system("observe", &["create"], |world, _| {
                assert_eq!(world.entities_with::<Marker>().len(), 1);
                assert_eq!(world.get::<Marker>(world.entities()[0]), Some(&Marker(42)));
                Ok(())
            })
            .expect("register observe");

        schedule.run(&mut world).expect("run schedule");
        assert_eq!(world.entities_with::<Marker>().len(), 1);
    }

    #[test]
    fn pending_entities_cannot_cross_command_buffers() {
        let mut world = world_with_marker();
        let mut foreign_commands = Commands::new();
        let foreign_pending = foreign_commands.spawn().expect("reserve foreign entity");
        let mut schedule = Schedule::new();

        schedule
            .add_system("create", &[], move |_, commands| {
                let local_pending = commands.spawn().map_err(|error| error.to_string())?;
                commands.insert(local_pending, Marker(1));
                commands.insert(foreign_pending.clone(), Marker(2));
                Ok(())
            })
            .expect("register create");

        assert!(matches!(
            schedule.run(&mut world),
            Err(ScheduleError::Command(crate::CommandError::PartialApplication {
                cause,
                ..
            })) if matches!(*cause, crate::CommandError::UnresolvedPendingEntity(_))
        ));
        let entities = world.entities_with::<Marker>();
        assert_eq!(entities.len(), 1);
        assert_eq!(world.get::<Marker>(entities[0]), Some(&Marker(1)));
    }

    #[test]
    fn failed_system_discards_its_commands() {
        let mut world = world_with_marker();
        let mut schedule = Schedule::new();
        schedule
            .add_system("failing", &[], |_, commands| {
                let pending = commands.spawn().map_err(|error| error.to_string())?;
                commands.insert(pending, Marker(7));
                Err("deliberate test failure".to_owned())
            })
            .expect("register failing system");

        assert_eq!(
            schedule.run(&mut world),
            Err(ScheduleError::SystemFailed {
                system: "failing".to_owned(),
                message: "deliberate test failure".to_owned()
            })
        );
        assert!(world.entities().is_empty());
    }
}
