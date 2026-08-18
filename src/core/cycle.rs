/// One externally visible turn requested by a role cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step<Query, Action> {
    Observe(Query),
    Act(Action),
    Finish,
}
