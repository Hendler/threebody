pub(crate) mod rollout;

pub(crate) use rollout::{
    SensitivityEval, VectorModel, format_vector_model, rollout_metrics, rollout_trace,
    sensitivity_eval,
};
