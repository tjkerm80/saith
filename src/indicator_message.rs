use crate::pipeline_state::PipelineState;

/// Messages sent from the pipeline thread to the indicator over the crossbeam channel.
///
/// Combines pipeline state transitions with real-time audio data so the indicator
/// can display both status changes and a live waveform visualization.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "indicator"), allow(dead_code))]
pub enum IndicatorMessage {
    /// The pipeline moved to a new state (recording, processing, transcribing, idle).
    StateChanged(PipelineState),
    /// A peak amplitude sample from the audio capture callback, in the range `0.0..=1.0`.
    /// Sent approximately 20 times per second during recording.
    WaveformSample(f32),
}
