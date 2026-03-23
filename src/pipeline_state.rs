/// Represents the current phase of the speech-to-text pipeline.
/// Sent from the pipeline thread to the indicator via crossbeam channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    /// No active operation. Indicator should hide.
    Idle,
    /// Microphone is capturing audio.
    Recording,
    /// Audio captured, performing VAD trimming and resampling.
    Processing,
    /// Whisper model is transcribing audio to text.
    Transcribing,
}
