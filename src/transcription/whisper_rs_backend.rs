use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::transcription::model_management::AcquiredModelArtifacts;

/// Transcriber backed by whisper-rs (whisper.cpp bindings).
pub struct WhisperRsTranscriber {
    context: WhisperContext,
}

impl WhisperRsTranscriber {
    /// Load a GGML Whisper model from pre-acquired artifacts.
    pub fn load(artifacts: AcquiredModelArtifacts) -> Result<Self> {
        let model_path = match artifacts {
            AcquiredModelArtifacts::GgmlSingleFile { model_path } => model_path,
            AcquiredModelArtifacts::Safetensors { .. } => {
                anyhow::bail!("whisper-rs backend requires GGML artifacts, got Safetensors")
            }
        };

        let context = WhisperContext::new_with_params(
            model_path
                .to_str()
                .context("Model path contains invalid UTF-8")?,
            WhisperContextParameters::default(),
        )
        .map_err(|error| anyhow::anyhow!("Failed to create WhisperContext: {error}"))?;

        Ok(Self { context })
    }

    /// Transcribe a buffer of mono 16kHz f32 audio samples to text.
    ///
    /// If `initial_prompt` is provided, it is set as context to bias the model
    /// toward expected vocabulary (e.g. programming terms).
    pub fn transcribe(
        &mut self,
        audio_samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<String> {
        let mut parameters = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        parameters.set_language(Some("en"));
        parameters.set_translate(false);
        parameters.set_no_timestamps(true);
        parameters.set_print_progress(false);
        parameters.set_print_realtime(false);
        parameters.set_print_special(false);
        parameters.set_print_timestamps(false);

        if let Some(prompt) = initial_prompt {
            parameters.set_initial_prompt(prompt);
        }

        let mut state = self
            .context
            .create_state()
            .map_err(|error| anyhow::anyhow!("Failed to create whisper state: {error}"))?;

        state
            .full(parameters, audio_samples)
            .map_err(|error| anyhow::anyhow!("Whisper transcription failed: {error}"))?;

        let segment_count = state
            .full_n_segments()
            .map_err(|error| anyhow::anyhow!("Failed to get segment count: {error}"))?;

        let mut transcription = String::new();
        for index in 0..segment_count {
            let segment_text = state
                .full_get_segment_text(index)
                .map_err(|error| anyhow::anyhow!("Failed to get segment {index} text: {error}"))?;
            transcription.push_str(&segment_text);
        }

        Ok(transcription.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::TARGET_SAMPLE_RATE;
    use crate::transcription::model_management::{
        HuggingFaceModelProvider, ModelProvider, ModelSize,
    };

    #[test]
    #[ignore] // Requires model download and significant time
    fn transcribe_silence_returns_empty_or_short() {
        let provider = HuggingFaceModelProvider;
        let request = ModelSize::BaseEnglish.ggml_artifact_request();
        let artifacts = provider.acquire(&request).unwrap();
        let mut transcriber = WhisperRsTranscriber::load(artifacts).unwrap();
        // 2 seconds of silence
        let silence = vec![0.0f32; TARGET_SAMPLE_RATE as usize * 2];
        let result = transcriber.transcribe(&silence, None).unwrap();
        assert!(
            result.len() < 20,
            "Expected short/empty transcription for silence, got: '{result}'"
        );
    }
}
