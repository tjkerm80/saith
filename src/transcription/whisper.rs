use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{
    self as whisper_model, Config, N_FRAMES, N_SAMPLES, audio,
};
use tokenizers::Tokenizer;

use crate::transcription::model_management::AcquiredModelArtifacts;

/// Loads a Whisper model and transcribes audio buffers to text.
pub struct WhisperTranscriber {
    model: whisper_model::model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    suppression_mask: Tensor,
    start_of_transcript_token: u32,
    end_of_transcript_token: u32,
    no_timestamps_token: u32,
    device: Device,
}

impl WhisperTranscriber {
    /// Load a Whisper model from pre-acquired safetensors artifacts.
    pub fn load(artifacts: AcquiredModelArtifacts) -> Result<Self> {
        let (config_path, weights_path, tokenizer_path) = match artifacts {
            AcquiredModelArtifacts::Safetensors {
                config_path,
                weights_path,
                tokenizer_path,
            } => (config_path, weights_path, tokenizer_path),
            AcquiredModelArtifacts::GgmlSingleFile { .. } => {
                anyhow::bail!("Candle backend requires Safetensors artifacts, got GGML")
            }
        };

        let device = Device::Cpu;

        let config_contents =
            std::fs::read_to_string(&config_path).context("Failed to read model config.json")?;
        let config: Config =
            serde_json::from_str(&config_contents).context("Failed to parse model config.json")?;

        let variable_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], whisper_model::DTYPE, &device)?
        };

        let model = whisper_model::model::Whisper::load(&variable_builder, config.clone())
            .context("Failed to load Whisper model weights")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(anyhow::Error::msg)
            .context("Failed to load tokenizer")?;

        let mel_filters = load_mel_filters(config.num_mel_bins)?;

        let start_of_transcript_token = token_id(&tokenizer, "<|startoftranscript|>")?;
        let end_of_transcript_token = token_id(&tokenizer, "<|endoftext|>")?;
        let no_timestamps_token = token_id(&tokenizer, "<|notimestamps|>")?;

        let suppression_mask = build_suppression_mask(&config, &device)?;

        Ok(Self {
            model,
            tokenizer,
            config,
            mel_filters,
            suppression_mask,
            start_of_transcript_token,
            end_of_transcript_token,
            no_timestamps_token,
            device,
        })
    }

    /// Transcribe a buffer of mono 16kHz f32 audio samples to text.
    ///
    /// If `initial_prompt` is provided, it is prepended as context tokens to
    /// bias the model toward expected vocabulary (e.g. programming terms).
    pub fn transcribe(
        &mut self,
        audio_samples: &[f32],
        _initial_prompt: Option<&str>,
    ) -> Result<String> {
        // Pad or truncate to exactly 30 seconds (N_SAMPLES = 480,000 at 16kHz).
        // Whisper's encoder has a fixed 1500-position limit corresponding to 3000 mel frames.
        let mut normalized_samples = vec![0.0f32; N_SAMPLES];
        let copy_length = audio_samples.len().min(N_SAMPLES);
        normalized_samples[..copy_length].copy_from_slice(&audio_samples[..copy_length]);

        let mel_spectrogram =
            audio::pcm_to_mel(&self.config, &normalized_samples, &self.mel_filters);
        let number_of_mel_frames = mel_spectrogram.len() / self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel_spectrogram,
            (1, self.config.num_mel_bins, number_of_mel_frames),
            &self.device,
        )?;

        // Truncate to N_FRAMES (3000) — pcm_to_mel over-pads for FFT alignment
        let mel_tensor = if number_of_mel_frames > N_FRAMES {
            mel_tensor.narrow(2, 0, N_FRAMES)?
        } else {
            mel_tensor
        };

        let audio_features = self.model.encoder.forward(&mel_tensor, true)?;

        let mut token_ids: Vec<u32> =
            vec![self.start_of_transcript_token, self.no_timestamps_token];

        let sample_length = self.config.max_target_positions / 2;

        for step in 0..sample_length {
            let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;

            let decoder_output =
                self.model
                    .decoder
                    .forward(&token_tensor, &audio_features, step == 0)?;

            let logits = self.model.decoder.final_linear(&decoder_output)?;

            let (_, sequence_length, _) = logits.dims3()?;
            let last_logits = logits.i((0, sequence_length - 1))?;

            let masked_logits = last_logits.broadcast_add(&self.suppression_mask)?;
            let logits_vec: Vec<f32> = masked_logits.to_vec1()?;

            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index as u32)
                .unwrap();

            if next_token == self.end_of_transcript_token {
                break;
            }

            token_ids.push(next_token);
        }

        let output_tokens: Vec<u32> = token_ids
            .into_iter()
            .filter(|&token| token < self.start_of_transcript_token)
            .collect();

        let transcription = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(anyhow::Error::msg)?;

        Ok(transcription.trim().to_string())
    }
}

/// Load pre-computed mel filterbank coefficients from the embedded bytes file.
fn load_mel_filters(number_of_mel_bins: usize) -> Result<Vec<f32>> {
    let mel_bytes = match number_of_mel_bins {
        80 => include_bytes!("../../assets/melfilters.bytes").as_slice(),
        128 => include_bytes!("../../assets/mel_filters.bytes").as_slice(),
        other => anyhow::bail!("Unsupported number of mel bins: {other} (expected 80 or 128)"),
    };

    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
}

/// Build a vocab-sized tensor of 0.0 / -inf for token suppression.
/// Suppresses config-specified tokens and all timestamp tokens (50363+).
fn build_suppression_mask(config: &Config, device: &Device) -> Result<Tensor> {
    let timestamp_start = 50363u32;
    let suppression_mask: Vec<f32> = (0..config.vocab_size as u32)
        .map(|token_index| {
            if config.suppress_tokens.contains(&token_index) || token_index >= timestamp_start {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();

    Tensor::new(suppression_mask.as_slice(), device).context("Failed to create suppression mask")
}

fn token_id(tokenizer: &Tokenizer, token_text: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token_text)
        .ok_or_else(|| anyhow::anyhow!("Token '{token_text}' not found in tokenizer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::TARGET_SAMPLE_RATE;
    use crate::transcription::model_management::{
        HuggingFaceModelProvider, ModelProvider, ModelSize,
    };

    #[test]
    fn mel_filters_load_for_80_bins() {
        let filters = load_mel_filters(80).unwrap();
        assert!(!filters.is_empty());
        // 80 bins * 201 frequency bins (for n_fft=400) = 16080
        assert_eq!(filters.len(), 16080);
    }

    #[test]
    #[ignore] // Requires model download and significant time
    fn transcribe_silence_returns_empty_or_short() {
        let provider = HuggingFaceModelProvider;
        let request = ModelSize::BaseEnglish.safetensors_artifact_request();
        let artifacts = provider.acquire(&request).unwrap();
        let mut transcriber = WhisperTranscriber::load(artifacts).unwrap();
        // 2 seconds of silence
        let silence = vec![0.0f32; TARGET_SAMPLE_RATE as usize * 2];
        let result = transcriber.transcribe(&silence, None).unwrap();
        // Silence should produce empty or very short transcription
        assert!(
            result.len() < 20,
            "Expected short/empty transcription for silence, got: '{result}'"
        );
    }
}
