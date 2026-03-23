#[cfg(not(any(feature = "candle-backend", feature = "whisper-rs-backend")))]
compile_error!(
    "At least one transcription backend must be enabled. \
     Enable the `candle-backend` or `whisper-rs-backend` feature."
);

mod model_management;

#[cfg(feature = "candle-backend")]
mod whisper;

#[cfg(feature = "whisper-rs-backend")]
mod whisper_rs_backend;

use anyhow::Result;

pub use model_management::ModelSize;
use model_management::{HuggingFaceModelProvider, ModelProvider};

/// Which backend the caller prefers. Not feature-gated — both variants always exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    Candle,
    WhisperRs,
}

/// Internal trait implemented by each backend transcriber.
trait TranscriberInternal: Send {
    fn transcribe(&mut self, audio_samples: &[f32], initial_prompt: Option<&str>)
    -> Result<String>;
}

#[cfg(feature = "candle-backend")]
impl TranscriberInternal for whisper::WhisperTranscriber {
    fn transcribe(
        &mut self,
        audio_samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<String> {
        whisper::WhisperTranscriber::transcribe(self, audio_samples, initial_prompt)
    }
}

#[cfg(feature = "whisper-rs-backend")]
impl TranscriberInternal for whisper_rs_backend::WhisperRsTranscriber {
    fn transcribe(
        &mut self,
        audio_samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<String> {
        whisper_rs_backend::WhisperRsTranscriber::transcribe(self, audio_samples, initial_prompt)
    }
}

/// Unified transcription facade. Hides backend selection, feature gates, and model acquisition.
pub struct TranscriptionEngine {
    inner: Box<dyn TranscriberInternal>,
    name: &'static str,
}

impl TranscriptionEngine {
    /// Load a transcription engine for the given model size and backend preference.
    ///
    /// If the preferred backend is not compiled in, falls back to the available one.
    /// Model files are acquired via HuggingFace Hub (downloaded or loaded from cache).
    pub fn load(model_size: ModelSize, backend: BackendPreference) -> Result<Self> {
        Self::load_with_provider(model_size, backend, Box::new(HuggingFaceModelProvider))
    }

    /// Load a transcription engine with an explicit model provider.
    ///
    /// This is the internal workhorse — `load()` delegates here with the default provider.
    /// Accessible to `#[cfg(test)]` code for injecting mock providers.
    fn load_with_provider(
        model_size: ModelSize,
        backend: BackendPreference,
        provider: Box<dyn ModelProvider>,
    ) -> Result<Self> {
        let effective_backend = resolve_effective_backend(backend);

        match effective_backend {
            #[cfg(feature = "candle-backend")]
            BackendPreference::Candle => {
                let request = model_size.safetensors_artifact_request();
                let artifacts = provider.acquire(&request)?;
                let transcriber = whisper::WhisperTranscriber::load(artifacts)?;
                Ok(Self {
                    name: "candle",
                    inner: Box::new(transcriber),
                })
            }
            #[cfg(feature = "whisper-rs-backend")]
            BackendPreference::WhisperRs => {
                let request = model_size.ggml_artifact_request();
                let artifacts = provider.acquire(&request)?;
                let transcriber = whisper_rs_backend::WhisperRsTranscriber::load(artifacts)?;
                Ok(Self {
                    name: "whisper-rs",
                    inner: Box::new(transcriber),
                })
            }
            // Unreachable because resolve_effective_backend guarantees a compiled backend,
            // but the compiler can't prove this across cfg boundaries.
            #[allow(unreachable_patterns)]
            other => anyhow::bail!("Backend {other:?} is not available in this build"),
        }
    }

    /// Human-readable name of the active backend.
    pub fn backend_name(&self) -> &'static str {
        self.name
    }

    /// Transcribe a buffer of mono 16kHz f32 audio samples to text.
    pub fn transcribe(
        &mut self,
        audio_samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<String> {
        self.inner.transcribe(audio_samples, initial_prompt)
    }
}

/// Resolve which backend to actually use, falling back if the preferred one isn't compiled.
fn resolve_effective_backend(preference: BackendPreference) -> BackendPreference {
    match preference {
        #[cfg(feature = "candle-backend")]
        BackendPreference::Candle => BackendPreference::Candle,
        #[cfg(not(feature = "candle-backend"))]
        BackendPreference::Candle => {
            eprintln!("Warning: candle backend not compiled in, falling back to whisper-rs");
            BackendPreference::WhisperRs
        }
        #[cfg(feature = "whisper-rs-backend")]
        BackendPreference::WhisperRs => BackendPreference::WhisperRs,
        #[cfg(not(feature = "whisper-rs-backend"))]
        BackendPreference::WhisperRs => {
            eprintln!("Warning: whisper-rs backend not compiled in, falling back to candle");
            BackendPreference::Candle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_management::{AcquiredModelArtifacts, ModelArtifactRequest};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn resolve_effective_backend_returns_compiled_backend() {
        #[cfg(feature = "candle-backend")]
        assert_eq!(
            resolve_effective_backend(BackendPreference::Candle),
            BackendPreference::Candle
        );

        #[cfg(feature = "whisper-rs-backend")]
        assert_eq!(
            resolve_effective_backend(BackendPreference::WhisperRs),
            BackendPreference::WhisperRs
        );
    }

    struct RecordingMockProvider {
        called: Arc<AtomicBool>,
        artifact_variant: &'static str,
    }

    impl ModelProvider for RecordingMockProvider {
        fn acquire(&self, _request: &ModelArtifactRequest) -> Result<AcquiredModelArtifacts> {
            self.called.store(true, Ordering::SeqCst);
            match self.artifact_variant {
                "safetensors" => Ok(AcquiredModelArtifacts::Safetensors {
                    config_path: PathBuf::from("/nonexistent/config.json"),
                    weights_path: PathBuf::from("/nonexistent/model.safetensors"),
                    tokenizer_path: PathBuf::from("/nonexistent/tokenizer.json"),
                }),
                "ggml" => Ok(AcquiredModelArtifacts::GgmlSingleFile {
                    model_path: PathBuf::from("/nonexistent/model.bin"),
                }),
                other => panic!("Unknown artifact variant: {other}"),
            }
        }
    }

    #[cfg(feature = "candle-backend")]
    #[test]
    fn transcription_engine_load_calls_provider_for_candle() {
        let called = Arc::new(AtomicBool::new(false));
        let provider = RecordingMockProvider {
            called: Arc::clone(&called),
            artifact_variant: "safetensors",
        };

        let result = TranscriptionEngine::load_with_provider(
            ModelSize::BaseEnglish,
            BackendPreference::Candle,
            Box::new(provider),
        );

        assert!(
            called.load(Ordering::SeqCst),
            "ModelProvider should have been called"
        );
        // Load fails because the dummy paths don't contain real model files,
        // but the provider was invoked — no network access occurred.
        assert!(result.is_err());
    }

    #[cfg(feature = "whisper-rs-backend")]
    #[test]
    fn transcription_engine_load_calls_provider_for_whisper_rs() {
        let called = Arc::new(AtomicBool::new(false));
        let provider = RecordingMockProvider {
            called: Arc::clone(&called),
            artifact_variant: "ggml",
        };

        let result = TranscriptionEngine::load_with_provider(
            ModelSize::BaseEnglish,
            BackendPreference::WhisperRs,
            Box::new(provider),
        );

        assert!(
            called.load(Ordering::SeqCst),
            "ModelProvider should have been called"
        );
        assert!(result.is_err());
    }

    struct FailingMockProvider;

    impl ModelProvider for FailingMockProvider {
        fn acquire(&self, _request: &ModelArtifactRequest) -> Result<AcquiredModelArtifacts> {
            anyhow::bail!("simulated provider failure")
        }
    }

    #[test]
    fn load_with_provider_propagates_provider_error() {
        let result = TranscriptionEngine::load_with_provider(
            ModelSize::LargeVersion3Turbo,
            BackendPreference::Candle,
            Box::new(FailingMockProvider),
        );
        match result {
            Ok(_) => panic!("load should fail when provider returns an error"),
            Err(error) => assert!(
                error.to_string().contains("simulated provider failure"),
                "Expected provider error, got: {error}"
            ),
        }
    }
}
