use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::fmt;
use std::path::PathBuf;

/// Supported Whisper model sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    BaseEnglish,
    SmallEnglish,
    LargeVersion3Turbo,
}

impl ModelSize {
    /// HuggingFace repository ID for the candle (safetensors) model.
    #[cfg_attr(not(feature = "candle-backend"), allow(dead_code))]
    pub(super) fn repository_id_for_safetensors(&self) -> &'static str {
        match self {
            ModelSize::BaseEnglish => "openai/whisper-base.en",
            ModelSize::SmallEnglish => "openai/whisper-small.en",
            ModelSize::LargeVersion3Turbo => "openai/whisper-large-v3-turbo",
        }
    }

    /// GGML model filename in the `ggerganov/whisper.cpp` HuggingFace repo.
    #[cfg_attr(not(feature = "whisper-rs-backend"), allow(dead_code))]
    pub(super) fn ggml_filename(&self) -> &'static str {
        match self {
            ModelSize::BaseEnglish => "ggml-base.en.bin",
            ModelSize::SmallEnglish => "ggml-small.en.bin",
            ModelSize::LargeVersion3Turbo => "ggml-large-v3-turbo.bin",
        }
    }

    /// Build a [`ModelArtifactRequest`] for the candle (safetensors) backend.
    #[cfg_attr(not(feature = "candle-backend"), allow(dead_code))]
    pub(super) fn safetensors_artifact_request(&self) -> ModelArtifactRequest {
        ModelArtifactRequest::Safetensors {
            repository_id: self.repository_id_for_safetensors().to_string(),
            revision: "main".to_string(),
            filenames: vec![
                "config.json".to_string(),
                "model.safetensors".to_string(),
                "tokenizer.json".to_string(),
            ],
        }
    }

    /// Build a [`ModelArtifactRequest`] for the whisper-rs (GGML) backend.
    #[cfg_attr(not(feature = "whisper-rs-backend"), allow(dead_code))]
    pub(super) fn ggml_artifact_request(&self) -> ModelArtifactRequest {
        ModelArtifactRequest::GgmlSingleFile {
            repository_id: "ggerganov/whisper.cpp".to_string(),
            revision: "main".to_string(),
            filename: self.ggml_filename().to_string(),
        }
    }
}

impl fmt::Display for ModelSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSize::BaseEnglish => write!(formatter, "base.en"),
            ModelSize::SmallEnglish => write!(formatter, "small.en"),
            ModelSize::LargeVersion3Turbo => write!(formatter, "large-v3-turbo"),
        }
    }
}

/// What kind of model artifacts are needed.
#[cfg_attr(
    not(all(feature = "candle-backend", feature = "whisper-rs-backend")),
    allow(dead_code)
)]
pub(super) enum ModelArtifactRequest {
    /// Safetensors format: config, weights, and tokenizer files from a HuggingFace repo.
    Safetensors {
        repository_id: String,
        revision: String,
        filenames: Vec<String>,
    },
    /// Single GGML binary file from a HuggingFace repo.
    GgmlSingleFile {
        repository_id: String,
        revision: String,
        filename: String,
    },
}

/// Resolved local paths to acquired model artifacts.
#[cfg_attr(
    not(all(feature = "candle-backend", feature = "whisper-rs-backend")),
    allow(dead_code)
)]
pub(super) enum AcquiredModelArtifacts {
    /// Paths to safetensors model files (config, weights, tokenizer).
    Safetensors {
        config_path: PathBuf,
        weights_path: PathBuf,
        tokenizer_path: PathBuf,
    },
    /// Path to a single GGML model binary.
    GgmlSingleFile { model_path: PathBuf },
}

/// Acquires model artifacts (e.g. by downloading or locating cached files).
pub(super) trait ModelProvider: Send {
    fn acquire(&self, request: &ModelArtifactRequest) -> Result<AcquiredModelArtifacts>;
}

/// Default [`ModelProvider`] that downloads models via HuggingFace Hub.
pub(super) struct HuggingFaceModelProvider;

impl ModelProvider for HuggingFaceModelProvider {
    fn acquire(&self, request: &ModelArtifactRequest) -> Result<AcquiredModelArtifacts> {
        let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;

        match request {
            ModelArtifactRequest::Safetensors {
                repository_id,
                revision,
                filenames,
            } => {
                let repository = api.repo(Repo::with_revision(
                    repository_id.clone(),
                    RepoType::Model,
                    revision.clone(),
                ));

                let config_path = repository
                    .get(&filenames[0])
                    .with_context(|| format!("Failed to download {}", filenames[0]))?;
                let weights_path = repository
                    .get(&filenames[1])
                    .with_context(|| format!("Failed to download {}", filenames[1]))?;
                let tokenizer_path = repository
                    .get(&filenames[2])
                    .with_context(|| format!("Failed to download {}", filenames[2]))?;

                Ok(AcquiredModelArtifacts::Safetensors {
                    config_path,
                    weights_path,
                    tokenizer_path,
                })
            }
            ModelArtifactRequest::GgmlSingleFile {
                repository_id,
                revision,
                filename,
            } => {
                let repository = api.repo(Repo::with_revision(
                    repository_id.clone(),
                    RepoType::Model,
                    revision.clone(),
                ));

                let model_path = repository
                    .get(filename)
                    .with_context(|| format!("Failed to download {filename}"))?;

                Ok(AcquiredModelArtifacts::GgmlSingleFile { model_path })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_size_repository_ids_are_valid() {
        assert_eq!(
            ModelSize::BaseEnglish.repository_id_for_safetensors(),
            "openai/whisper-base.en"
        );
        assert_eq!(
            ModelSize::SmallEnglish.repository_id_for_safetensors(),
            "openai/whisper-small.en"
        );
        assert_eq!(
            ModelSize::LargeVersion3Turbo.repository_id_for_safetensors(),
            "openai/whisper-large-v3-turbo"
        );
    }

    #[test]
    fn model_size_display() {
        assert_eq!(format!("{}", ModelSize::BaseEnglish), "base.en");
        assert_eq!(format!("{}", ModelSize::SmallEnglish), "small.en");
        assert_eq!(
            format!("{}", ModelSize::LargeVersion3Turbo),
            "large-v3-turbo"
        );
    }

    #[test]
    fn ggml_model_filenames_are_valid() {
        assert_eq!(ModelSize::BaseEnglish.ggml_filename(), "ggml-base.en.bin");
        assert_eq!(ModelSize::SmallEnglish.ggml_filename(), "ggml-small.en.bin");
        assert_eq!(
            ModelSize::LargeVersion3Turbo.ggml_filename(),
            "ggml-large-v3-turbo.bin"
        );
    }

    #[test]
    fn model_size_safetensors_artifact_request() {
        let request = ModelSize::BaseEnglish.safetensors_artifact_request();
        match request {
            ModelArtifactRequest::Safetensors {
                repository_id,
                revision,
                filenames,
            } => {
                assert_eq!(repository_id, "openai/whisper-base.en");
                assert_eq!(revision, "main");
                assert_eq!(filenames.len(), 3);
                assert!(filenames.contains(&"config.json".to_string()));
                assert!(filenames.contains(&"model.safetensors".to_string()));
                assert!(filenames.contains(&"tokenizer.json".to_string()));
            }
            _ => panic!("Expected Safetensors variant"),
        }
    }

    #[test]
    fn model_size_ggml_artifact_request() {
        let request = ModelSize::BaseEnglish.ggml_artifact_request();
        match request {
            ModelArtifactRequest::GgmlSingleFile {
                repository_id,
                revision,
                filename,
            } => {
                assert_eq!(repository_id, "ggerganov/whisper.cpp");
                assert_eq!(revision, "main");
                assert_eq!(filename, "ggml-base.en.bin");
            }
            _ => panic!("Expected GgmlSingleFile variant"),
        }
    }

    #[test]
    #[ignore] // Requires network access
    fn hugging_face_provider_acquire_safetensors() {
        let provider = HuggingFaceModelProvider;
        let request = ModelSize::BaseEnglish.safetensors_artifact_request();
        let artifacts = provider.acquire(&request).unwrap();
        match artifacts {
            AcquiredModelArtifacts::Safetensors {
                config_path,
                weights_path,
                tokenizer_path,
            } => {
                assert!(config_path.exists());
                assert!(weights_path.exists());
                assert!(tokenizer_path.exists());
            }
            _ => panic!("Expected Safetensors artifacts"),
        }
    }

    #[test]
    #[ignore] // Requires network access
    fn hugging_face_provider_acquire_ggml() {
        let provider = HuggingFaceModelProvider;
        let request = ModelSize::BaseEnglish.ggml_artifact_request();
        let artifacts = provider.acquire(&request).unwrap();
        match artifacts {
            AcquiredModelArtifacts::GgmlSingleFile { model_path } => {
                assert!(model_path.exists());
            }
            _ => panic!("Expected GgmlSingleFile artifacts"),
        }
    }
}
