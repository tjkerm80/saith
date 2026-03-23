use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::audio::pipeline::VoiceActivityDetectionConfig;
use crate::transcription::BackendPreference;
use crate::transcription::ModelSize;

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Configuration {
    pub model_size: ModelSizeConfiguration,
    pub backend: TranscriptionBackendConfiguration,
    pub interaction_mode: InteractionMode,
    pub audio_device: Option<String>,
    pub model_path_override: Option<PathBuf>,
    pub dictionary: DictionaryConfiguration,
    pub hotkey: HotkeyConfiguration,
    pub indicator: IndicatorConfiguration,
    pub voice_activity_detection: VoiceActivityDetectionConfiguration,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            model_size: ModelSizeConfiguration::default(),
            backend: TranscriptionBackendConfiguration::default(),
            interaction_mode: InteractionMode::PushToTalk,
            audio_device: None,
            model_path_override: None,
            dictionary: DictionaryConfiguration::default(),
            hotkey: HotkeyConfiguration::default(),
            indicator: IndicatorConfiguration::default(),
            voice_activity_detection: VoiceActivityDetectionConfiguration::default(),
        }
    }
}

impl Configuration {
    /// Resolve the ModelSize enum from the configuration string.
    pub fn resolved_model_size(&self) -> ModelSize {
        self.model_size.resolve()
    }

    /// Resolve the backend preference from configuration.
    pub fn resolved_backend_preference(&self) -> BackendPreference {
        self.backend.resolve()
    }
}

/// Serializable wrapper for ModelSize since the enum lives in another module.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSizeConfiguration {
    #[default]
    BaseEnglish,
    SmallEnglish,
    LargeVersion3Turbo,
}

impl ModelSizeConfiguration {
    pub fn resolve(&self) -> ModelSize {
        match self {
            ModelSizeConfiguration::BaseEnglish => ModelSize::BaseEnglish,
            ModelSizeConfiguration::SmallEnglish => ModelSize::SmallEnglish,
            ModelSizeConfiguration::LargeVersion3Turbo => ModelSize::LargeVersion3Turbo,
        }
    }
}

/// Which transcription backend to use.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionBackendConfiguration {
    #[default]
    Candle,
    WhisperRs,
}

impl TranscriptionBackendConfiguration {
    /// Resolve to a [`BackendPreference`]. Fallback logic for unavailable
    /// backends is handled by [`crate::transcription::TranscriptionEngine::load`].
    pub fn resolve(&self) -> BackendPreference {
        match self {
            TranscriptionBackendConfiguration::Candle => BackendPreference::Candle,
            TranscriptionBackendConfiguration::WhisperRs => BackendPreference::WhisperRs,
        }
    }
}

/// Screen position preset for the floating status indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndicatorPosition {
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

/// Configuration for the floating status indicator overlay.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndicatorConfiguration {
    /// Whether to show the floating indicator window.
    pub show: bool,
    /// Screen position preset for the indicator pill.
    pub position: IndicatorPosition,
}

impl Default for IndicatorConfiguration {
    fn default() -> Self {
        Self {
            show: true,
            position: IndicatorPosition::BottomCenter,
        }
    }
}

/// Configuration for voice activity detection thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceActivityDetectionConfiguration {
    /// RMS energy threshold below which audio is considered silence.
    pub energy_threshold: f32,
    /// How many seconds of continuous silence triggers end-of-speech.
    pub silence_duration_limit: f32,
    /// Minimum duration of speech (in seconds) to consider a recording valid.
    pub minimum_speech_duration: f32,
}

impl Default for VoiceActivityDetectionConfiguration {
    fn default() -> Self {
        Self {
            energy_threshold: 0.002,
            silence_duration_limit: 1.5,
            minimum_speech_duration: 0.1,
        }
    }
}

impl From<&VoiceActivityDetectionConfiguration> for VoiceActivityDetectionConfig {
    fn from(configuration: &VoiceActivityDetectionConfiguration) -> Self {
        Self {
            energy_threshold: configuration.energy_threshold,
            silence_duration_limit: std::time::Duration::from_secs_f32(
                configuration.silence_duration_limit,
            ),
            minimum_speech_duration: std::time::Duration::from_secs_f32(
                configuration.minimum_speech_duration,
            ),
        }
    }
}

/// How the user activates dictation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionMode {
    PushToTalk,
    ToggleToTalk,
}

/// Dictionary configuration for biasing transcription output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DictionaryConfiguration {
    /// Whisper initial_prompt string to bias toward expected vocabulary.
    pub initial_prompt: Option<String>,
    /// Post-processing text replacements applied to transcription output.
    pub replacement_rules: Vec<ReplacementRule>,
}

/// Hotkey configuration for the global keyboard shortcut.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HotkeyConfiguration {
    /// The evdev key code name (e.g. "KEY_RIGHTMETA", "KEY_F12").
    pub key_code: String,
}

impl Default for HotkeyConfiguration {
    fn default() -> Self {
        Self {
            key_code: "KEY_RIGHTMETA".to_string(),
        }
    }
}

/// A find-and-replace rule applied to transcription output.
///
/// The `pattern` field accepts either a single string or an array of strings.
/// All patterns are replaced with the same replacement text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplacementRule {
    #[serde(
        deserialize_with = "deserialize_patterns",
        serialize_with = "serialize_patterns"
    )]
    pub pattern: Vec<String>,
    pub replacement: String,
}

fn deserialize_patterns<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct PatternsVisitor;

    impl<'de> de::Visitor<'de> for PatternsVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of strings")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Vec<String>, E> {
            Ok(vec![value.to_string()])
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Vec<String>, A::Error> {
            let mut patterns = Vec::new();
            while let Some(value) = sequence.next_element()? {
                patterns.push(value);
            }
            Ok(patterns)
        }
    }

    deserializer.deserialize_any(PatternsVisitor)
}

fn serialize_patterns<S>(patterns: &[String], serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if patterns.len() == 1 {
        serializer.serialize_str(&patterns[0])
    } else {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(Some(patterns.len()))?;
        for pattern in patterns {
            sequence.serialize_element(pattern)?;
        }
        sequence.end()
    }
}

impl ReplacementRule {
    /// Apply this replacement rule to the given text.
    pub fn apply(&self, text: &str) -> String {
        let mut result = text.to_string();
        for pattern in &self.pattern {
            result = result.replace(pattern.as_str(), &self.replacement);
        }
        result
    }
}

/// Apply all replacement rules from a dictionary configuration to the given text.
pub fn apply_replacement_rules(dictionary: &DictionaryConfiguration, text: &str) -> String {
    let mut result = text.to_string();
    for rule in &dictionary.replacement_rules {
        result = rule.apply(&result);
    }
    result
}

/// Path to the configuration file: ~/.config/saith/config.json
fn configuration_file_path() -> Result<PathBuf> {
    let config_directory = dirs::config_dir()
        .context("Could not determine XDG config directory")?
        .join("saith");
    Ok(config_directory.join("config.json"))
}

/// Load configuration from disk. Returns defaults if the file does not exist.
/// Creates the config directory and writes defaults if missing.
pub fn load_configuration() -> Result<Configuration> {
    let path = configuration_file_path()?;
    load_configuration_from_path(&path)
}

/// Load configuration from a specific path. Returns defaults if the file does not exist.
pub fn load_configuration_from_path(path: &Path) -> Result<Configuration> {
    if !path.exists() {
        let configuration = Configuration::default();
        save_configuration_to_path(&configuration, path)?;
        return Ok(configuration);
    }

    let contents = std::fs::read_to_string(path).context("Failed to read configuration file")?;
    let configuration: Configuration =
        serde_json::from_str(&contents).context("Failed to parse configuration file")?;
    Ok(configuration)
}

/// Save configuration to a specific path, creating parent directories if needed.
fn save_configuration_to_path(configuration: &Configuration, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create configuration directory")?;
    }
    let contents =
        serde_json::to_string_pretty(configuration).context("Failed to serialize configuration")?;
    std::fs::write(path, contents).context("Failed to write configuration file")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn default_configuration_has_expected_values() {
        let configuration = Configuration::default();
        assert!(matches!(
            configuration.model_size,
            ModelSizeConfiguration::BaseEnglish
        ));
        assert!(matches!(
            configuration.backend,
            TranscriptionBackendConfiguration::Candle
        ));
        assert_eq!(configuration.interaction_mode, InteractionMode::PushToTalk);
        assert!(configuration.audio_device.is_none());
        assert!(configuration.model_path_override.is_none());
        assert!(configuration.dictionary.initial_prompt.is_none());
        assert!(configuration.dictionary.replacement_rules.is_empty());
        assert_eq!(configuration.hotkey.key_code, "KEY_RIGHTMETA");
        assert!(configuration.indicator.show);
        assert_eq!(
            configuration.indicator.position,
            IndicatorPosition::BottomCenter
        );
    }

    #[test]
    fn round_trip_serialization() {
        let configuration = Configuration {
            model_size: ModelSizeConfiguration::LargeVersion3Turbo,
            backend: TranscriptionBackendConfiguration::WhisperRs,
            interaction_mode: InteractionMode::ToggleToTalk,
            audio_device: Some("hw:1,0".to_string()),
            model_path_override: Some(PathBuf::from("/custom/models")),
            dictionary: DictionaryConfiguration {
                initial_prompt: Some("Claude, Anthropic, Rust, Candle".to_string()),
                replacement_rules: vec![ReplacementRule {
                    pattern: vec!["chat gpt".to_string()],
                    replacement: "ChatGPT".to_string(),
                }],
            },
            hotkey: HotkeyConfiguration {
                key_code: "KEY_F12".to_string(),
            },
            indicator: IndicatorConfiguration {
                show: false,
                position: IndicatorPosition::TopRight,
            },
            voice_activity_detection: VoiceActivityDetectionConfiguration {
                energy_threshold: 0.005,
                silence_duration_limit: 2.0,
                minimum_speech_duration: 0.2,
            },
        };

        let json = serde_json::to_string_pretty(&configuration).unwrap();
        let deserialized: Configuration = serde_json::from_str(&json).unwrap();

        assert!(matches!(
            deserialized.model_size,
            ModelSizeConfiguration::LargeVersion3Turbo
        ));
        assert!(matches!(
            deserialized.backend,
            TranscriptionBackendConfiguration::WhisperRs
        ));
        assert_eq!(deserialized.interaction_mode, InteractionMode::ToggleToTalk);
        assert_eq!(deserialized.audio_device.as_deref(), Some("hw:1,0"));
        assert_eq!(
            deserialized.dictionary.initial_prompt.as_deref(),
            Some("Claude, Anthropic, Rust, Candle")
        );
        assert_eq!(deserialized.dictionary.replacement_rules.len(), 1);
        assert_eq!(deserialized.hotkey.key_code, "KEY_F12");
        assert!(!deserialized.indicator.show);
        assert_eq!(deserialized.indicator.position, IndicatorPosition::TopRight);
    }

    #[test]
    fn backend_configuration_round_trip() {
        let candle_json =
            serde_json::to_string(&TranscriptionBackendConfiguration::Candle).unwrap();
        assert_eq!(candle_json, r#""candle""#);

        let whisper_rs_json =
            serde_json::to_string(&TranscriptionBackendConfiguration::WhisperRs).unwrap();
        assert_eq!(whisper_rs_json, r#""whisper_rs""#);

        let deserialized: TranscriptionBackendConfiguration =
            serde_json::from_str(r#""whisper_rs""#).unwrap();
        assert!(matches!(
            deserialized,
            TranscriptionBackendConfiguration::WhisperRs
        ));
    }

    #[test]
    fn missing_file_returns_defaults_and_creates_file() {
        let temporary_directory = TempDir::new().unwrap();
        let path = temporary_directory.path().join("saith").join("config.json");

        let configuration = load_configuration_from_path(&path).unwrap();

        assert!(matches!(
            configuration.model_size,
            ModelSizeConfiguration::BaseEnglish
        ));
        assert!(path.exists(), "Config file should have been created");
    }

    #[test]
    fn partial_json_fills_defaults() {
        let temporary_directory = TempDir::new().unwrap();
        let path = temporary_directory.path().join("config.json");

        std::fs::write(&path, r#"{"interaction_mode": "toggle_to_talk"}"#).unwrap();

        let configuration = load_configuration_from_path(&path).unwrap();

        assert_eq!(
            configuration.interaction_mode,
            InteractionMode::ToggleToTalk
        );
        // Everything else should be defaults
        assert!(matches!(
            configuration.model_size,
            ModelSizeConfiguration::BaseEnglish
        ));
        assert!(configuration.dictionary.replacement_rules.is_empty());
    }

    #[test]
    fn replacement_rules_apply_correctly() {
        let dictionary = DictionaryConfiguration {
            initial_prompt: None,
            replacement_rules: vec![
                ReplacementRule {
                    pattern: vec!["chat gpt".to_string()],
                    replacement: "ChatGPT".to_string(),
                },
                ReplacementRule {
                    pattern: vec!["open ai".to_string()],
                    replacement: "OpenAI".to_string(),
                },
            ],
        };

        let input = "I used chat gpt from open ai to help.";
        let result = apply_replacement_rules(&dictionary, input);
        assert_eq!(result, "I used ChatGPT from OpenAI to help.");
    }

    #[test]
    fn replacement_rule_with_multiple_patterns() {
        let dictionary = DictionaryConfiguration {
            initial_prompt: None,
            replacement_rules: vec![ReplacementRule {
                pattern: vec![
                    "system D".to_string(),
                    "Systemd".to_string(),
                    "System D".to_string(),
                ],
                replacement: "systemd".to_string(),
            }],
        };

        let input = "I configured system D and Systemd services with System D tools.";
        let result = apply_replacement_rules(&dictionary, input);
        assert_eq!(
            result,
            "I configured systemd and systemd services with systemd tools."
        );
    }

    #[test]
    fn empty_replacement_rules_return_original_text() {
        let dictionary = DictionaryConfiguration::default();
        let input = "Hello world";
        let result = apply_replacement_rules(&dictionary, input);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn indicator_position_serialization_round_trip() {
        let positions = [
            (IndicatorPosition::TopLeft, r#""top_left""#),
            (IndicatorPosition::TopCenter, r#""top_center""#),
            (IndicatorPosition::TopRight, r#""top_right""#),
            (IndicatorPosition::BottomLeft, r#""bottom_left""#),
            (IndicatorPosition::BottomCenter, r#""bottom_center""#),
            (IndicatorPosition::BottomRight, r#""bottom_right""#),
        ];

        for (position, expected_json) in positions {
            let json = serde_json::to_string(&position).unwrap();
            assert_eq!(json, expected_json);
            let deserialized: IndicatorPosition = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, position);
        }
    }

    #[test]
    fn pattern_deserializes_from_single_string() {
        let json = r#"{"pattern": "hello", "replacement": "world"}"#;
        let rule: ReplacementRule = serde_json::from_str(json).unwrap();
        assert_eq!(rule.pattern, vec!["hello".to_string()]);
    }

    #[test]
    fn pattern_deserializes_from_array() {
        let json = r#"{"pattern": ["hello", "hi"], "replacement": "world"}"#;
        let rule: ReplacementRule = serde_json::from_str(json).unwrap();
        assert_eq!(rule.pattern, vec!["hello".to_string(), "hi".to_string()]);
    }

    #[test]
    fn single_pattern_serializes_as_string() {
        let rule = ReplacementRule {
            pattern: vec!["hello".to_string()],
            replacement: "world".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        assert_eq!(json, r#"{"pattern":"hello","replacement":"world"}"#);
    }

    #[test]
    fn multiple_patterns_serialize_as_array() {
        let rule = ReplacementRule {
            pattern: vec!["hello".to_string(), "hi".to_string()],
            replacement: "world".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        assert_eq!(json, r#"{"pattern":["hello","hi"],"replacement":"world"}"#);
    }

    #[test]
    fn backend_configuration_resolves_to_backend_preference() {
        assert_eq!(
            TranscriptionBackendConfiguration::Candle.resolve(),
            BackendPreference::Candle
        );
        assert_eq!(
            TranscriptionBackendConfiguration::WhisperRs.resolve(),
            BackendPreference::WhisperRs
        );
    }

    #[test]
    fn existing_config_without_indicator_gets_defaults() {
        let temporary_directory = TempDir::new().unwrap();
        let path = temporary_directory.path().join("config.json");

        std::fs::write(&path, r#"{"interaction_mode": "push_to_talk"}"#).unwrap();

        let configuration = load_configuration_from_path(&path).unwrap();

        assert!(configuration.indicator.show);
        assert_eq!(
            configuration.indicator.position,
            IndicatorPosition::BottomCenter
        );
    }
}
