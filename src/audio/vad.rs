use crate::audio::pipeline::VoiceActivityDetectionConfig;
use crate::audio::source::TARGET_SAMPLE_RATE;

/// Energy-based voice activity detector using RMS thresholds.
///
/// Used for two purposes:
/// - Trimming leading/trailing silence from toggle-to-talk recordings
/// - Detecting end-of-speech in toggle-to-talk mode
pub struct VoiceActivityDetector {
    /// RMS energy threshold below which audio is considered silence.
    energy_threshold: f32,
    /// How many seconds of continuous silence triggers end-of-speech.
    silence_duration_limit: f32,
    /// Number of consecutive silent samples accumulated.
    silent_sample_count: usize,
    /// Sample rate used for time calculations.
    sample_rate: u32,
}

impl VoiceActivityDetector {
    pub fn new() -> Self {
        Self::with_config(&VoiceActivityDetectionConfig::default())
    }

    /// Create a detector from an explicit configuration.
    pub fn with_config(config: &VoiceActivityDetectionConfig) -> Self {
        Self {
            energy_threshold: config.energy_threshold,
            silence_duration_limit: config.silence_duration_limit.as_secs_f32(),
            silent_sample_count: 0,
            sample_rate: TARGET_SAMPLE_RATE,
        }
    }

    /// Returns true if the given audio chunk contains speech based on RMS energy.
    pub fn is_speech(&self, chunk: &[f32]) -> bool {
        if chunk.is_empty() {
            return false;
        }
        let root_mean_square = compute_rms(chunk);
        root_mean_square >= self.energy_threshold
    }

    /// Returns true if any 20ms frame in the buffer contains speech.
    pub fn contains_speech(&self, audio: &[f32]) -> bool {
        if audio.is_empty() {
            return false;
        }
        let frame_size = (self.sample_rate as f32 * 0.02) as usize; // 20ms frames
        audio.chunks(frame_size).any(|frame| self.is_speech(frame))
    }

    /// Feed a chunk of audio and return true if sustained silence indicates
    /// end-of-speech. Resets the silence counter when speech is detected.
    pub fn detect_end_of_speech(&mut self, chunk: &[f32]) -> bool {
        if self.is_speech(chunk) {
            self.silent_sample_count = 0;
            return false;
        }

        self.silent_sample_count += chunk.len();
        let silent_duration = self.silent_sample_count as f32 / self.sample_rate as f32;
        silent_duration >= self.silence_duration_limit
    }

    /// Trim leading and trailing silence from an audio buffer.
    /// Returns a sub-slice containing only the speech region, or an empty
    /// slice if the entire buffer is silence.
    pub fn trim_silence<'a>(&self, audio: &'a [f32]) -> &'a [f32] {
        let frame_size = (self.sample_rate as f32 * 0.02) as usize; // 20ms frames

        // Find first frame with speech
        let first_speech_frame = audio
            .chunks(frame_size)
            .position(|frame| self.is_speech(frame));

        let first_speech_frame = match first_speech_frame {
            Some(index) => index,
            None => return &audio[0..0],
        };

        // Find last frame with speech
        let last_speech_frame = audio
            .chunks(frame_size)
            .rposition(|frame| self.is_speech(frame))
            .unwrap(); // Safe: we found at least one speech frame above

        let start_sample = first_speech_frame * frame_size;
        let end_sample = ((last_speech_frame + 1) * frame_size).min(audio.len());

        &audio[start_sample..end_sample]
    }
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the root mean square of an audio buffer.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_of_squares: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum_of_squares / samples.len() as f32).sqrt()
}
