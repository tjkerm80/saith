use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use super::resampling::AudioResampler;
use super::source::{AudioCaptureHandle, AudioSource, TARGET_SAMPLE_RATE};
use super::vad::VoiceActivityDetector;

/// Configuration for voice activity detection thresholds.
#[derive(Debug, Clone)]
pub struct VoiceActivityDetectionConfig {
    /// RMS energy threshold below which audio is considered silence.
    pub energy_threshold: f32,
    /// How long continuous silence must last to trigger end-of-speech.
    pub silence_duration_limit: Duration,
    /// Minimum duration of speech to consider a recording valid.
    pub minimum_speech_duration: Duration,
}

impl Default for VoiceActivityDetectionConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.002,
            silence_duration_limit: Duration::from_millis(1500),
            minimum_speech_duration: Duration::from_millis(100),
        }
    }
}

/// How a recording session is controlled.
pub enum RecordingMode {
    /// User holds a key to record; no auto-stop, no silence trimming.
    PushToTalk,
    /// User toggles recording on/off; VAD auto-stops and silence is trimmed.
    ToggleToTalk {
        on_auto_stop: Box<dyn FnOnce() + Send + 'static>,
    },
}

/// The result of finishing a recording.
#[derive(Debug)]
pub enum RecordingOutcome {
    /// Usable audio samples ready for transcription.
    Audio(Vec<f32>),
    /// Speech was detected but the recording was shorter than the minimum duration.
    TooShort,
    /// No speech was detected in the recording.
    NoSpeechDetected,
}

/// Facade that owns the full audio processing pipeline: capture, resampling,
/// VAD, silence trimming, and padding.
///
/// Callers interact through a two-step recording interface:
/// [`record`](AudioPipeline::record) → [`LiveRecording::finish`].
pub struct AudioPipeline {
    source: Box<dyn AudioSource>,
    device_description_cached: String,
    voice_activity_detection_config: VoiceActivityDetectionConfig,
}

impl AudioPipeline {
    /// Create a new pipeline backed by the given audio source.
    pub fn new(
        source: Box<dyn AudioSource>,
        voice_activity_detection_config: VoiceActivityDetectionConfig,
    ) -> Self {
        let device_description_cached = source.device_description();
        Self {
            source,
            device_description_cached,
            voice_activity_detection_config,
        }
    }

    /// The description of the audio source device.
    pub fn device_description(&self) -> &str {
        &self.device_description_cached
    }

    /// Start a recording session. The `mode` controls whether VAD auto-stop
    /// and silence trimming are applied.
    pub fn record(
        &mut self,
        mode: RecordingMode,
        on_waveform_peak: Box<dyn Fn(f32) + Send + 'static>,
    ) -> Result<LiveRecording> {
        let sample_rate = self.source.sample_rate();
        let channels = self.source.channels();
        let needs_resampling = sample_rate != TARGET_SAMPLE_RATE;

        let (chunk_sender, chunk_receiver) = crossbeam_channel::unbounded();
        let capture_handle = self.source.start_capture(chunk_sender)?;

        println!(
            "  Recording from '{}' ({}Hz, {} ch{})",
            self.device_description_cached,
            sample_rate,
            channels,
            if needs_resampling {
                ", will resample"
            } else {
                ""
            }
        );

        let finished = Arc::new(AtomicBool::new(false));
        let finished_for_thread = Arc::clone(&finished);

        // Peak amplitude tracking: ~50ms windows -> ~20 messages/sec
        let peak_window_size: usize = (sample_rate / 20) as usize;

        // Create a single VAD from config, shared between monitor thread and finish()
        let voice_activity_detection_config = self.voice_activity_detection_config.clone();
        let has_vad = matches!(mode, RecordingMode::ToggleToTalk { .. });

        // Extract the on_auto_stop closure if in toggle-to-talk mode
        let on_auto_stop = match mode {
            RecordingMode::ToggleToTalk { on_auto_stop } => Some(on_auto_stop),
            RecordingMode::PushToTalk => None,
        };

        // Determine the mode variant for LiveRecording (without the closure)
        let is_push_to_talk = on_auto_stop.is_none();

        let monitor_thread = std::thread::Builder::new()
            .name("audio-monitor".to_string())
            .spawn(move || {
                let mut voice_activity_detector = if has_vad {
                    Some(VoiceActivityDetector::with_config(
                        &voice_activity_detection_config,
                    ))
                } else {
                    None
                };

                let mut peak_sample_counter: usize = 0;
                let mut peak_accumulator: f32 = 0.0;

                loop {
                    match chunk_receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(chunk) => {
                            // Peak computation
                            for &sample in &chunk {
                                peak_accumulator = peak_accumulator.max(sample.abs());
                                peak_sample_counter += 1;
                                if peak_sample_counter >= peak_window_size {
                                    on_waveform_peak(peak_accumulator);
                                    peak_sample_counter = 0;
                                    peak_accumulator = 0.0;
                                }
                            }

                            // VAD end-of-speech detection
                            if let Some(ref mut detector) = voice_activity_detector
                                && detector.detect_end_of_speech(&chunk)
                            {
                                finished_for_thread.store(true, Ordering::Release);
                                if let Some(callback) = on_auto_stop {
                                    callback();
                                }
                                return;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if finished_for_thread.load(Ordering::Acquire) {
                                return;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            return;
                        }
                    }
                }
            })
            .context("Failed to spawn audio monitor thread")?;

        Ok(LiveRecording {
            capture_handle: Some(capture_handle),
            source_sample_rate: sample_rate,
            needs_resampling,
            finished,
            monitor_thread: Some(monitor_thread),
            is_push_to_talk,
            voice_activity_detection_config: self.voice_activity_detection_config.clone(),
        })
    }
}

/// A handle to an in-progress audio recording.
///
/// Call [`finish`](LiveRecording::finish) to stop capturing, process the
/// audio (resample, trim silence, pad), and get the result. Dropping without
/// calling `finish` will stop the capture and discard the audio.
pub struct LiveRecording {
    capture_handle: Option<Box<dyn AudioCaptureHandle>>,
    source_sample_rate: u32,
    needs_resampling: bool,
    finished: Arc<AtomicBool>,
    monitor_thread: Option<JoinHandle<()>>,
    is_push_to_talk: bool,
    voice_activity_detection_config: VoiceActivityDetectionConfig,
}

impl LiveRecording {
    /// Stop recording, process the audio, and return a structured outcome.
    ///
    /// Propagates monitor thread panics as errors.
    pub fn finish(mut self) -> Result<RecordingOutcome> {
        self.finished.store(true, Ordering::Release);

        // Stop the audio capture first so the chunk channel disconnects,
        // allowing the monitor thread to exit its receive loop.
        let capture_handle = self
            .capture_handle
            .take()
            .expect("Recording::finish called after capture already consumed");

        let raw_samples = capture_handle.stop_and_drain()?;

        if let Some(monitor_thread) = self.monitor_thread.take() {
            monitor_thread
                .join()
                .map_err(|_| anyhow::anyhow!("Audio monitor thread panicked"))?;
        }
        let raw_duration = raw_samples.len() as f32 / self.source_sample_rate as f32;
        println!("  Captured {raw_duration:.1}s of audio");

        let samples = if self.needs_resampling {
            let resampler = AudioResampler::new(self.source_sample_rate);
            let resampled = resampler
                .resample(&raw_samples)
                .context("Resampling failed")?;
            println!(
                "  Resampled from {}Hz to {TARGET_SAMPLE_RATE}Hz ({} samples)",
                self.source_sample_rate,
                resampled.len()
            );
            resampled
        } else {
            raw_samples
        };

        if samples.is_empty() {
            return Ok(RecordingOutcome::NoSpeechDetected);
        }

        let root_mean_square: f32 = (samples.iter().map(|sample| sample * sample).sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        let peak_amplitude = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0f32, f32::max);
        println!("  Audio levels: RMS={root_mean_square:.6}, peak={peak_amplitude:.6}");

        let voice_activity_detector =
            VoiceActivityDetector::with_config(&self.voice_activity_detection_config);

        // Push-to-talk: skip silence trimming (user controls boundaries)
        // Toggle-to-talk: trim leading/trailing silence
        let (speech_samples, owns_buffer) = if self.is_push_to_talk {
            (&samples[..], true)
        } else {
            let trimmed = voice_activity_detector.trim_silence(&samples);
            let trimmed_duration = trimmed.len() as f32 / TARGET_SAMPLE_RATE as f32;
            println!("  After silence trimming: {trimmed_duration:.1}s");
            (trimmed, false)
        };

        // Check if there's any speech at all
        if !voice_activity_detector.contains_speech(speech_samples) {
            println!("  No speech detected.");
            return Ok(RecordingOutcome::NoSpeechDetected);
        }

        // Check minimum speech duration
        let speech_duration_seconds = speech_samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        let minimum_speech_seconds = self
            .voice_activity_detection_config
            .minimum_speech_duration
            .as_secs_f32();
        if speech_duration_seconds < minimum_speech_seconds {
            println!(
                "  Recording too short ({speech_duration_seconds:.2}s), skipping transcription."
            );
            return Ok(RecordingOutcome::TooShort);
        }

        // Whisper RS rejects audio shorter than 1000ms. Its internal frame alignment
        // can report slightly fewer milliseconds than the raw sample count implies,
        // so we pad to 1100ms to leave a comfortable margin.
        let minimum_sample_count = (TARGET_SAMPLE_RATE as f32 * 1.1) as usize;
        let final_samples = if speech_samples.len() < minimum_sample_count {
            println!(
                "  Padded audio with silence ({} -> {} samples)",
                speech_samples.len(),
                minimum_sample_count
            );
            let mut padded = speech_samples.to_vec();
            padded.resize(minimum_sample_count, 0.0);
            padded
        } else if owns_buffer {
            // Push-to-talk, no padding needed — consume the owned vec directly
            samples
        } else {
            speech_samples.to_vec()
        };

        Ok(RecordingOutcome::Audio(final_samples))
    }
}

impl Drop for LiveRecording {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);

        // Drop the capture handle first to disconnect the chunk channel,
        // allowing the monitor thread to exit its receive loop.
        self.capture_handle.take();

        if let Some(monitor_thread) = self.monitor_thread.take() {
            let _ = monitor_thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_source::InMemoryAudioSource;
    use super::*;

    fn noop_peak(_peak: f32) {}
    fn boxed_noop_peak() -> Box<dyn Fn(f32) + Send + 'static> {
        Box::new(noop_peak)
    }

    fn default_pipeline(source: impl super::super::source::AudioSource + 'static) -> AudioPipeline {
        AudioPipeline::new(Box::new(source), VoiceActivityDetectionConfig::default())
    }

    #[test]
    fn push_to_talk_tone_produces_audio() {
        let source = InMemoryAudioSource::tone(440.0, 2.0, 48000);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();
        let outcome = recording.finish().unwrap();

        match outcome {
            RecordingOutcome::Audio(samples) => {
                let duration_seconds = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
                assert!(
                    duration_seconds > 1.5,
                    "Duration should be roughly 2s, got {duration_seconds}"
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn push_to_talk_silence_returns_no_speech() {
        let source = InMemoryAudioSource::silence(1.0, 48000);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();
        let outcome = recording.finish().unwrap();
        assert!(
            matches!(outcome, RecordingOutcome::NoSpeechDetected),
            "Pure silence should return NoSpeechDetected, got {outcome:?}"
        );
    }

    #[test]
    fn push_to_talk_short_recording_returns_too_short() {
        let source = InMemoryAudioSource::tone(440.0, 0.05, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();
        let outcome = recording.finish().unwrap();
        assert!(
            matches!(outcome, RecordingOutcome::TooShort),
            "Sub-100ms burst should return TooShort, got {outcome:?}"
        );
    }

    #[test]
    fn push_to_talk_short_speech_is_padded_to_minimum() {
        let source = InMemoryAudioSource::tone(440.0, 0.2, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();
        let outcome = recording.finish().unwrap();

        match outcome {
            RecordingOutcome::Audio(samples) => {
                let minimum_samples = (TARGET_SAMPLE_RATE as f32 * 1.1) as usize;
                assert!(
                    samples.len() >= minimum_samples,
                    "Output should have at least {minimum_samples} samples for Whisper, got {}",
                    samples.len()
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn push_to_talk_waveform_peak_callback_is_invoked() {
        let source = InMemoryAudioSource::tone(440.0, 1.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);

        let peak_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak_count_for_callback = Arc::clone(&peak_count);

        let recording = pipeline
            .record(
                RecordingMode::PushToTalk,
                Box::new(move |peak| {
                    assert!((0.0..=1.0).contains(&peak), "Peak out of range: {peak}");
                    peak_count_for_callback.fetch_add(1, Ordering::Relaxed);
                }),
            )
            .unwrap();

        let outcome = recording.finish().unwrap();
        assert!(matches!(outcome, RecordingOutcome::Audio(_)));

        let count = peak_count.load(Ordering::Relaxed);
        assert!(
            count > 0,
            "on_waveform_peak should have been called at least once, got {count}"
        );
    }

    #[test]
    fn toggle_to_talk_speech_then_silence_fires_auto_stop() {
        let source = InMemoryAudioSource::speech_then_silence(0.5, 2.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);

        let (auto_stop_sender, auto_stop_receiver) = crossbeam_channel::unbounded::<()>();
        let recording = pipeline
            .record(
                RecordingMode::ToggleToTalk {
                    on_auto_stop: Box::new(move || {
                        let _ = auto_stop_sender.send(());
                    }),
                },
                boxed_noop_peak(),
            )
            .unwrap();

        auto_stop_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("Should receive auto-stop callback");

        let outcome = recording.finish().unwrap();
        assert!(
            matches!(outcome, RecordingOutcome::Audio(_)),
            "Should still produce audio after auto-stop, got {outcome:?}"
        );
    }

    #[test]
    fn toggle_to_talk_trims_trailing_silence() {
        // A signal with 0.5s speech + 2.0s silence
        let source = InMemoryAudioSource::speech_then_silence(0.5, 2.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);

        let (auto_stop_sender, auto_stop_receiver) = crossbeam_channel::unbounded::<()>();
        let recording = pipeline
            .record(
                RecordingMode::ToggleToTalk {
                    on_auto_stop: Box::new(move || {
                        let _ = auto_stop_sender.send(());
                    }),
                },
                boxed_noop_peak(),
            )
            .unwrap();

        auto_stop_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("Should receive auto-stop callback");

        match recording.finish().unwrap() {
            RecordingOutcome::Audio(samples) => {
                // After trimming, should be much shorter than the full 2.5s
                let duration = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
                assert!(
                    duration < 1.5,
                    "Toggle-to-talk should trim silence, got {duration:.1}s"
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn push_to_talk_does_not_trim_silence() {
        // Push-to-talk with a constant signal should return the full duration
        let source = InMemoryAudioSource::tone(440.0, 2.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();

        match recording.finish().unwrap() {
            RecordingOutcome::Audio(samples) => {
                let duration = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
                assert!(
                    duration > 1.8,
                    "Push-to-talk should not trim, got {duration:.1}s"
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn custom_vad_config_thresholds_are_respected() {
        // Use a very high energy threshold so that a tone is treated as silence
        let config = VoiceActivityDetectionConfig {
            energy_threshold: 0.99,
            silence_duration_limit: Duration::from_millis(1500),
            minimum_speech_duration: Duration::from_millis(100),
        };
        let source = InMemoryAudioSource::tone(440.0, 1.0, TARGET_SAMPLE_RATE);
        let mut pipeline = AudioPipeline::new(Box::new(source), config);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();
        let outcome = recording.finish().unwrap();
        assert!(
            matches!(outcome, RecordingOutcome::NoSpeechDetected),
            "High threshold should detect no speech, got {outcome:?}"
        );
    }

    #[test]
    fn resampling_48khz_produces_correct_length() {
        let source = InMemoryAudioSource::tone(440.0, 2.0, 48000);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();

        match recording.finish().unwrap() {
            RecordingOutcome::Audio(samples) => {
                let expected = (TARGET_SAMPLE_RATE as f32 * 2.0) as usize;
                let tolerance = 3200;
                assert!(
                    samples.len().abs_diff(expected) < tolerance,
                    "Expected ~{expected} samples, got {}",
                    samples.len()
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn no_resampling_at_16khz() {
        let source = InMemoryAudioSource::tone(440.0, 2.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();

        match recording.finish().unwrap() {
            RecordingOutcome::Audio(samples) => {
                let expected = (TARGET_SAMPLE_RATE as f32 * 2.0) as usize;
                let tolerance = 3200;
                assert!(
                    samples.len().abs_diff(expected) < tolerance,
                    "Expected ~{expected} samples without resampling, got {}",
                    samples.len()
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn rms_and_peak_correct_for_known_signal() {
        let source = InMemoryAudioSource::constant(0.5, 1.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();

        match recording.finish().unwrap() {
            RecordingOutcome::Audio(samples) => {
                let root_mean_square: f32 =
                    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
                assert!(
                    (root_mean_square - 0.5).abs() < 0.05,
                    "RMS of constant 0.5 signal should be ~0.5, got {root_mean_square}"
                );
            }
            other => panic!("Expected Audio outcome, got {other:?}"),
        }
    }

    #[test]
    fn drop_without_finish_does_not_panic() {
        let source = InMemoryAudioSource::tone(440.0, 1.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);
        let recording = pipeline
            .record(RecordingMode::PushToTalk, boxed_noop_peak())
            .unwrap();
        drop(recording);
    }

    #[test]
    fn monitor_thread_panic_propagates_as_error() {
        let source = InMemoryAudioSource::tone(440.0, 1.0, TARGET_SAMPLE_RATE);
        let mut pipeline = default_pipeline(source);

        let recording = pipeline
            .record(
                RecordingMode::PushToTalk,
                Box::new(|_peak| {
                    panic!("intentional test panic in monitor thread");
                }),
            )
            .unwrap();

        let result = recording.finish();
        assert!(
            result.is_err(),
            "finish() should return Err when monitor thread panicked"
        );
        let error_message = result.unwrap_err().to_string();
        assert!(
            error_message.contains("monitor thread panicked"),
            "Error should mention monitor thread panic, got: {error_message}"
        );
    }
}
