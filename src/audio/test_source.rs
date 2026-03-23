use anyhow::Result;
use crossbeam_channel::Sender;

use super::source::{AudioCaptureHandle, AudioSource};

/// In-memory [`AudioSource`] that yields pre-built samples for deterministic
/// testing without audio hardware.
pub struct InMemoryAudioSource {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl InMemoryAudioSource {
    /// Create a source that produces `duration` seconds of silence.
    pub fn silence(duration_seconds: f32, sample_rate: u32) -> Self {
        let sample_count = (sample_rate as f32 * duration_seconds) as usize;
        Self {
            samples: vec![0.0; sample_count],
            sample_rate,
        }
    }

    /// Create a source that produces a sine tone at the given frequency.
    pub fn tone(frequency: f32, duration_seconds: f32, sample_rate: u32) -> Self {
        let sample_count = (sample_rate as f32 * duration_seconds) as usize;
        let samples = (0..sample_count)
            .map(|index| {
                let time = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * time).sin() * 0.5
            })
            .collect();
        Self {
            samples,
            sample_rate,
        }
    }

    /// Create a source that produces speech (a tone) followed by silence.
    pub fn speech_then_silence(
        speech_duration_seconds: f32,
        silence_duration_seconds: f32,
        sample_rate: u32,
    ) -> Self {
        let speech_count = (sample_rate as f32 * speech_duration_seconds) as usize;
        let silence_count = (sample_rate as f32 * silence_duration_seconds) as usize;
        let mut samples = Vec::with_capacity(speech_count + silence_count);
        for index in 0..speech_count {
            let time = index as f32 / sample_rate as f32;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * time).sin() * 0.5);
        }
        samples.extend(std::iter::repeat_n(0.0f32, silence_count));
        Self {
            samples,
            sample_rate,
        }
    }

    /// Create a source from a constant amplitude signal.
    pub fn constant(amplitude: f32, duration_seconds: f32, sample_rate: u32) -> Self {
        let sample_count = (sample_rate as f32 * duration_seconds) as usize;
        Self {
            samples: vec![amplitude; sample_count],
            sample_rate,
        }
    }
}

impl AudioSource for InMemoryAudioSource {
    fn device_description(&self) -> String {
        "InMemoryAudioSource".to_string()
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        1
    }

    fn start_capture(&self, chunk_sender: Sender<Vec<f32>>) -> Result<Box<dyn AudioCaptureHandle>> {
        // Send all samples in chunks on a background thread, then hold them
        // in the handle so stop_and_drain can return whatever is left.
        let samples = self.samples.clone();
        let chunk_size = (self.sample_rate as usize) / 10; // 100ms chunks

        let (remaining_sender, remaining_receiver) = crossbeam_channel::bounded::<Vec<f32>>(1);

        std::thread::Builder::new()
            .name("in-memory-capture".to_string())
            .spawn(move || {
                let mut sent_up_to = 0;
                for chunk in samples.chunks(chunk_size) {
                    if chunk_sender.send(chunk.to_vec()).is_err() {
                        // Receiver dropped — return full buffer so stop_and_drain
                        // sees all audio captured up to this point.
                        let _ = remaining_sender.send(samples[..sent_up_to].to_vec());
                        return;
                    }
                    sent_up_to += chunk.len();
                }
                // All chunks sent. Return the full sample buffer for stop_and_drain.
                let _ = remaining_sender.send(samples);
            })
            .expect("Failed to spawn in-memory capture thread");

        Ok(Box::new(InMemoryCaptureHandle { remaining_receiver }))
    }
}

struct InMemoryCaptureHandle {
    remaining_receiver: crossbeam_channel::Receiver<Vec<f32>>,
}

impl AudioCaptureHandle for InMemoryCaptureHandle {
    fn stop_and_drain(self: Box<Self>) -> Result<Vec<f32>> {
        // The thread sends the full sample buffer once it finishes sending chunks.
        let samples = self
            .remaining_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap_or_default();
        Ok(samples)
    }
}
