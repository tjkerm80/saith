use anyhow::{Context, Result};
use crossbeam_channel::Sender;

/// Abstraction over audio input hardware. Implementations provide device
/// metadata and the ability to start a capture session that streams mono
/// f32 samples over a crossbeam channel.
///
/// The only production implementation is [`CpalAudioSource`]; tests use
/// [`InMemoryAudioSource`](super::test_source::InMemoryAudioSource).
pub trait AudioSource: Send + 'static {
    fn device_description(&self) -> String;
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
    fn start_capture(&self, chunk_sender: Sender<Vec<f32>>) -> Result<Box<dyn AudioCaptureHandle>>;
}

/// Handle to a running audio capture session. Stopping the capture drains
/// any remaining buffered samples and returns them.
pub trait AudioCaptureHandle: Send {
    fn stop_and_drain(self: Box<Self>) -> Result<Vec<f32>>;
}

/// Target sample rate for Whisper input.
pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;

/// [`AudioSource`] backed by the default cpal input device.
pub struct CpalAudioSource {
    device_description: String,
    sample_rate: u32,
    channels: u16,
}

impl CpalAudioSource {
    /// Probe the default audio input device and return a source ready for capture.
    pub fn from_default_device() -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait};

        let host = cpal::default_host();
        let input_device = host
            .default_input_device()
            .context("No default audio input device found")?;

        let device_description = input_device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "Unknown".to_string());

        let default_config = input_device
            .default_input_config()
            .context("Failed to get default input config")?;

        Ok(Self {
            device_description,
            sample_rate: default_config.sample_rate(),
            channels: default_config.channels(),
        })
    }
}

impl AudioSource for CpalAudioSource {
    fn device_description(&self) -> String {
        self.device_description.clone()
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn start_capture(&self, chunk_sender: Sender<Vec<f32>>) -> Result<Box<dyn AudioCaptureHandle>> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use std::sync::{Arc, Mutex};

        let host = cpal::default_host();
        let input_device = host
            .default_input_device()
            .context("No default audio input device found")?;

        let stream_config = cpal::StreamConfig {
            channels: self.channels,
            sample_rate: self.sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let channel_count = self.channels as usize;
        let accumulated_samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let samples_for_callback = accumulated_samples.clone();

        let stream = input_device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix to mono inline
                let mono_samples: Vec<f32> = if channel_count == 1 {
                    data.to_vec()
                } else {
                    data.chunks(channel_count)
                        .map(|frame| frame.iter().sum::<f32>() / channel_count as f32)
                        .collect()
                };

                let mut buffer = samples_for_callback.lock().unwrap();
                buffer.extend_from_slice(&mono_samples);
                drop(buffer);
                let _ = chunk_sender.send(mono_samples);
            },
            |error| {
                eprintln!("Audio capture stream error: {error}");
            },
            None,
        )?;

        stream
            .play()
            .context("Failed to start audio capture stream")?;

        Ok(Box::new(CpalCaptureHandle {
            stream,
            accumulated_samples,
        }))
    }
}

struct CpalCaptureHandle {
    stream: cpal::Stream,
    accumulated_samples: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
}

impl AudioCaptureHandle for CpalCaptureHandle {
    fn stop_and_drain(self: Box<Self>) -> Result<Vec<f32>> {
        use cpal::traits::StreamTrait;

        self.stream
            .pause()
            .context("Failed to pause audio capture stream")?;
        let mut buffer = self.accumulated_samples.lock().unwrap();
        let samples = std::mem::take(&mut *buffer);
        Ok(samples)
    }
}
