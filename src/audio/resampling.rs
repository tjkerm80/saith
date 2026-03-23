use anyhow::{Context, Result};
use rubato::audioadapter::Adapter;
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};

use crate::audio::source::TARGET_SAMPLE_RATE;

/// Resamples mono audio from a source sample rate to the target 16kHz rate
/// expected by Whisper.
pub(crate) struct AudioResampler {
    source_sample_rate: u32,
}

impl AudioResampler {
    /// Create a new resampler for the given source sample rate.
    /// Panics if source_sample_rate equals TARGET_SAMPLE_RATE (resampling is unnecessary).
    pub fn new(source_sample_rate: u32) -> Self {
        assert_ne!(
            source_sample_rate, TARGET_SAMPLE_RATE,
            "AudioResampler should not be constructed when source rate already matches target"
        );
        Self { source_sample_rate }
    }

    /// Resample a complete mono audio buffer to 16kHz.
    pub fn resample(&self, input_samples: &[f32]) -> Result<Vec<f32>> {
        let resample_ratio = TARGET_SAMPLE_RATE as f64 / self.source_sample_rate as f64;
        let chunk_size = 1024;

        let mut resampler = Async::<f32>::new_poly(
            resample_ratio,
            1.1,
            PolynomialDegree::Septic,
            chunk_size,
            1, // mono
            FixedAsync::Input,
        )
        .context("Failed to create rubato resampler")?;

        let mut output_samples = Vec::new();
        let input_frames_per_chunk = resampler.input_frames_next();
        let mut position = 0;

        while position < input_samples.len() {
            let remaining = input_samples.len() - position;
            let frames_this_chunk = remaining.min(input_frames_per_chunk);

            // Pad the last chunk with silence if it's shorter than expected
            let chunk: Vec<f32> = if frames_this_chunk < input_frames_per_chunk {
                let mut padded = vec![0.0f32; input_frames_per_chunk];
                padded[..frames_this_chunk]
                    .copy_from_slice(&input_samples[position..position + frames_this_chunk]);
                padded
            } else {
                input_samples[position..position + frames_this_chunk].to_vec()
            };

            let input_adapter = MonoSliceAdapter::new(&chunk);

            let result = resampler
                .process(&input_adapter, 0, None)
                .context("Resampling chunk failed")?;

            // InterleavedOwned: mono so just read all frames from channel 0
            for frame_index in 0..result.frames() {
                output_samples.push(result.read_sample(0, frame_index).unwrap_or(0.0));
            }

            position += frames_this_chunk;
        }

        Ok(output_samples)
    }
}

/// Minimal adapter that presents a `&[f32]` slice as a single-channel audio buffer.
struct MonoSliceAdapter<'a> {
    data: &'a [f32],
}

impl<'a> MonoSliceAdapter<'a> {
    fn new(data: &'a [f32]) -> Self {
        Self { data }
    }
}

impl<'a> Adapter<'a, f32> for MonoSliceAdapter<'a> {
    unsafe fn read_sample_unchecked(&self, _channel: usize, frame: usize) -> f32 {
        unsafe { *self.data.get_unchecked(frame) }
    }

    fn channels(&self) -> usize {
        1
    }

    fn frames(&self) -> usize {
        self.data.len()
    }
}
