use std::time::Duration;

use crate::pipeline_state::PipelineState;

/// Number of bars in the waveform visualization.
pub const WAVEFORM_BAR_COUNT: usize = 30;

/// Duration in milliseconds for the waveform decay transition from real data to sine wave.
pub const WAVEFORM_TRANSITION_DURATION_MILLISECONDS: f32 = 300.0;

const RECORDING_PULSE_PERIOD: f32 = 1800.0;
const PROCESSING_PULSE_PERIOD: f32 = 1000.0;
const TRANSCRIBING_PULSE_PERIOD: f32 = 2200.0;

const RECORDING_COLOR: LinearColor = LinearColor::new(1.0, 0.271, 0.227, 1.0);
const PROCESSING_COLOR: LinearColor = LinearColor::new(1.0, 0.839, 0.039, 1.0);
const TRANSCRIBING_COLOR: LinearColor = LinearColor::new(0.392, 0.824, 1.0, 1.0);

/// An RGBA color represented as linear floating-point values in `0.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearColor {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub alpha: f32,
}

impl LinearColor {
    pub const fn new(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }
}

/// Action the GUI shell should take after a state machine method call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowAction {
    None,
    Open,
    Close,
}

/// A fully computed frame ready for rendering — no iced dependency.
#[derive(Debug, Clone)]
pub struct IndicatorFrame {
    #[cfg(test)]
    pub pipeline_state: PipelineState,
    pub dot_color: LinearColor,
    pub glow_color: LinearColor,
    pub waveform_amplitudes: [f32; WAVEFORM_BAR_COUNT],
}

/// Pure animation state machine for the indicator. Tracks pipeline state, waveform
/// samples, transition snapshots, and sine animation phase. All timestamps are
/// injected as `Duration` (elapsed since an external epoch) — no `Instant::now()`
/// calls inside.
pub struct IndicatorState {
    current_state: PipelineState,
    animation_start: Duration,
    waveform_amplitudes: [f32; WAVEFORM_BAR_COUNT],
    transition_snapshot: [f32; WAVEFORM_BAR_COUNT],
    transition_start: Option<Duration>,
    sine_phase: f32,
}

impl IndicatorState {
    pub fn new() -> Self {
        Self {
            current_state: PipelineState::Idle,
            animation_start: Duration::ZERO,
            waveform_amplitudes: [0.0; WAVEFORM_BAR_COUNT],
            transition_snapshot: [0.0; WAVEFORM_BAR_COUNT],
            transition_start: None,
            sine_phase: 0.0,
        }
    }

    /// Records a waveform amplitude sample. Only takes effect during recording.
    pub fn push_waveform_sample(&mut self, amplitude: f32) -> WindowAction {
        if self.current_state == PipelineState::Recording {
            let scaled = perceptual_amplitude(amplitude);
            scroll_waveform(&mut self.waveform_amplitudes, scaled);
        }
        WindowAction::None
    }

    /// Transitions to a new pipeline state. Returns `Open` when entering an active
    /// state from idle, `Close` when returning to idle, or `None` otherwise.
    pub fn transition_to(&mut self, state: PipelineState, timestamp: Duration) -> WindowAction {
        if state == self.current_state {
            return WindowAction::None;
        }

        let previous_state = self.current_state;
        self.current_state = state;
        self.animation_start = timestamp;

        match state {
            PipelineState::Processing | PipelineState::Transcribing
                if previous_state == PipelineState::Recording =>
            {
                self.transition_snapshot = self.waveform_amplitudes;
                self.transition_start = Some(timestamp);
            }
            PipelineState::Idle => {
                self.waveform_amplitudes = [0.0; WAVEFORM_BAR_COUNT];
                self.transition_snapshot = [0.0; WAVEFORM_BAR_COUNT];
                self.transition_start = None;
                self.sine_phase = 0.0;
            }
            _ => {}
        }

        if state != PipelineState::Idle && previous_state == PipelineState::Idle {
            WindowAction::Open
        } else if state == PipelineState::Idle {
            WindowAction::Close
        } else {
            WindowAction::None
        }
    }

    /// Advances the sine wave animation by one step. Should be called on each
    /// animation tick while in processing or transcribing state.
    pub fn tick(&mut self, timestamp: Duration) -> WindowAction {
        if self.current_state == PipelineState::Processing
            || self.current_state == PipelineState::Transcribing
        {
            self.sine_phase += 0.15;

            let progress = self
                .transition_start
                .map(|start| {
                    ((timestamp - start).as_secs_f32() * 1000.0
                        / WAVEFORM_TRANSITION_DURATION_MILLISECONDS)
                        .min(1.0)
                })
                .unwrap_or(1.0);

            compute_transition_frame(
                &self.transition_snapshot,
                self.sine_phase,
                progress,
                &mut self.waveform_amplitudes,
            );
        }
        WindowAction::None
    }

    /// Computes a renderable frame for the current state. Returns `None` when idle
    /// (the indicator window should be hidden).
    pub fn frame(&self, timestamp: Duration) -> Option<IndicatorFrame> {
        let (base_color, pulse_period) = match self.current_state {
            PipelineState::Recording => (RECORDING_COLOR, RECORDING_PULSE_PERIOD),
            PipelineState::Processing => (PROCESSING_COLOR, PROCESSING_PULSE_PERIOD),
            PipelineState::Transcribing => (TRANSCRIBING_COLOR, TRANSCRIBING_PULSE_PERIOD),
            PipelineState::Idle => return None,
        };

        let elapsed_milliseconds = (timestamp - self.animation_start).as_millis() as f32;
        let phase = (elapsed_milliseconds / pulse_period * std::f32::consts::TAU).sin();
        let pulsed_opacity = 0.6 + 0.4 * (phase * 0.5 + 0.5);
        let glow_opacity = 0.3 + 0.4 * (phase * 0.5 + 0.5);

        Some(IndicatorFrame {
            #[cfg(test)]
            pipeline_state: self.current_state,
            dot_color: LinearColor::new(
                base_color.red,
                base_color.green,
                base_color.blue,
                pulsed_opacity,
            ),
            glow_color: LinearColor::new(
                base_color.red,
                base_color.green,
                base_color.blue,
                glow_opacity,
            ),
            waveform_amplitudes: self.waveform_amplitudes,
        })
    }
}

fn perceptual_amplitude(raw: f32) -> f32 {
    raw.clamp(0.0, 1.0).powf(0.3)
}

fn scroll_waveform(amplitudes: &mut [f32; WAVEFORM_BAR_COUNT], new_amplitude: f32) {
    amplitudes.rotate_left(1);
    amplitudes[WAVEFORM_BAR_COUNT - 1] = new_amplitude;
}

fn compute_transition_frame(
    snapshot: &[f32; WAVEFORM_BAR_COUNT],
    sine_phase: f32,
    progress: f32,
    output: &mut [f32; WAVEFORM_BAR_COUNT],
) {
    for index in 0..WAVEFORM_BAR_COUNT {
        let sine_value = (std::f32::consts::TAU * index as f32 / WAVEFORM_BAR_COUNT as f32
            + sine_phase)
            .sin()
            .abs()
            * 0.3;
        let snapshot_value = snapshot[index];
        output[index] = snapshot_value + (sine_value - snapshot_value) * progress;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording_state() -> IndicatorState {
        let mut state = IndicatorState::new();
        state.transition_to(PipelineState::Recording, Duration::from_secs(1));
        state
    }

    #[test]
    fn idle_state_frame_returns_none() {
        let state = IndicatorState::new();
        assert!(state.frame(Duration::from_secs(0)).is_none());
    }

    #[test]
    fn transition_to_recording_returns_open() {
        let mut state = IndicatorState::new();
        let action = state.transition_to(PipelineState::Recording, Duration::from_secs(1));
        assert_eq!(action, WindowAction::Open);
    }

    #[test]
    fn transition_to_idle_returns_close() {
        let mut state = recording_state();
        let action = state.transition_to(PipelineState::Idle, Duration::from_secs(2));
        assert_eq!(action, WindowAction::Close);
    }

    #[test]
    fn duplicate_transition_returns_none() {
        let mut state = recording_state();
        let action = state.transition_to(PipelineState::Recording, Duration::from_secs(2));
        assert_eq!(action, WindowAction::None);
    }

    #[test]
    fn recording_frame_has_recording_color() {
        let state = recording_state();
        let frame = state.frame(Duration::from_secs(1)).unwrap();
        assert_eq!(frame.pipeline_state, PipelineState::Recording);
        assert!((frame.dot_color.red - 1.0).abs() < f32::EPSILON);
        assert!((frame.dot_color.green - 0.271).abs() < f32::EPSILON);
        assert!((frame.dot_color.blue - 0.227).abs() < f32::EPSILON);
    }

    #[test]
    fn processing_frame_has_processing_color() {
        let mut state = recording_state();
        state.transition_to(PipelineState::Processing, Duration::from_secs(2));
        let frame = state.frame(Duration::from_secs(2)).unwrap();
        assert_eq!(frame.pipeline_state, PipelineState::Processing);
        assert!((frame.dot_color.red - 1.0).abs() < f32::EPSILON);
        assert!((frame.dot_color.green - 0.839).abs() < f32::EPSILON);
        assert!((frame.dot_color.blue - 0.039).abs() < f32::EPSILON);
    }

    #[test]
    fn transcribing_frame_has_transcribing_color() {
        let mut state = recording_state();
        state.transition_to(PipelineState::Transcribing, Duration::from_secs(2));
        let frame = state.frame(Duration::from_secs(2)).unwrap();
        assert_eq!(frame.pipeline_state, PipelineState::Transcribing);
        assert!((frame.dot_color.red - 0.392).abs() < f32::EPSILON);
        assert!((frame.dot_color.green - 0.824).abs() < f32::EPSILON);
        assert!((frame.dot_color.blue - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn waveform_sample_scrolls_into_frame() {
        let mut state = recording_state();
        state.push_waveform_sample(0.5);
        let frame = state.frame(Duration::from_secs(1)).unwrap();
        let expected = perceptual_amplitude(0.5);
        assert!(
            (frame.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1] - expected).abs() < f32::EPSILON,
            "last bar should be perceptual_amplitude(0.5) = {expected}",
        );
    }

    #[test]
    fn waveform_samples_ignored_when_not_recording() {
        let mut state = IndicatorState::new();
        // Idle: sample should be ignored
        state.push_waveform_sample(0.8);
        assert!(state.frame(Duration::ZERO).is_none());

        // Processing: sample should also be ignored
        state.transition_to(PipelineState::Recording, Duration::from_secs(1));
        state.transition_to(PipelineState::Processing, Duration::from_secs(2));
        let frame_before = state.frame(Duration::from_secs(2)).unwrap();
        let last_before = frame_before.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1];

        state.push_waveform_sample(0.9);
        let frame_after = state.frame(Duration::from_secs(2)).unwrap();
        assert!(
            (frame_after.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1] - last_before).abs()
                < f32::EPSILON,
            "waveform should not change during processing",
        );
    }

    #[test]
    fn transition_from_recording_to_processing_snapshots_waveform() {
        let mut state = recording_state();
        state.push_waveform_sample(0.6);
        let expected_last = perceptual_amplitude(0.6);

        state.transition_to(PipelineState::Processing, Duration::from_secs(2));
        // At zero progress (same timestamp as transition), tick preserves snapshot
        state.tick(Duration::from_secs(2));
        let frame = state.frame(Duration::from_secs(2)).unwrap();
        assert!(
            (frame.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1] - expected_last).abs() < 1e-6,
            "snapshot should be preserved at zero transition progress",
        );
    }

    #[test]
    fn tick_advances_sine_animation() {
        let mut state = recording_state();
        state.push_waveform_sample(0.5);
        state.transition_to(PipelineState::Processing, Duration::from_secs(2));

        // After enough ticks past the transition duration, the waveform should be
        // fully driven by the sine wave (different from the original snapshot).
        let far_future = Duration::from_secs(10);
        state.tick(far_future);
        let frame_a = state.frame(far_future).unwrap();

        state.tick(far_future);
        let frame_b = state.frame(far_future).unwrap();

        // Sine phase advances each tick, so at least some bars should differ
        let any_differ = frame_a
            .waveform_amplitudes
            .iter()
            .zip(frame_b.waveform_amplitudes.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_differ, "tick should advance sine animation");
    }

    #[test]
    fn transition_to_idle_resets_waveform() {
        let mut state = recording_state();
        state.push_waveform_sample(0.7);
        state.transition_to(PipelineState::Idle, Duration::from_secs(2));

        // After going idle, transition back to recording to observe the waveform
        state.transition_to(PipelineState::Recording, Duration::from_secs(3));
        let frame = state.frame(Duration::from_secs(3)).unwrap();
        for (index, &amplitude) in frame.waveform_amplitudes.iter().enumerate() {
            assert!(
                amplitude.abs() < f32::EPSILON,
                "bar {index} should be zero after idle reset, got {amplitude}",
            );
        }
    }

    #[test]
    fn pulse_opacity_varies_with_time() {
        let state = recording_state();
        let frame_early = state.frame(Duration::from_secs(1)).unwrap();
        // Quarter of a pulse period later (450ms for recording with 1800ms period)
        let frame_later = state.frame(Duration::from_millis(1_000 + 450)).unwrap();
        assert!(
            (frame_early.dot_color.alpha - frame_later.dot_color.alpha).abs() > 0.01,
            "opacity should change over time due to pulse animation",
        );
    }

    #[test]
    fn frame_at_zero_elapsed_has_known_opacity() {
        let state = recording_state();
        // Frame at exactly the animation_start timestamp => zero elapsed
        let frame = state.frame(Duration::from_secs(1)).unwrap();
        // phase = sin(0) = 0, pulsed_opacity = 0.6 + 0.4 * (0*0.5+0.5) = 0.8
        assert!(
            (frame.dot_color.alpha - 0.8).abs() < f32::EPSILON,
            "dot opacity at zero elapsed should be 0.8, got {}",
            frame.dot_color.alpha,
        );
        // glow_opacity = 0.3 + 0.4 * 0.5 = 0.5
        assert!(
            (frame.glow_color.alpha - 0.5).abs() < f32::EPSILON,
            "glow opacity at zero elapsed should be 0.5, got {}",
            frame.glow_color.alpha,
        );
    }

    #[test]
    fn perceptual_scaling_boosts_quiet_signals() {
        let mut state = recording_state();
        state.push_waveform_sample(0.05);
        let frame = state.frame(Duration::from_secs(1)).unwrap();
        let scaled = frame.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1];
        assert!(scaled > 0.30, "expected > 0.30, got {scaled}");
        assert!(scaled < 0.45, "expected < 0.45, got {scaled}");
    }

    #[test]
    fn perceptual_scaling_clamps_out_of_range() {
        let mut state = recording_state();
        state.push_waveform_sample(-0.5);
        let frame = state.frame(Duration::from_secs(1)).unwrap();
        assert!(
            (frame.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1]).abs() < f32::EPSILON,
            "negative amplitude should clamp to 0.0",
        );

        state.push_waveform_sample(1.5);
        let frame = state.frame(Duration::from_secs(1)).unwrap();
        assert!(
            (frame.waveform_amplitudes[WAVEFORM_BAR_COUNT - 1] - 1.0).abs() < f32::EPSILON,
            "amplitude > 1.0 should clamp to 1.0",
        );
    }
}
