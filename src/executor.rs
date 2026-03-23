use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::Sender;

use crate::audio::pipeline::{AudioPipeline, LiveRecording, RecordingMode, RecordingOutcome};
use crate::dictation::{DictationStateMachine, Effect, EffectOutcome, Transition};
use crate::hotkey::HotkeyEvent;
use crate::indicator_message::IndicatorMessage;
use crate::output::TranscriptionOutput;
use crate::pipeline_state::PipelineState;
use crate::transcription::TranscriptionEngine;

/// Abstraction over audio recording initiation.
pub trait AudioRecorder {
    type Recording: FinishableRecording;

    fn record(
        &mut self,
        mode: RecordingMode,
        on_waveform_peak: Box<dyn Fn(f32) + Send + 'static>,
    ) -> Result<Self::Recording>;
}

/// Abstraction over completing a recording and extracting audio samples.
pub trait FinishableRecording {
    fn finish(self) -> Result<RecordingOutcome>;
}

/// Abstraction over speech-to-text transcription.
pub trait Transcriber {
    fn transcribe(&mut self, audio_samples: &[f32], initial_prompt: Option<&str>)
    -> Result<String>;
}

/// Abstraction over outputting transcribed text (e.g., virtual keyboard typing).
pub trait TextOutput {
    fn output(&mut self, text: &str) -> Result<()>;
}

/// Abstraction over notifying an indicator of pipeline state changes.
pub trait IndicatorNotifier {
    fn notify_state_changed(&self, state: PipelineState);
    fn notify_waveform_peak(&self, peak: f32);
}

// --- Production trait implementations ---

impl AudioRecorder for AudioPipeline {
    type Recording = LiveRecording;

    fn record(
        &mut self,
        mode: RecordingMode,
        on_waveform_peak: Box<dyn Fn(f32) + Send + 'static>,
    ) -> Result<Self::Recording> {
        AudioPipeline::record(self, mode, on_waveform_peak)
    }
}

impl FinishableRecording for LiveRecording {
    fn finish(self) -> Result<RecordingOutcome> {
        LiveRecording::finish(self)
    }
}

impl Transcriber for TranscriptionEngine {
    fn transcribe(
        &mut self,
        audio_samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<String> {
        TranscriptionEngine::transcribe(self, audio_samples, initial_prompt)
    }
}

impl TextOutput for TranscriptionOutput {
    fn output(&mut self, text: &str) -> Result<()> {
        TranscriptionOutput::output(self, text)
    }
}

impl IndicatorNotifier for Sender<IndicatorMessage> {
    fn notify_state_changed(&self, state: PipelineState) {
        let _ = self.send(IndicatorMessage::StateChanged(state));
    }

    fn notify_waveform_peak(&self, peak: f32) {
        let _ = self.send(IndicatorMessage::WaveformSample(peak));
    }
}

// --- Generic executor ---

/// Owns all I/O resources and maps `Effect` values to real operations.
///
/// Generic over the four subsystem traits so that tests can substitute mocks.
pub struct DictationExecutor<Recorder, TranscriberImpl, Output, Indicator>
where
    Recorder: AudioRecorder,
{
    audio_recorder: Recorder,
    transcriber: TranscriberImpl,
    text_output: Output,
    indicator_notifier: Indicator,
    hotkey_sender: Sender<HotkeyEvent>,
    active_recording: Option<Recorder::Recording>,
    output_delay: Duration,
}

impl<Recorder, TranscriberImpl, Output, Indicator>
    DictationExecutor<Recorder, TranscriberImpl, Output, Indicator>
where
    Recorder: AudioRecorder,
    TranscriberImpl: Transcriber,
    Output: TextOutput,
    Indicator: IndicatorNotifier + Clone + Send + 'static,
{
    pub fn new(
        audio_recorder: Recorder,
        transcriber: TranscriberImpl,
        text_output: Output,
        indicator_notifier: Indicator,
        hotkey_sender: Sender<HotkeyEvent>,
    ) -> Self {
        Self {
            audio_recorder,
            transcriber,
            text_output,
            indicator_notifier,
            hotkey_sender,
            active_recording: None,
            output_delay: Duration::from_millis(150),
        }
    }

    /// Override the delay before typing output. Production uses 150ms to let
    /// the compositor return focus; tests use `Duration::ZERO`.
    #[cfg(test)]
    pub fn set_output_delay(&mut self, delay: Duration) {
        self.output_delay = delay;
    }

    /// Drive the outcome-feedback loop: execute each effect, feed outcomes back
    /// into the state machine, and recursively execute any follow-up transitions.
    ///
    /// Returns `true` if a `Shutdown` effect was encountered.
    pub fn execute_transition(
        &mut self,
        state_machine: &mut DictationStateMachine,
        transition: Transition,
    ) -> bool {
        for effect in transition.effects {
            if matches!(effect, Effect::Shutdown) {
                return true;
            }
            if let Some(outcome) = self.execute_effect(effect) {
                let followup = state_machine.handle_outcome(outcome);
                if self.execute_transition(state_machine, followup) {
                    return true;
                }
            }
        }
        false
    }

    fn execute_effect(&mut self, effect: Effect) -> Option<EffectOutcome> {
        match effect {
            Effect::StartRecording => self.start_recording(false),

            Effect::StartRecordingWithVoiceActivityDetection => self.start_recording(true),

            Effect::FinishRecording => {
                if let Some(recording) = self.active_recording.take() {
                    match recording.finish() {
                        Ok(outcome) => Some(EffectOutcome::RecordingFinished { outcome }),
                        Err(error) => Some(EffectOutcome::RecordingFinishFailed {
                            error: error.to_string(),
                        }),
                    }
                } else {
                    Some(EffectOutcome::RecordingFinishFailed {
                        error: "No active recording to finish".to_string(),
                    })
                }
            }

            Effect::Transcribe {
                samples,
                initial_prompt,
            } => {
                let transcription_start = Instant::now();
                match self
                    .transcriber
                    .transcribe(&samples, initial_prompt.as_deref())
                {
                    Ok(transcription) => {
                        let transcription_duration = transcription_start.elapsed();
                        if !transcription.is_empty() {
                            println!(
                                "  Transcription ({:.0}ms): {}",
                                transcription_duration.as_millis(),
                                transcription
                            );
                        }
                        Some(EffectOutcome::TranscriptionCompleted(transcription))
                    }
                    Err(error) => Some(EffectOutcome::TranscriptionFailed(error.to_string())),
                }
            }

            Effect::OutputTranscription(text) => {
                // Brief delay to let the compositor destroy the indicator window and return
                // focus to the previously-focused application before we start typing.
                if !self.output_delay.is_zero() {
                    std::thread::sleep(self.output_delay);
                }

                if let Err(error) = self.text_output.output(&text) {
                    eprintln!("  Warning: typing failed ({error})");
                }
                Some(EffectOutcome::OutputCompleted)
            }

            Effect::UpdateIndicator(state) => {
                self.indicator_notifier.notify_state_changed(state);
                None
            }

            Effect::DiscardRecording => {
                self.active_recording.take();
                None
            }

            Effect::Shutdown => None,
        }
    }

    fn start_recording(&mut self, with_voice_activity_detection: bool) -> Option<EffectOutcome> {
        let indicator = self.indicator_notifier.clone();
        let on_waveform_peak: Box<dyn Fn(f32) + Send + 'static> = Box::new(move |peak: f32| {
            indicator.notify_waveform_peak(peak);
        });

        let mode = if with_voice_activity_detection {
            let hotkey_sender = self.hotkey_sender.clone();
            RecordingMode::ToggleToTalk {
                on_auto_stop: Box::new(move || {
                    let _ = hotkey_sender.send(HotkeyEvent::AutoStopped);
                }),
            }
        } else {
            RecordingMode::PushToTalk
        };

        match self.audio_recorder.record(mode, on_waveform_peak) {
            Ok(recording) => {
                self.active_recording = Some(recording);
                Some(EffectOutcome::RecordingStarted)
            }
            Err(error) => Some(EffectOutcome::RecordingFailed(error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::InteractionMode;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    // --- Mock types ---

    struct MockRecording {
        result: Result<RecordingOutcome>,
    }

    impl FinishableRecording for MockRecording {
        fn finish(self) -> Result<RecordingOutcome> {
            self.result
        }
    }

    struct MockAudioRecorder {
        results: VecDeque<Result<MockRecording>>,
    }

    impl MockAudioRecorder {
        fn new() -> Self {
            Self {
                results: VecDeque::new(),
            }
        }

        fn queue_audio(&mut self, samples: Vec<f32>) {
            self.results.push_back(Ok(MockRecording {
                result: Ok(RecordingOutcome::Audio(samples)),
            }));
        }

        fn queue_no_speech(&mut self) {
            self.results.push_back(Ok(MockRecording {
                result: Ok(RecordingOutcome::NoSpeechDetected),
            }));
        }

        fn queue_too_short(&mut self) {
            self.results.push_back(Ok(MockRecording {
                result: Ok(RecordingOutcome::TooShort),
            }));
        }

        fn queue_failure(&mut self, message: &str) {
            self.results.push_back(Err(anyhow::anyhow!("{}", message)));
        }
    }

    impl AudioRecorder for MockAudioRecorder {
        type Recording = MockRecording;

        fn record(
            &mut self,
            _mode: RecordingMode,
            _on_waveform_peak: Box<dyn Fn(f32) + Send + 'static>,
        ) -> Result<Self::Recording> {
            self.results
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("No more queued recordings")))
        }
    }

    struct MockTranscriber {
        results: VecDeque<Result<String>>,
    }

    impl MockTranscriber {
        fn new() -> Self {
            Self {
                results: VecDeque::new(),
            }
        }

        fn queue_success(&mut self, text: &str) {
            self.results.push_back(Ok(text.to_string()));
        }

        fn queue_failure(&mut self, message: &str) {
            self.results.push_back(Err(anyhow::anyhow!("{}", message)));
        }
    }

    impl Transcriber for MockTranscriber {
        fn transcribe(
            &mut self,
            _audio_samples: &[f32],
            _initial_prompt: Option<&str>,
        ) -> Result<String> {
            self.results
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("No more queued transcriptions")))
        }
    }

    struct MockTextOutput {
        typed_texts: Arc<Mutex<Vec<String>>>,
    }

    impl MockTextOutput {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let typed_texts = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    typed_texts: Arc::clone(&typed_texts),
                },
                typed_texts,
            )
        }
    }

    impl TextOutput for MockTextOutput {
        fn output(&mut self, text: &str) -> Result<()> {
            self.typed_texts.lock().unwrap().push(text.to_string());
            Ok(())
        }
    }

    #[derive(Clone)]
    struct MockIndicatorNotifier {
        states: Arc<Mutex<Vec<PipelineState>>>,
        waveform_peaks: Arc<Mutex<Vec<f32>>>,
    }

    type IndicatorSpies = (
        MockIndicatorNotifier,
        Arc<Mutex<Vec<PipelineState>>>,
        Arc<Mutex<Vec<f32>>>,
    );

    impl MockIndicatorNotifier {
        fn new() -> IndicatorSpies {
            let states = Arc::new(Mutex::new(Vec::new()));
            let waveform_peaks = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    states: Arc::clone(&states),
                    waveform_peaks: Arc::clone(&waveform_peaks),
                },
                states,
                waveform_peaks,
            )
        }
    }

    impl IndicatorNotifier for MockIndicatorNotifier {
        fn notify_state_changed(&self, state: PipelineState) {
            self.states.lock().unwrap().push(state);
        }

        fn notify_waveform_peak(&self, peak: f32) {
            self.waveform_peaks.lock().unwrap().push(peak);
        }
    }

    // --- Test helpers ---

    type TestExecutor = DictationExecutor<
        MockAudioRecorder,
        MockTranscriber,
        MockTextOutput,
        MockIndicatorNotifier,
    >;

    struct TestHarness {
        executor: TestExecutor,
        state_machine: DictationStateMachine,
        typed_texts: Arc<Mutex<Vec<String>>>,
        indicator_states: Arc<Mutex<Vec<PipelineState>>>,
        #[allow(dead_code)]
        hotkey_receiver: crossbeam_channel::Receiver<HotkeyEvent>,
    }

    impl TestHarness {
        fn new(interaction_mode: InteractionMode, initial_prompt: Option<String>) -> Self {
            Self::with_mocks(
                interaction_mode,
                initial_prompt,
                MockAudioRecorder::new(),
                MockTranscriber::new(),
            )
        }

        fn with_mocks(
            interaction_mode: InteractionMode,
            initial_prompt: Option<String>,
            recorder: MockAudioRecorder,
            transcriber: MockTranscriber,
        ) -> Self {
            let (text_output, typed_texts) = MockTextOutput::new();
            let (indicator, indicator_states, _waveform_peaks) = MockIndicatorNotifier::new();
            let (hotkey_sender, hotkey_receiver) = crossbeam_channel::unbounded();

            let mut executor = DictationExecutor::new(
                recorder,
                transcriber,
                text_output,
                indicator,
                hotkey_sender,
            );
            executor.set_output_delay(Duration::ZERO);

            let state_machine = DictationStateMachine::new(interaction_mode, initial_prompt);

            Self {
                executor,
                state_machine,
                typed_texts,
                indicator_states,
                hotkey_receiver,
            }
        }

        fn handle_event(&mut self, event: HotkeyEvent) -> bool {
            let transition = self.state_machine.handle_event(event);
            self.executor
                .execute_transition(&mut self.state_machine, transition)
        }
    }

    // --- Tests ---

    #[test]
    fn push_to_talk_full_cycle() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.1, 0.2, 0.3]);

        let mut transcriber = MockTranscriber::new();
        transcriber.queue_success("hello world");

        let mut harness =
            TestHarness::with_mocks(InteractionMode::PushToTalk, None, recorder, transcriber);

        // Press -> starts recording
        let shutdown = harness.handle_event(HotkeyEvent::Pressed);
        assert!(!shutdown);

        // Release -> finish recording -> transcribe -> output
        let shutdown = harness.handle_event(HotkeyEvent::Released);
        assert!(!shutdown);

        // Verify text was output
        let texts = harness.typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "hello world");

        // Verify indicator state flow: Recording -> Processing -> Transcribing -> Idle
        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Transcribing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn toggle_to_talk_with_voice_activity_detection_auto_stop() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.5, 0.6]);

        let mut transcriber = MockTranscriber::new();
        transcriber.queue_success("auto stopped text");

        let mut harness =
            TestHarness::with_mocks(InteractionMode::ToggleToTalk, None, recorder, transcriber);

        // Press -> starts recording with VAD
        let shutdown = harness.handle_event(HotkeyEvent::Pressed);
        assert!(!shutdown);

        // AutoStopped -> finish recording -> transcribe -> output
        let shutdown = harness.handle_event(HotkeyEvent::AutoStopped);
        assert!(!shutdown);

        let texts = harness.typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "auto stopped text");

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Transcribing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn recording_failure_recovery() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_failure("device error");

        let mut harness = TestHarness::with_mocks(
            InteractionMode::PushToTalk,
            None,
            recorder,
            MockTranscriber::new(),
        );

        let shutdown = harness.handle_event(HotkeyEvent::Pressed);
        assert!(!shutdown);

        // No text should be output
        assert!(harness.typed_texts.lock().unwrap().is_empty());
        // No indicator states (RecordingFailed produces no UpdateIndicator effects)
        assert!(harness.indicator_states.lock().unwrap().is_empty());
    }

    #[test]
    fn transcription_failure_recovery() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.1, 0.2]);

        let mut transcriber = MockTranscriber::new();
        transcriber.queue_failure("model error");

        let mut harness =
            TestHarness::with_mocks(InteractionMode::PushToTalk, None, recorder, transcriber);

        harness.handle_event(HotkeyEvent::Pressed);
        harness.handle_event(HotkeyEvent::Released);

        // No text output
        assert!(harness.typed_texts.lock().unwrap().is_empty());

        // Indicator should end at Idle
        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Transcribing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn empty_transcription_skips_output() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.1]);

        let mut transcriber = MockTranscriber::new();
        transcriber.queue_success("");

        let mut harness =
            TestHarness::with_mocks(InteractionMode::PushToTalk, None, recorder, transcriber);

        harness.handle_event(HotkeyEvent::Pressed);
        harness.handle_event(HotkeyEvent::Released);

        assert!(harness.typed_texts.lock().unwrap().is_empty());

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Transcribing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn no_speech_detected_skips_transcription() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_no_speech();

        let mut harness = TestHarness::with_mocks(
            InteractionMode::PushToTalk,
            None,
            recorder,
            MockTranscriber::new(),
        );

        harness.handle_event(HotkeyEvent::Pressed);
        harness.handle_event(HotkeyEvent::Released);

        assert!(harness.typed_texts.lock().unwrap().is_empty());

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn recording_too_short_skips_transcription() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_too_short();

        let mut harness = TestHarness::with_mocks(
            InteractionMode::PushToTalk,
            None,
            recorder,
            MockTranscriber::new(),
        );

        harness.handle_event(HotkeyEvent::Pressed);
        harness.handle_event(HotkeyEvent::Released);

        assert!(harness.typed_texts.lock().unwrap().is_empty());

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn shutdown_during_idle() {
        let mut harness = TestHarness::new(InteractionMode::PushToTalk, None);

        let shutdown = harness.handle_event(HotkeyEvent::Shutdown);
        assert!(shutdown);

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(*states, vec![PipelineState::Idle]);
    }

    #[test]
    fn shutdown_during_recording() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.1]);

        let mut harness = TestHarness::with_mocks(
            InteractionMode::PushToTalk,
            None,
            recorder,
            MockTranscriber::new(),
        );

        // Start recording
        harness.handle_event(HotkeyEvent::Pressed);

        // Shutdown while recording
        let shutdown = harness.handle_event(HotkeyEvent::Shutdown);
        assert!(shutdown);

        // No text output
        assert!(harness.typed_texts.lock().unwrap().is_empty());

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(*states, vec![PipelineState::Recording, PipelineState::Idle]);
    }

    #[test]
    fn no_active_recording_finish_recovers() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.1]);

        let mut harness = TestHarness::with_mocks(
            InteractionMode::PushToTalk,
            None,
            recorder,
            MockTranscriber::new(),
        );

        // Start recording
        harness.handle_event(HotkeyEvent::Pressed);

        // Manually clear the active recording to simulate a missing recording
        harness.executor.active_recording.take();

        // Release -> tries to finish recording -> no active recording -> recovers
        let shutdown = harness.handle_event(HotkeyEvent::Released);
        assert!(!shutdown);

        assert!(harness.typed_texts.lock().unwrap().is_empty());

        // Should recover to Idle via RecordingFinishFailed
        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn multiple_sequential_dictations() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.1]);
        recorder.queue_audio(vec![0.2]);

        let mut transcriber = MockTranscriber::new();
        transcriber.queue_success("first");
        transcriber.queue_success("second");

        let mut harness =
            TestHarness::with_mocks(InteractionMode::PushToTalk, None, recorder, transcriber);

        // First dictation
        harness.handle_event(HotkeyEvent::Pressed);
        harness.handle_event(HotkeyEvent::Released);

        // Second dictation
        harness.handle_event(HotkeyEvent::Pressed);
        harness.handle_event(HotkeyEvent::Released);

        let texts = harness.typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], "first");
        assert_eq!(texts[1], "second");

        let states = harness.indicator_states.lock().unwrap();
        assert_eq!(
            *states,
            vec![
                // First cycle
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Transcribing,
                PipelineState::Idle,
                // Second cycle
                PipelineState::Recording,
                PipelineState::Processing,
                PipelineState::Transcribing,
                PipelineState::Idle,
            ]
        );
    }

    #[test]
    fn toggle_to_talk_enters_recording_phase() {
        let mut recorder = MockAudioRecorder::new();
        recorder.queue_audio(vec![0.5]);

        let mut harness = TestHarness::with_mocks(
            InteractionMode::ToggleToTalk,
            None,
            recorder,
            MockTranscriber::new(),
        );

        // Press -> starts recording with VAD
        harness.handle_event(HotkeyEvent::Pressed);

        assert_eq!(
            harness.state_machine.phase(),
            &crate::dictation::DictationPhase::Recording
        );
    }
}
