use crate::audio::pipeline::RecordingOutcome;
use crate::configuration::InteractionMode;
use crate::hotkey::HotkeyEvent;
use crate::pipeline_state::PipelineState;

/// Side effects the state machine requests. Produced by handle_event/handle_outcome.
#[derive(Debug, PartialEq)]
pub enum Effect {
    StartRecording,
    StartRecordingWithVoiceActivityDetection,
    FinishRecording,
    Transcribe {
        samples: Vec<f32>,
        initial_prompt: Option<String>,
    },
    OutputTranscription(String),
    UpdateIndicator(PipelineState),
    DiscardRecording,
    Shutdown,
}

/// Results fed back after the executor completes an effect.
#[derive(Debug)]
pub enum EffectOutcome {
    RecordingStarted,
    RecordingFailed(String),
    RecordingFinished { outcome: RecordingOutcome },
    RecordingFinishFailed { error: String },
    TranscriptionCompleted(String),
    TranscriptionFailed(String),
    OutputCompleted,
}

/// Internal phase of the dictation pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DictationPhase {
    Idle,
    Recording,
    Processing,
    Transcribing,
    Typing,
}

/// A set of effects returned from a state machine transition.
pub struct Transition {
    pub effects: Vec<Effect>,
}

impl Transition {
    fn empty() -> Self {
        Transition {
            effects: Vec::new(),
        }
    }

    fn with(effects: Vec<Effect>) -> Self {
        Transition { effects }
    }
}

/// Pure state machine for dictation pipeline logic.
///
/// Contains no hardware dependencies — produces `Effect` values that an executor
/// interprets using real hardware, and accepts `EffectOutcome` values as feedback.
pub struct DictationStateMachine {
    interaction_mode: InteractionMode,
    initial_prompt: Option<String>,
    phase: DictationPhase,
}

impl DictationStateMachine {
    pub fn new(interaction_mode: InteractionMode, initial_prompt: Option<String>) -> Self {
        Self {
            interaction_mode,
            initial_prompt,
            phase: DictationPhase::Idle,
        }
    }

    #[cfg(test)]
    pub fn phase(&self) -> &DictationPhase {
        &self.phase
    }

    /// Process a hotkey event and return the effects to execute.
    pub fn handle_event(&mut self, event: HotkeyEvent) -> Transition {
        if event == HotkeyEvent::Shutdown {
            return self.handle_shutdown();
        }

        match self.interaction_mode {
            InteractionMode::PushToTalk => self.handle_push_to_talk_event(event),
            InteractionMode::ToggleToTalk => self.handle_toggle_to_talk_event(event),
        }
    }

    /// Process the result of a completed effect and return any follow-up effects.
    pub fn handle_outcome(&mut self, outcome: EffectOutcome) -> Transition {
        match outcome {
            EffectOutcome::RecordingStarted => {
                self.phase = DictationPhase::Recording;
                Transition::with(vec![Effect::UpdateIndicator(PipelineState::Recording)])
            }

            EffectOutcome::RecordingFailed(error) => {
                eprintln!("Failed to start recording: {error}");
                self.phase = DictationPhase::Idle;
                Transition::empty()
            }

            EffectOutcome::RecordingFinished { outcome } => match outcome {
                RecordingOutcome::Audio(samples) => {
                    self.phase = DictationPhase::Transcribing;
                    Transition::with(vec![
                        Effect::UpdateIndicator(PipelineState::Transcribing),
                        Effect::Transcribe {
                            samples,
                            initial_prompt: self.initial_prompt.clone(),
                        },
                    ])
                }
                RecordingOutcome::TooShort | RecordingOutcome::NoSpeechDetected => {
                    self.phase = DictationPhase::Idle;
                    Transition::with(vec![Effect::UpdateIndicator(PipelineState::Idle)])
                }
            },

            EffectOutcome::RecordingFinishFailed { error } => {
                eprintln!("Recording processing failed: {error}");
                self.phase = DictationPhase::Idle;
                Transition::with(vec![Effect::UpdateIndicator(PipelineState::Idle)])
            }

            EffectOutcome::TranscriptionCompleted(text) => {
                if text.is_empty() {
                    println!("  Empty transcription, nothing to output.");
                    self.phase = DictationPhase::Idle;
                    Transition::with(vec![Effect::UpdateIndicator(PipelineState::Idle)])
                } else {
                    self.phase = DictationPhase::Typing;
                    Transition::with(vec![
                        Effect::UpdateIndicator(PipelineState::Idle),
                        Effect::OutputTranscription(text),
                    ])
                }
            }

            EffectOutcome::TranscriptionFailed(error) => {
                eprintln!("Transcription failed: {error}");
                self.phase = DictationPhase::Idle;
                Transition::with(vec![Effect::UpdateIndicator(PipelineState::Idle)])
            }

            EffectOutcome::OutputCompleted => {
                self.phase = DictationPhase::Idle;
                Transition::empty()
            }
        }
    }

    fn handle_shutdown(&mut self) -> Transition {
        let mut effects = Vec::new();

        if self.phase == DictationPhase::Recording {
            effects.push(Effect::DiscardRecording);
        }

        effects.push(Effect::UpdateIndicator(PipelineState::Idle));
        effects.push(Effect::Shutdown);

        self.phase = DictationPhase::Idle;
        Transition::with(effects)
    }

    fn handle_push_to_talk_event(&mut self, event: HotkeyEvent) -> Transition {
        match event {
            HotkeyEvent::Pressed => {
                if self.phase == DictationPhase::Idle {
                    Transition::with(vec![Effect::StartRecording])
                } else {
                    Transition::empty()
                }
            }
            HotkeyEvent::Released => {
                if self.phase == DictationPhase::Recording {
                    self.phase = DictationPhase::Processing;
                    Transition::with(vec![
                        Effect::UpdateIndicator(PipelineState::Processing),
                        Effect::FinishRecording,
                    ])
                } else {
                    Transition::empty()
                }
            }
            HotkeyEvent::AutoStopped => Transition::empty(),
            HotkeyEvent::Shutdown => unreachable!("handled in handle_event"),
        }
    }

    fn handle_toggle_to_talk_event(&mut self, event: HotkeyEvent) -> Transition {
        match event {
            HotkeyEvent::Pressed => {
                if self.phase == DictationPhase::Idle {
                    Transition::with(vec![Effect::StartRecordingWithVoiceActivityDetection])
                } else if self.phase == DictationPhase::Recording {
                    self.phase = DictationPhase::Processing;
                    Transition::with(vec![
                        Effect::UpdateIndicator(PipelineState::Processing),
                        Effect::FinishRecording,
                    ])
                } else {
                    Transition::empty()
                }
            }
            HotkeyEvent::AutoStopped => {
                if self.phase == DictationPhase::Recording {
                    self.phase = DictationPhase::Processing;
                    Transition::with(vec![
                        Effect::UpdateIndicator(PipelineState::Processing),
                        Effect::FinishRecording,
                    ])
                } else {
                    Transition::empty()
                }
            }
            HotkeyEvent::Released => Transition::empty(),
            HotkeyEvent::Shutdown => unreachable!("handled in handle_event"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper ---

    fn assert_effects(transition: &Transition, expected: &[Effect]) {
        assert_eq!(
            transition.effects.len(),
            expected.len(),
            "Expected {} effects, got {}: {:?}",
            expected.len(),
            transition.effects.len(),
            transition.effects
        );
        for (actual, expected) in transition.effects.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }

    // --- PushToTalk happy path ---

    #[test]
    fn push_to_talk_happy_path() {
        let mut state_machine =
            DictationStateMachine::new(InteractionMode::PushToTalk, Some("test prompt".into()));

        // Pressed -> StartRecording
        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(&transition, &[Effect::StartRecording]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle); // still idle until outcome

        // RecordingStarted -> Recording phase + UpdateIndicator
        let transition = state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        assert_effects(
            &transition,
            &[Effect::UpdateIndicator(PipelineState::Recording)],
        );
        assert_eq!(state_machine.phase(), &DictationPhase::Recording);

        // Released -> Processing + FinishRecording
        let transition = state_machine.handle_event(HotkeyEvent::Released);
        assert_effects(
            &transition,
            &[
                Effect::UpdateIndicator(PipelineState::Processing),
                Effect::FinishRecording,
            ],
        );
        assert_eq!(state_machine.phase(), &DictationPhase::Processing);

        // RecordingFinished with audio -> Transcribing + Transcribe
        let samples = vec![0.1, 0.2, 0.3];
        let transition = state_machine.handle_outcome(EffectOutcome::RecordingFinished {
            outcome: RecordingOutcome::Audio(samples.clone()),
        });
        assert_effects(
            &transition,
            &[
                Effect::UpdateIndicator(PipelineState::Transcribing),
                Effect::Transcribe {
                    samples,
                    initial_prompt: Some("test prompt".into()),
                },
            ],
        );
        assert_eq!(state_machine.phase(), &DictationPhase::Transcribing);

        // TranscriptionCompleted -> OutputTranscription
        let transition = state_machine
            .handle_outcome(EffectOutcome::TranscriptionCompleted("hello world".into()));
        assert_effects(
            &transition,
            &[
                Effect::UpdateIndicator(PipelineState::Idle),
                Effect::OutputTranscription("hello world".into()),
            ],
        );
        assert_eq!(state_machine.phase(), &DictationPhase::Typing);

        // OutputCompleted -> Idle
        let transition = state_machine.handle_outcome(EffectOutcome::OutputCompleted);
        assert_effects(&transition, &[]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    // --- ToggleToTalk happy path ---

    #[test]
    fn toggle_to_talk_happy_path_press_to_stop() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::ToggleToTalk, None);

        // Pressed -> StartRecordingWithVAD
        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(
            &transition,
            &[Effect::StartRecordingWithVoiceActivityDetection],
        );

        // RecordingStarted
        let transition = state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        assert_effects(
            &transition,
            &[Effect::UpdateIndicator(PipelineState::Recording)],
        );

        // Pressed again -> FinishRecording
        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(
            &transition,
            &[
                Effect::UpdateIndicator(PipelineState::Processing),
                Effect::FinishRecording,
            ],
        );
        assert_eq!(state_machine.phase(), &DictationPhase::Processing);
    }

    #[test]
    fn toggle_to_talk_auto_stopped() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::ToggleToTalk, None);

        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(
            &transition,
            &[Effect::StartRecordingWithVoiceActivityDetection],
        );

        state_machine.handle_outcome(EffectOutcome::RecordingStarted);

        // AutoStopped -> FinishRecording
        let transition = state_machine.handle_event(HotkeyEvent::AutoStopped);
        assert_effects(
            &transition,
            &[
                Effect::UpdateIndicator(PipelineState::Processing),
                Effect::FinishRecording,
            ],
        );
    }

    // --- Ignored events ---

    #[test]
    fn push_to_talk_ignores_auto_stopped() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        let transition = state_machine.handle_event(HotkeyEvent::AutoStopped);
        assert_effects(&transition, &[]);
    }

    #[test]
    fn toggle_to_talk_ignores_released() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::ToggleToTalk, None);

        let transition = state_machine.handle_event(HotkeyEvent::Released);
        assert_effects(&transition, &[]);
    }

    #[test]
    fn released_while_idle_ignored() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        let transition = state_machine.handle_event(HotkeyEvent::Released);
        assert_effects(&transition, &[]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    #[test]
    fn double_press_while_recording_ignored() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        // Start recording
        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        assert_eq!(state_machine.phase(), &DictationPhase::Recording);

        // Second press while recording -> ignored
        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(&transition, &[]);
        assert_eq!(state_machine.phase(), &DictationPhase::Recording);
    }

    // --- Shutdown ---

    #[test]
    fn shutdown_while_idle() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        let transition = state_machine.handle_event(HotkeyEvent::Shutdown);
        assert_effects(
            &transition,
            &[
                Effect::UpdateIndicator(PipelineState::Idle),
                Effect::Shutdown,
            ],
        );
    }

    #[test]
    fn shutdown_while_recording() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);

        let transition = state_machine.handle_event(HotkeyEvent::Shutdown);
        assert_effects(
            &transition,
            &[
                Effect::DiscardRecording,
                Effect::UpdateIndicator(PipelineState::Idle),
                Effect::Shutdown,
            ],
        );
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    // --- Error recovery ---

    #[test]
    fn recording_failed_returns_to_idle() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        state_machine.handle_event(HotkeyEvent::Pressed);
        let transition =
            state_machine.handle_outcome(EffectOutcome::RecordingFailed("device error".into()));
        assert_effects(&transition, &[]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    #[test]
    fn recording_too_short_returns_to_idle() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        state_machine.handle_event(HotkeyEvent::Released);

        let transition = state_machine.handle_outcome(EffectOutcome::RecordingFinished {
            outcome: RecordingOutcome::TooShort,
        });
        assert_effects(&transition, &[Effect::UpdateIndicator(PipelineState::Idle)]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    #[test]
    fn recording_finish_failed_returns_to_idle() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        state_machine.handle_event(HotkeyEvent::Released);

        let transition = state_machine.handle_outcome(EffectOutcome::RecordingFinishFailed {
            error: "processing error".into(),
        });
        assert_effects(&transition, &[Effect::UpdateIndicator(PipelineState::Idle)]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    #[test]
    fn transcription_failed_returns_to_idle() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        let transition =
            state_machine.handle_outcome(EffectOutcome::TranscriptionFailed("model error".into()));
        assert_effects(&transition, &[Effect::UpdateIndicator(PipelineState::Idle)]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    #[test]
    fn empty_transcription_returns_to_idle_without_output() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        let transition =
            state_machine.handle_outcome(EffectOutcome::TranscriptionCompleted(String::new()));
        assert_effects(&transition, &[Effect::UpdateIndicator(PipelineState::Idle)]);
        assert_eq!(state_machine.phase(), &DictationPhase::Idle);
    }

    // --- Initial prompt forwarding ---

    #[test]
    fn initial_prompt_none_forwarded_to_transcribe_effect() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        state_machine.handle_event(HotkeyEvent::Released);

        let transition = state_machine.handle_outcome(EffectOutcome::RecordingFinished {
            outcome: RecordingOutcome::Audio(vec![0.5]),
        });

        assert_eq!(transition.effects.len(), 2);
        match &transition.effects[1] {
            Effect::Transcribe { initial_prompt, .. } => assert_eq!(initial_prompt, &None),
            other => panic!("Expected Transcribe effect, got {other:?}"),
        }
    }

    // --- Events during non-idle non-recording phases ---

    #[test]
    fn press_during_processing_ignored() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        // Get to Processing phase
        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        state_machine.handle_event(HotkeyEvent::Released);
        assert_eq!(state_machine.phase(), &DictationPhase::Processing);

        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(&transition, &[]);
    }

    #[test]
    fn press_during_transcribing_ignored() {
        let mut state_machine = DictationStateMachine::new(InteractionMode::PushToTalk, None);

        // Get to Transcribing phase
        state_machine.handle_event(HotkeyEvent::Pressed);
        state_machine.handle_outcome(EffectOutcome::RecordingStarted);
        state_machine.handle_event(HotkeyEvent::Released);
        state_machine.handle_outcome(EffectOutcome::RecordingFinished {
            outcome: RecordingOutcome::Audio(vec![0.1]),
        });
        assert_eq!(state_machine.phase(), &DictationPhase::Transcribing);

        let transition = state_machine.handle_event(HotkeyEvent::Pressed);
        assert_effects(&transition, &[]);
    }
}
