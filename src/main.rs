mod audio;
mod configuration;
mod dictation;
mod executor;
mod hotkey;
#[cfg(feature = "indicator")]
mod indicator;
mod indicator_message;
#[cfg(feature = "indicator")]
mod indicator_state;
mod output;
mod pipeline_state;
mod transcription;

use std::time::Instant;

use anyhow::{Context, Result};
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;

use audio::pipeline::{AudioPipeline, VoiceActivityDetectionConfig};
use audio::source::CpalAudioSource;
use configuration::Configuration;
use dictation::DictationStateMachine;
use executor::DictationExecutor;
use hotkey::{HotkeyEvent, HotkeyListener};
use indicator_message::IndicatorMessage;
use output::TranscriptionOutput;
use transcription::TranscriptionEngine;

fn main() -> Result<()> {
    println!("saith — local speech-to-text dictation\n");

    println!("Loading configuration...");
    let configuration =
        configuration::load_configuration().context("Failed to load configuration")?;
    let model_size = configuration.resolved_model_size();
    println!("  Model: {model_size}");
    println!("  Interaction mode: {:?}", configuration.interaction_mode);
    println!("  Hotkey: {}", configuration.hotkey.key_code);

    let (pipeline_state_sender, _pipeline_state_receiver) =
        crossbeam_channel::unbounded::<IndicatorMessage>();

    #[cfg(feature = "indicator")]
    if configuration.indicator.show {
        let pipeline_state_receiver = _pipeline_state_receiver;
        let indicator_position = configuration.indicator.position;
        std::thread::Builder::new()
            .name("pipeline".to_string())
            .spawn(move || {
                if let Err(error) = run_pipeline(configuration, pipeline_state_sender) {
                    eprintln!("Pipeline error: {error}");
                    std::process::exit(1);
                }
            })?;
        return indicator::run_with_indicator(pipeline_state_receiver, indicator_position)
            .map_err(|error| anyhow::anyhow!("Indicator error: {error}"));
    }

    run_pipeline(configuration, pipeline_state_sender)
}

fn run_pipeline(
    configuration: Configuration,
    pipeline_state_sender: crossbeam_channel::Sender<IndicatorMessage>,
) -> Result<()> {
    let model_size = configuration.resolved_model_size();
    let backend_preference = configuration.resolved_backend_preference();

    println!("Loading Whisper model (this may download on first run)...");
    let load_start = Instant::now();
    let engine = TranscriptionEngine::load(model_size, backend_preference)
        .context("Failed to load Whisper model")?;
    println!(
        "  Model loaded in {:.1}s (backend: {})",
        load_start.elapsed().as_secs_f64(),
        engine.backend_name()
    );

    println!("Creating virtual keyboard...");
    let transcription_output = TranscriptionOutput::new(configuration.dictionary.clone())
        .context("Failed to create virtual keyboard")?;
    println!("  Virtual keyboard ready");

    println!("Probing audio input device...");
    let source =
        CpalAudioSource::from_default_device().context("Failed to probe audio input device")?;
    let voice_activity_detection_config =
        VoiceActivityDetectionConfig::from(&configuration.voice_activity_detection);
    let audio_pipeline = AudioPipeline::new(Box::new(source), voice_activity_detection_config);
    println!("  Audio device: {}", audio_pipeline.device_description());

    let hotkey_code =
        hotkey::parse_key_code(&configuration.hotkey.key_code).with_context(|| {
            format!(
                "Unknown hotkey key code '{}'. Use evdev names like KEY_RIGHTMETA, KEY_F12, etc.",
                configuration.hotkey.key_code
            )
        })?;

    println!("Setting up global hotkey listener...");
    let (hotkey_sender, hotkey_receiver) = crossbeam_channel::unbounded();

    let mut signals =
        Signals::new([SIGTERM, SIGINT]).context("Failed to register signal handlers")?;
    let shutdown_sender = hotkey_sender.clone();
    std::thread::Builder::new()
        .name("signal-handler".to_string())
        .spawn(move || {
            if signals.forever().next().is_some() {
                let _ = shutdown_sender.send(HotkeyEvent::Shutdown);
            }
        })
        .context("Failed to spawn signal handler thread")?;

    let _hotkey_listener = HotkeyListener::new(hotkey_code, hotkey_sender.clone())
        .context("Failed to set up hotkey listener")?;

    println!(
        "\nReady! Press {} to dictate.\n",
        configuration.hotkey.key_code
    );

    let mut state_machine = DictationStateMachine::new(
        configuration.interaction_mode,
        configuration.dictionary.initial_prompt.clone(),
    );
    let mut executor = DictationExecutor::new(
        audio_pipeline,
        engine,
        transcription_output,
        pipeline_state_sender,
        hotkey_sender,
    );

    loop {
        let event = hotkey_receiver
            .recv()
            .context("Hotkey listener channel closed unexpectedly")?;

        let transition = state_machine.handle_event(event);
        if executor.execute_transition(&mut state_machine, transition) {
            println!("\nReceived shutdown signal, exiting...");
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use signal_hook::consts::{SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    #[test]
    fn signal_handler_registers_and_drops_cleanly() {
        let signals = Signals::new([SIGTERM, SIGINT]);
        assert!(signals.is_ok(), "Failed to register signal handlers");
        drop(signals.unwrap());
    }
}
