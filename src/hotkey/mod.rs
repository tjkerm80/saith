pub(crate) mod evdev_provider;
pub(crate) mod event_routing;
pub(crate) mod provider;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use evdev::KeyCode;
use std::sync::{Arc, Mutex};
use std::thread;

use event_routing::process_event_batch;
use provider::{CombinedCapabilities, DeviceProvider, EventSink, EventSource};

/// Events sent from the hotkey listener to the main loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    Pressed,
    Released,
    AutoStopped,
    Shutdown,
}

/// Listens for a global hotkey by grabbing all keyboard devices via evdev.
///
/// Non-hotkey events are forwarded through a uinput virtual device so that
/// normal typing continues to work while the hotkey is captured.
#[derive(Debug)]
pub struct HotkeyListener {
    _threads: Vec<thread::JoinHandle<()>>,
}

impl HotkeyListener {
    /// Create a new hotkey listener that grabs all detected keyboards.
    ///
    /// Requires root or membership in the `input` group.
    pub fn new(hotkey_code: KeyCode, sender: Sender<HotkeyEvent>) -> Result<Self> {
        Self::with_provider(evdev_provider::EvdevProvider, hotkey_code, sender)
    }

    /// Create a hotkey listener using the given device provider.
    ///
    /// This constructor exists so that tests can substitute a mock provider.
    pub(crate) fn with_provider<P: DeviceProvider>(
        device_provider: P,
        hotkey_code: KeyCode,
        sender: Sender<HotkeyEvent>,
    ) -> Result<Self> {
        let all_devices = device_provider
            .enumerate_devices()
            .context("Failed to enumerate input devices")?;

        let keyboards: Vec<_> = all_devices
            .into_iter()
            .filter(|device| {
                event_routing::is_keyboard(device) && !device.name.starts_with("saith-")
            })
            .collect();

        if keyboards.is_empty() {
            anyhow::bail!(
                "No keyboard devices found. Are you running as root or in the 'input' group?"
            );
        }

        let mut capabilities = CombinedCapabilities::default();
        for keyboard in &keyboards {
            for key in &keyboard.supported_key_codes {
                if !capabilities.key_codes.contains(key) {
                    capabilities.key_codes.push(*key);
                }
            }
            for axis in &keyboard.supported_relative_axis_codes {
                if !capabilities.relative_axis_codes.contains(axis) {
                    capabilities.relative_axis_codes.push(*axis);
                }
            }
        }

        let forwarding_device = device_provider
            .create_forwarding_device(&capabilities)
            .context("Failed to create event forwarding device")?;
        let forwarding_device = Arc::new(Mutex::new(forwarding_device));

        let mut threads = Vec::new();

        for keyboard in keyboards {
            println!(
                "  Grabbing keyboard: {} ({})",
                keyboard.name,
                keyboard.path.display()
            );

            let source = device_provider.grab_device(&keyboard).with_context(|| {
                format!(
                    "Failed to grab device '{}' at {}",
                    keyboard.name,
                    keyboard.path.display()
                )
            })?;

            let sender = sender.clone();
            let forwarding = Arc::clone(&forwarding_device);

            let handle = thread::Builder::new()
                .name(format!("hotkey-{}", keyboard.name))
                .spawn(move || {
                    keyboard_event_loop(source, hotkey_code, sender, forwarding);
                })
                .with_context(|| {
                    format!("Failed to spawn listener thread for '{}'", keyboard.name)
                })?;

            threads.push(handle);
        }

        Ok(Self { _threads: threads })
    }
}

/// Blocking event loop for a single grabbed keyboard device.
///
/// Reads events, forwards non-hotkey events through the virtual device,
/// and sends hotkey press/release events to the channel.
fn keyboard_event_loop<S: EventSource, K: EventSink>(
    mut source: S,
    hotkey_code: KeyCode,
    sender: Sender<HotkeyEvent>,
    forwarding_device: Arc<Mutex<K>>,
) {
    loop {
        let events = match source.fetch_events() {
            Ok(events) => events,
            Err(error) => {
                eprintln!("Keyboard device disconnected or error: {error}");
                break;
            }
        };

        let (hotkey_events, forwarded_events) = process_event_batch(&events, hotkey_code);

        for hotkey_event in hotkey_events {
            let _ = sender.send(hotkey_event);
        }

        if !forwarded_events.is_empty()
            && let Ok(mut forward) = forwarding_device.lock()
        {
            let _ = forward.emit(&forwarded_events);
        }
    }
}

/// Parse a key code name like "KEY_RIGHTMETA" into an evdev KeyCode.
pub fn parse_key_code(name: &str) -> Option<KeyCode> {
    for code in 0..768u16 {
        let key_code = KeyCode::new(code);
        let debug_name = format!("{key_code:?}");
        if debug_name == name {
            return Some(key_code);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use evdev::{EventType, InputEvent, RelativeAxisCode};
    use provider::DiscoveredDevice;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- Mock types ---

    /// An event source that yields pre-scripted batches, then returns an error.
    struct ScriptedSource {
        batches: Vec<Vec<InputEvent>>,
        index: AtomicUsize,
    }

    impl ScriptedSource {
        fn new(batches: Vec<Vec<InputEvent>>) -> Self {
            Self {
                batches,
                index: AtomicUsize::new(0),
            }
        }
    }

    impl EventSource for ScriptedSource {
        fn fetch_events(&mut self) -> Result<Vec<InputEvent>> {
            let index = self.index.fetch_add(1, Ordering::SeqCst);
            if index < self.batches.len() {
                Ok(self.batches[index].clone())
            } else {
                anyhow::bail!("Source disconnected")
            }
        }
    }

    /// An event sink that records all emitted events.
    struct RecordingSink {
        recorded_events: Vec<InputEvent>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                recorded_events: Vec::new(),
            }
        }
    }

    impl EventSink for RecordingSink {
        fn emit(&mut self, events: &[InputEvent]) -> Result<()> {
            self.recorded_events.extend_from_slice(events);
            Ok(())
        }
    }

    /// A mock device provider for integration tests.
    struct MockProvider {
        devices: Vec<DiscoveredDevice>,
        /// Pre-scripted event batches for each device (indexed by grab order).
        source_batches: Vec<Vec<Vec<InputEvent>>>,
    }

    impl DeviceProvider for MockProvider {
        type Source = ScriptedSource;
        type Sink = RecordingSink;

        fn enumerate_devices(&self) -> Result<Vec<DiscoveredDevice>> {
            Ok(self.devices.clone())
        }

        fn grab_device(&self, device: &DiscoveredDevice) -> Result<Self::Source> {
            let index = self
                .devices
                .iter()
                .position(|discovered| discovered.path == device.path)
                .unwrap_or(0);
            let batches = self.source_batches.get(index).cloned().unwrap_or_default();
            Ok(ScriptedSource::new(batches))
        }

        fn create_forwarding_device(
            &self,
            _capabilities: &CombinedCapabilities,
        ) -> Result<Self::Sink> {
            Ok(RecordingSink::new())
        }
    }

    fn make_keyboard_device(name: &str, path: &str) -> DiscoveredDevice {
        DiscoveredDevice {
            path: path.into(),
            name: name.to_string(),
            supports_event_type_key: true,
            supported_key_codes: vec![
                KeyCode::KEY_A,
                KeyCode::KEY_Z,
                KeyCode::KEY_SPACE,
                KeyCode::KEY_ENTER,
            ],
            supported_relative_axis_codes: vec![],
        }
    }

    fn make_mouse_device(name: &str, path: &str) -> DiscoveredDevice {
        DiscoveredDevice {
            path: path.into(),
            name: name.to_string(),
            supports_event_type_key: true,
            supported_key_codes: vec![KeyCode::BTN_LEFT, KeyCode::BTN_RIGHT],
            supported_relative_axis_codes: vec![RelativeAxisCode::REL_X, RelativeAxisCode::REL_Y],
        }
    }

    fn make_key_event(code: KeyCode, value: i32) -> InputEvent {
        InputEvent::new(EventType::KEY.0, code.code(), value)
    }

    // --- Existing tests (preserved) ---

    #[test]
    fn parse_known_key_codes() {
        assert_eq!(parse_key_code("KEY_A"), Some(KeyCode::KEY_A));
        assert_eq!(
            parse_key_code("KEY_RIGHTMETA"),
            Some(KeyCode::KEY_RIGHTMETA)
        );
        assert_eq!(parse_key_code("KEY_LEFTCTRL"), Some(KeyCode::KEY_LEFTCTRL));
        assert_eq!(parse_key_code("KEY_F12"), Some(KeyCode::KEY_F12));
    }

    #[test]
    fn parse_unknown_key_code_returns_none() {
        assert_eq!(parse_key_code("KEY_NONEXISTENT"), None);
        assert_eq!(parse_key_code(""), None);
        assert_eq!(parse_key_code("not_a_key"), None);
    }

    #[test]
    fn shutdown_event_sends_through_channel() {
        let (sender, receiver) = crossbeam_channel::unbounded();
        sender.send(HotkeyEvent::Shutdown).unwrap();
        assert_eq!(receiver.recv().unwrap(), HotkeyEvent::Shutdown);
    }

    #[test]
    fn shutdown_event_is_distinct_from_key_events() {
        assert_ne!(HotkeyEvent::Shutdown, HotkeyEvent::Pressed);
        assert_ne!(HotkeyEvent::Shutdown, HotkeyEvent::Released);
    }

    #[test]
    fn auto_stopped_is_distinct_from_other_events() {
        assert_ne!(HotkeyEvent::AutoStopped, HotkeyEvent::Pressed);
        assert_ne!(HotkeyEvent::AutoStopped, HotkeyEvent::Released);
        assert_ne!(HotkeyEvent::AutoStopped, HotkeyEvent::Shutdown);
    }

    // --- Event loop tests ---

    #[test]
    fn event_loop_sends_hotkey_pressed() {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let source = ScriptedSource::new(vec![vec![make_key_event(KeyCode::KEY_RIGHTMETA, 1)]]);
        let sink = Arc::new(Mutex::new(RecordingSink::new()));

        keyboard_event_loop(source, KeyCode::KEY_RIGHTMETA, sender, Arc::clone(&sink));

        assert_eq!(receiver.recv().unwrap(), HotkeyEvent::Pressed);
    }

    #[test]
    fn event_loop_sends_hotkey_released() {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let source = ScriptedSource::new(vec![vec![make_key_event(KeyCode::KEY_RIGHTMETA, 0)]]);
        let sink = Arc::new(Mutex::new(RecordingSink::new()));

        keyboard_event_loop(source, KeyCode::KEY_RIGHTMETA, sender, Arc::clone(&sink));

        assert_eq!(receiver.recv().unwrap(), HotkeyEvent::Released);
    }

    #[test]
    fn event_loop_forwards_non_hotkey_events() {
        let (sender, _receiver) = crossbeam_channel::unbounded();
        let source = ScriptedSource::new(vec![vec![
            make_key_event(KeyCode::KEY_A, 1),
            make_key_event(KeyCode::KEY_A, 0),
        ]]);
        let sink = Arc::new(Mutex::new(RecordingSink::new()));

        keyboard_event_loop(source, KeyCode::KEY_RIGHTMETA, sender, Arc::clone(&sink));

        let recorded = &sink.lock().unwrap().recorded_events;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].code(), KeyCode::KEY_A.code());
        assert_eq!(recorded[1].code(), KeyCode::KEY_A.code());
    }

    #[test]
    fn event_loop_mixed_batch_correctness() {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let source = ScriptedSource::new(vec![vec![
            make_key_event(KeyCode::KEY_A, 1),
            make_key_event(KeyCode::KEY_RIGHTMETA, 1),
            make_key_event(KeyCode::KEY_B, 1),
            make_key_event(KeyCode::KEY_RIGHTMETA, 0),
        ]]);
        let sink = Arc::new(Mutex::new(RecordingSink::new()));

        keyboard_event_loop(source, KeyCode::KEY_RIGHTMETA, sender, Arc::clone(&sink));

        assert_eq!(receiver.recv().unwrap(), HotkeyEvent::Pressed);
        assert_eq!(receiver.recv().unwrap(), HotkeyEvent::Released);

        let recorded = &sink.lock().unwrap().recorded_events;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].code(), KeyCode::KEY_A.code());
        assert_eq!(recorded[1].code(), KeyCode::KEY_B.code());
    }

    #[test]
    fn event_loop_handles_source_disconnect() {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let source = ScriptedSource::new(vec![]);
        let sink = Arc::new(Mutex::new(RecordingSink::new()));

        keyboard_event_loop(source, KeyCode::KEY_RIGHTMETA, sender, Arc::clone(&sink));

        // Loop should exit cleanly — no events sent, no panic
        assert!(receiver.try_recv().is_err());
    }

    // --- with_provider integration tests ---

    #[test]
    fn with_provider_filters_non_keyboards() {
        let provider = MockProvider {
            devices: vec![
                make_mouse_device("Test Mouse", "/dev/input/event0"),
                make_keyboard_device("Test Keyboard", "/dev/input/event1"),
            ],
            source_batches: vec![
                vec![], // mouse — should not be grabbed
                vec![], // keyboard — will be grabbed, source disconnects immediately
            ],
        };

        let (sender, _receiver) = crossbeam_channel::unbounded();
        let result = HotkeyListener::with_provider(provider, KeyCode::KEY_RIGHTMETA, sender);

        assert!(result.is_ok());
        // Only one thread spawned (for the keyboard, not the mouse)
        assert_eq!(result.unwrap()._threads.len(), 1);
    }

    #[test]
    fn with_provider_excludes_saith_devices() {
        let mut saith_device = make_keyboard_device("saith-hotkey-forward", "/dev/input/event0");
        saith_device.name = "saith-hotkey-forward".to_string();

        let provider = MockProvider {
            devices: vec![
                saith_device,
                make_keyboard_device("Real Keyboard", "/dev/input/event1"),
            ],
            source_batches: vec![
                vec![], // saith device — should be skipped
                vec![], // real keyboard
            ],
        };

        let (sender, _receiver) = crossbeam_channel::unbounded();
        let result = HotkeyListener::with_provider(provider, KeyCode::KEY_RIGHTMETA, sender);

        assert!(result.is_ok());
        assert_eq!(result.unwrap()._threads.len(), 1);
    }

    #[test]
    fn with_provider_errors_when_no_keyboards() {
        let provider = MockProvider {
            devices: vec![make_mouse_device("Only Mouse", "/dev/input/event0")],
            source_batches: vec![vec![]],
        };

        let (sender, _receiver) = crossbeam_channel::unbounded();
        let result = HotkeyListener::with_provider(provider, KeyCode::KEY_RIGHTMETA, sender);

        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("No keyboard devices found"));
    }
}
