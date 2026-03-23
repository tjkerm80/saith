use crate::hotkey::HotkeyEvent;
use crate::hotkey::provider::DiscoveredDevice;
use evdev::{EventType, InputEvent, KeyCode};

/// The disposition of a single input event after classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EventDisposition {
    /// The hotkey was pressed — send HotkeyEvent::Pressed.
    HotkeyPressed,
    /// The hotkey was released — send HotkeyEvent::Released.
    HotkeyReleased,
    /// The hotkey autorepeat — ignore silently.
    HotkeyAutorepeat,
    /// Not a hotkey event — forward to the virtual device.
    Forward,
}

/// Classify a single input event relative to the configured hotkey.
pub(crate) fn classify_event(event: &InputEvent, hotkey_code: KeyCode) -> EventDisposition {
    if event.event_type() == EventType::KEY && event.code() == hotkey_code.code() {
        match event.value() {
            1 => EventDisposition::HotkeyPressed,
            0 => EventDisposition::HotkeyReleased,
            _ => EventDisposition::HotkeyAutorepeat,
        }
    } else {
        EventDisposition::Forward
    }
}

/// Process a batch of events, splitting them into hotkey events and forwarded events.
pub(crate) fn process_event_batch(
    events: &[InputEvent],
    hotkey_code: KeyCode,
) -> (Vec<HotkeyEvent>, Vec<InputEvent>) {
    let mut hotkey_events = Vec::new();
    let mut forwarded_events = Vec::new();

    for event in events {
        match classify_event(event, hotkey_code) {
            EventDisposition::HotkeyPressed => hotkey_events.push(HotkeyEvent::Pressed),
            EventDisposition::HotkeyReleased => hotkey_events.push(HotkeyEvent::Released),
            EventDisposition::HotkeyAutorepeat => {}
            EventDisposition::Forward => forwarded_events.push(*event),
        }
    }

    (hotkey_events, forwarded_events)
}

/// Determine whether a discovered device looks like a physical keyboard.
///
/// Must support EV_KEY with letter keys (KEY_A, KEY_Z) and KEY_SPACE. Devices
/// that look like mice (relative axes + mouse buttons but no letter keys) are
/// excluded by the letter-key check.
pub(crate) fn is_keyboard(device: &DiscoveredDevice) -> bool {
    if !device.supports_event_type_key {
        return false;
    }

    let keys = &device.supported_key_codes;

    keys.contains(&KeyCode::KEY_A)
        && keys.contains(&KeyCode::KEY_Z)
        && keys.contains(&KeyCode::KEY_SPACE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key_event(code: KeyCode, value: i32) -> InputEvent {
        InputEvent::new(EventType::KEY.0, code.code(), value)
    }

    fn make_sync_event() -> InputEvent {
        InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0)
    }

    fn make_keyboard_device() -> DiscoveredDevice {
        DiscoveredDevice {
            path: "/dev/input/event0".into(),
            name: "Test Keyboard".to_string(),
            supports_event_type_key: true,
            supported_key_codes: vec![
                KeyCode::KEY_A,
                KeyCode::KEY_Z,
                KeyCode::KEY_SPACE,
                KeyCode::KEY_ENTER,
                KeyCode::KEY_LEFTSHIFT,
            ],
            supported_relative_axis_codes: vec![],
        }
    }

    // --- classify_event tests ---

    #[test]
    fn classify_hotkey_press() {
        let event = make_key_event(KeyCode::KEY_RIGHTMETA, 1);
        assert_eq!(
            classify_event(&event, KeyCode::KEY_RIGHTMETA),
            EventDisposition::HotkeyPressed
        );
    }

    #[test]
    fn classify_hotkey_release() {
        let event = make_key_event(KeyCode::KEY_RIGHTMETA, 0);
        assert_eq!(
            classify_event(&event, KeyCode::KEY_RIGHTMETA),
            EventDisposition::HotkeyReleased
        );
    }

    #[test]
    fn classify_hotkey_autorepeat_ignored() {
        let event = make_key_event(KeyCode::KEY_RIGHTMETA, 2);
        assert_eq!(
            classify_event(&event, KeyCode::KEY_RIGHTMETA),
            EventDisposition::HotkeyAutorepeat
        );
    }

    #[test]
    fn classify_non_hotkey_key_forwarded() {
        let event = make_key_event(KeyCode::KEY_A, 1);
        assert_eq!(
            classify_event(&event, KeyCode::KEY_RIGHTMETA),
            EventDisposition::Forward
        );
    }

    #[test]
    fn classify_non_key_event_forwarded() {
        let event = make_sync_event();
        assert_eq!(
            classify_event(&event, KeyCode::KEY_RIGHTMETA),
            EventDisposition::Forward
        );
    }

    // --- process_event_batch tests ---

    #[test]
    fn process_empty_batch() {
        let (hotkey_events, forwarded) = process_event_batch(&[], KeyCode::KEY_RIGHTMETA);
        assert!(hotkey_events.is_empty());
        assert!(forwarded.is_empty());
    }

    #[test]
    fn process_all_hotkey_events() {
        let events = vec![
            make_key_event(KeyCode::KEY_RIGHTMETA, 1),
            make_key_event(KeyCode::KEY_RIGHTMETA, 0),
        ];
        let (hotkey_events, forwarded) = process_event_batch(&events, KeyCode::KEY_RIGHTMETA);
        assert_eq!(
            hotkey_events,
            vec![HotkeyEvent::Pressed, HotkeyEvent::Released]
        );
        assert!(forwarded.is_empty());
    }

    #[test]
    fn process_no_hotkey_events() {
        let events = vec![
            make_key_event(KeyCode::KEY_A, 1),
            make_key_event(KeyCode::KEY_A, 0),
        ];
        let (hotkey_events, forwarded) = process_event_batch(&events, KeyCode::KEY_RIGHTMETA);
        assert!(hotkey_events.is_empty());
        assert_eq!(forwarded.len(), 2);
    }

    #[test]
    fn process_mixed_batch() {
        let events = vec![
            make_key_event(KeyCode::KEY_A, 1),
            make_key_event(KeyCode::KEY_RIGHTMETA, 1),
            make_key_event(KeyCode::KEY_B, 1),
            make_key_event(KeyCode::KEY_RIGHTMETA, 0),
            make_sync_event(),
        ];
        let (hotkey_events, forwarded) = process_event_batch(&events, KeyCode::KEY_RIGHTMETA);
        assert_eq!(
            hotkey_events,
            vec![HotkeyEvent::Pressed, HotkeyEvent::Released]
        );
        assert_eq!(forwarded.len(), 3);
    }

    // --- is_keyboard tests ---

    #[test]
    fn is_keyboard_accepts_full_keyboard() {
        let device = make_keyboard_device();
        assert!(is_keyboard(&device));
    }

    #[test]
    fn is_keyboard_rejects_mouse() {
        let device = DiscoveredDevice {
            path: "/dev/input/event1".into(),
            name: "Test Mouse".to_string(),
            supports_event_type_key: true,
            supported_key_codes: vec![KeyCode::BTN_LEFT, KeyCode::BTN_RIGHT],
            supported_relative_axis_codes: vec![
                evdev::RelativeAxisCode::REL_X,
                evdev::RelativeAxisCode::REL_Y,
            ],
        };
        assert!(!is_keyboard(&device));
    }

    #[test]
    fn is_keyboard_rejects_no_key_support() {
        let device = DiscoveredDevice {
            path: "/dev/input/event2".into(),
            name: "Test Touchpad".to_string(),
            supports_event_type_key: false,
            supported_key_codes: vec![],
            supported_relative_axis_codes: vec![],
        };
        assert!(!is_keyboard(&device));
    }

    #[test]
    fn is_keyboard_rejects_partial_keyboard() {
        let device = DiscoveredDevice {
            path: "/dev/input/event3".into(),
            name: "Partial Device".to_string(),
            supports_event_type_key: true,
            supported_key_codes: vec![KeyCode::KEY_A, KeyCode::KEY_Z],
            supported_relative_axis_codes: vec![],
        };
        assert!(!is_keyboard(&device));
    }

    #[test]
    fn is_keyboard_rejects_device_missing_key_space() {
        let device = DiscoveredDevice {
            path: "/dev/input/event4".into(),
            name: "Almost Keyboard".to_string(),
            supports_event_type_key: true,
            supported_key_codes: vec![KeyCode::KEY_A, KeyCode::KEY_Z, KeyCode::KEY_ENTER],
            supported_relative_axis_codes: vec![],
        };
        assert!(!is_keyboard(&device));
    }
}
