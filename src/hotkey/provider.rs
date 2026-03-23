use anyhow::Result;
use evdev::{InputEvent, KeyCode, RelativeAxisCode};
use std::path::PathBuf;

/// Information about a discovered input device, decoupled from the real evdev Device.
#[derive(Debug, Clone)]
pub(crate) struct DiscoveredDevice {
    pub path: PathBuf,
    pub name: String,
    pub supports_event_type_key: bool,
    pub supported_key_codes: Vec<KeyCode>,
    pub supported_relative_axis_codes: Vec<RelativeAxisCode>,
}

/// Combined capabilities aggregated from all keyboards that will be grabbed.
#[derive(Debug, Clone, Default)]
pub(crate) struct CombinedCapabilities {
    pub key_codes: Vec<KeyCode>,
    pub relative_axis_codes: Vec<RelativeAxisCode>,
}

/// Provides access to input devices on the system.
///
/// The production implementation uses evdev; tests substitute a mock.
pub(crate) trait DeviceProvider {
    type Source: EventSource + Send + 'static;
    type Sink: EventSink + Send + 'static;

    /// Enumerate all input devices on the system.
    fn enumerate_devices(&self) -> Result<Vec<DiscoveredDevice>>;

    /// Grab a device for exclusive access and return an event source.
    fn grab_device(&self, device: &DiscoveredDevice) -> Result<Self::Source>;

    /// Create a virtual device for forwarding events with the given capabilities.
    fn create_forwarding_device(&self, capabilities: &CombinedCapabilities) -> Result<Self::Sink>;
}

/// Reads events from a grabbed input device.
pub(crate) trait EventSource {
    /// Fetch the next batch of events. Returns an error if the device disconnects.
    fn fetch_events(&mut self) -> Result<Vec<InputEvent>>;
}

/// Emits events to a virtual forwarding device.
pub(crate) trait EventSink {
    /// Emit a batch of events to the virtual device.
    fn emit(&mut self, events: &[InputEvent]) -> Result<()>;
}
