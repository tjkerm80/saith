use anyhow::{Context, Result};
use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, Device, EventType, InputEvent, KeyCode, RelativeAxisCode};
use std::thread;
use std::time::Duration;

use crate::hotkey::provider::{
    CombinedCapabilities, DeviceProvider, DiscoveredDevice, EventSink, EventSource,
};

/// Production implementation of `DeviceProvider` using real evdev/uinput.
pub(crate) struct EvdevProvider;

/// Event source wrapping a grabbed evdev `Device`.
pub(crate) struct EvdevSource {
    device: Device,
}

/// Event sink wrapping a uinput `VirtualDevice`.
pub(crate) struct EvdevSink {
    device: VirtualDevice,
}

impl DeviceProvider for EvdevProvider {
    type Source = EvdevSource;
    type Sink = EvdevSink;

    fn enumerate_devices(&self) -> Result<Vec<DiscoveredDevice>> {
        let mut devices = Vec::new();

        for (path, device) in evdev::enumerate() {
            let name = device.name().unwrap_or("unknown").to_string();
            let supports_event_type_key = device.supported_events().contains(EventType::KEY);

            let supported_key_codes: Vec<KeyCode> = device
                .supported_keys()
                .map(|keys| keys.iter().collect())
                .unwrap_or_default();

            let supported_relative_axis_codes: Vec<RelativeAxisCode> = device
                .supported_relative_axes()
                .map(|axes| axes.iter().collect())
                .unwrap_or_default();

            devices.push(DiscoveredDevice {
                path,
                name,
                supports_event_type_key,
                supported_key_codes,
                supported_relative_axis_codes,
            });
        }

        Ok(devices)
    }

    fn grab_device(&self, device: &DiscoveredDevice) -> Result<Self::Source> {
        let mut evdev_device = Device::open(&device.path).with_context(|| {
            format!(
                "Failed to open device '{}' at {}",
                device.name,
                device.path.display()
            )
        })?;

        evdev_device.grab().with_context(|| {
            format!(
                "Failed to grab device '{}' at {}",
                device.name,
                device.path.display()
            )
        })?;

        Ok(EvdevSource {
            device: evdev_device,
        })
    }

    fn create_forwarding_device(&self, capabilities: &CombinedCapabilities) -> Result<Self::Sink> {
        let mut key_set = AttributeSet::<KeyCode>::new();
        for key in &capabilities.key_codes {
            key_set.insert(*key);
        }

        let mut relative_axis_set = AttributeSet::<RelativeAxisCode>::new();
        for axis in &capabilities.relative_axis_codes {
            relative_axis_set.insert(*axis);
        }

        let mut builder = VirtualDevice::builder()
            .context("Failed to open /dev/uinput for forwarding device")?
            .name("saith-hotkey-forward")
            .with_keys(&key_set)
            .context("Failed to register forwarding key capabilities")?;

        if !capabilities.relative_axis_codes.is_empty() {
            builder = builder
                .with_relative_axes(&relative_axis_set)
                .context("Failed to register forwarding relative axis capabilities")?;
        }

        let virtual_device = builder
            .build()
            .context("Failed to build forwarding device")?;

        // Give the system time to register the forwarding device
        thread::sleep(Duration::from_millis(100));

        Ok(EvdevSink {
            device: virtual_device,
        })
    }
}

impl EventSource for EvdevSource {
    fn fetch_events(&mut self) -> Result<Vec<InputEvent>> {
        let events = self
            .device
            .fetch_events()
            .context("Device disconnected or read error")?
            .collect();
        Ok(events)
    }
}

impl EventSink for EvdevSink {
    fn emit(&mut self, events: &[InputEvent]) -> Result<()> {
        self.device
            .emit(events)
            .context("Failed to emit events to forwarding device")?;
        Ok(())
    }
}
