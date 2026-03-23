use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use iced::widget::{canvas, center, container, row, text};
use iced::window;
use iced::{Color, Element, Gradient, Length, Point, Radians, Size, Subscription, Task, Theme};

use crate::configuration::IndicatorPosition;
use crate::indicator_message::IndicatorMessage;
use crate::indicator_state::{IndicatorState, WAVEFORM_BAR_COUNT, WindowAction};
use crate::pipeline_state::PipelineState;

use super::position::calculate_position;
use super::theme;

/// Global storage for the pipeline receiver so it can be accessed from a `fn` pointer.
static PIPELINE_RECEIVER: OnceLock<Mutex<Option<crossbeam_channel::Receiver<IndicatorMessage>>>> =
    OnceLock::new();

/// Global storage for the target window position so it can be accessed from the `new()` fn pointer.
static TARGET_POSITION: OnceLock<Point> = OnceLock::new();

/// Launch the iced daemon on the current thread (blocks forever).
///
/// Unlike an `iced::application`, a daemon starts with no windows. The indicator
/// window is created when recording begins and destroyed when the pipeline returns
/// to idle.
pub fn run_with_indicator(
    receiver: crossbeam_channel::Receiver<IndicatorMessage>,
    position: IndicatorPosition,
) -> iced::Result {
    PIPELINE_RECEIVER.get_or_init(|| Mutex::new(Some(receiver)));
    let initial_position = calculate_position(position);
    TARGET_POSITION.get_or_init(|| initial_position);

    iced::daemon(
        StatusIndicator::new,
        StatusIndicator::update,
        StatusIndicator::view,
    )
    .subscription(StatusIndicator::subscription)
    .style(
        |_state: &StatusIndicator, _theme: &Theme| iced::theme::Style {
            background_color: Color::TRANSPARENT,
            text_color: Color::WHITE,
        },
    )
    .theme(theme_fn)
    .run()
}

fn theme_fn(_state: &StatusIndicator, _window: window::Id) -> Theme {
    Theme::Dark
}

#[derive(Debug, Clone)]
enum Message {
    StateChanged(PipelineState),
    WaveformSample(f32),
    AnimationTick(()),
    IndicatorWindowOpened,
    PipelineExited,
}

struct StatusIndicator {
    state: IndicatorState,
    indicator_window_id: Option<window::Id>,
    target_position: Point,
    epoch: Instant,
}

impl StatusIndicator {
    fn new() -> (Self, Task<Message>) {
        let target_position = TARGET_POSITION.get().copied().unwrap_or(Point::ORIGIN);

        (
            Self {
                state: IndicatorState::new(),
                indicator_window_id: None,
                target_position,
                epoch: Instant::now(),
            },
            Task::none(),
        )
    }

    fn indicator_window_settings(&self) -> window::Settings {
        window::Settings {
            size: Size::new(theme::PILL_WIDTH, theme::PILL_HEIGHT),
            position: window::Position::Specific(self.target_position),
            visible: true,
            transparent: true,
            decorations: false,
            level: window::Level::AlwaysOnTop,
            #[cfg(target_os = "linux")]
            platform_specific: window::settings::PlatformSpecific {
                application_id: "saith-indicator".to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn open_indicator_window(&mut self) -> Task<Message> {
        let (id, open_task) = window::open(self.indicator_window_settings());
        self.indicator_window_id = Some(id);
        let target = self.target_position;
        Task::batch([
            open_task.map(move |_| Message::IndicatorWindowOpened),
            window::move_to(id, target),
        ])
    }

    fn close_indicator_window(&mut self) -> Task<Message> {
        if let Some(id) = self.indicator_window_id.take() {
            window::close::<Message>(id)
        } else {
            Task::none()
        }
    }

    fn handle_window_action(&mut self, action: WindowAction) -> Task<Message> {
        match action {
            WindowAction::None => Task::none(),
            WindowAction::Open => self.open_indicator_window(),
            WindowAction::Close => self.close_indicator_window(),
        }
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PipelineExited => iced::exit(),
            Message::IndicatorWindowOpened => Task::none(),
            Message::WaveformSample(amplitude) => {
                self.state.push_waveform_sample(amplitude);
                Task::none()
            }
            Message::StateChanged(pipeline_state) => {
                let action = self
                    .state
                    .transition_to(pipeline_state, self.epoch.elapsed());
                self.handle_window_action(action)
            }
            Message::AnimationTick(()) => {
                self.state.tick(self.epoch.elapsed());
                Task::none()
            }
        }
    }

    fn view(&self, _window_id: window::Id) -> Element<'_, Message> {
        let Some(frame) = self.state.frame(self.epoch.elapsed()) else {
            return container(text(""))
                .width(Length::Fill)
                .height(Length::Fill)
                .style(|_theme: &Theme| container::Style {
                    background: Some(Color::TRANSPARENT.into()),
                    ..Default::default()
                })
                .into();
        };

        let pulsed_color = theme::to_iced_color(&frame.dot_color);
        let glow_color = theme::to_iced_color(&frame.glow_color);

        let dot = canvas(DotCanvas {
            color: pulsed_color,
            glow_color,
        })
        .width(20)
        .height(20);

        let waveform = canvas(WaveformCanvas {
            amplitudes: frame.waveform_amplitudes,
        })
        .width(theme::WAVEFORM_TOTAL_WIDTH)
        .height(40);

        let pill_content = row![dot, waveform]
            .spacing(10)
            .align_y(iced::Alignment::Center);

        let border_width = 2.0;
        let inner_radius = theme::PILL_BORDER_RADIUS - border_width;

        let inner_pill = container(pill_content)
            .padding(iced::Padding {
                top: 14.0,
                right: 24.0,
                bottom: 14.0,
                left: 18.0,
            })
            .style(move |_theme: &Theme| container::Style {
                background: Some(theme::PILL_BACKGROUND.into()),
                border: iced::Border {
                    radius: inner_radius.into(),
                    width: 0.0,
                    color: Color::TRANSPARENT,
                },
                ..Default::default()
            });

        let metallic_gradient = Gradient::Linear(
            iced::gradient::Linear::new(Radians(std::f32::consts::PI))
                .add_stop(0.0, Color::from_rgb(0.52, 0.54, 0.58))
                .add_stop(0.3, Color::from_rgb(0.38, 0.40, 0.44))
                .add_stop(0.5, Color::from_rgb(0.28, 0.30, 0.33))
                .add_stop(0.7, Color::from_rgb(0.38, 0.40, 0.44))
                .add_stop(1.0, Color::from_rgb(0.48, 0.50, 0.54)),
        );

        let pill = container(inner_pill)
            .padding(border_width)
            .style(move |_theme: &Theme| container::Style {
                background: Some(metallic_gradient.into()),
                border: iced::Border {
                    radius: theme::PILL_BORDER_RADIUS.into(),
                    width: 0.0,
                    color: Color::TRANSPARENT,
                },
                shadow: iced::Shadow {
                    color: Color::from_rgba(0.6, 0.62, 0.66, 0.2),
                    offset: iced::Vector::new(0.0, 1.0),
                    blur_radius: 6.0,
                },
                text_color: None,
                snap: false,
            });

        center(pill).width(Length::Fill).height(Length::Fill).into()
    }

    fn subscription(&self) -> Subscription<Message> {
        let mut subscriptions = vec![];

        subscriptions.push(Subscription::run(pipeline_state_stream));

        if self.state.frame(self.epoch.elapsed()).is_some() {
            subscriptions.push(
                iced::time::every(Duration::from_millis(
                    theme::ANIMATION_INTERVAL_MILLISECONDS,
                ))
                .map(|_| Message::AnimationTick(())),
            );
        }

        Subscription::batch(subscriptions)
    }
}

fn pipeline_state_stream() -> impl iced::futures::Stream<Item = Message> {
    iced::stream::channel(
        32,
        |mut output: iced::futures::channel::mpsc::Sender<Message>| async move {
            use iced::futures::SinkExt;

            let receiver = PIPELINE_RECEIVER
                .get()
                .and_then(|mutex| mutex.lock().unwrap().take());

            let Some(receiver) = receiver else {
                std::future::pending::<()>().await;
                unreachable!();
            };

            loop {
                let receiver_for_blocking = receiver.clone();
                let result = tokio::task::spawn_blocking(move || receiver_for_blocking.recv())
                    .await
                    .unwrap();
                match result {
                    Ok(indicator_message) => {
                        let message = match indicator_message {
                            IndicatorMessage::StateChanged(state) => Message::StateChanged(state),
                            IndicatorMessage::WaveformSample(amplitude) => {
                                Message::WaveformSample(amplitude)
                            }
                        };
                        let _ = output.send(message).await;
                    }
                    Err(_) => {
                        let _ = output.send(Message::PipelineExited).await;
                        return;
                    }
                }
            }
        },
    )
}

struct WaveformCanvas {
    amplitudes: [f32; WAVEFORM_BAR_COUNT],
}

impl<Message> canvas::Program<Message> for WaveformCanvas {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let center_y = bounds.height / 2.0;
        let max_half_height = center_y - 1.0;

        for (index, &amplitude) in self.amplitudes.iter().enumerate() {
            let clamped = amplitude.clamp(0.0, 1.0);
            let half_height = (clamped * max_half_height).max(1.0);
            let opacity = 0.4 + 0.6 * clamped;
            let color = Color::from_rgba(1.0, 1.0, 1.0, opacity);
            let x = index as f32 * (theme::WAVEFORM_BAR_WIDTH + theme::WAVEFORM_BAR_GAP);

            frame.fill_rectangle(
                Point::new(x, center_y - half_height),
                Size::new(theme::WAVEFORM_BAR_WIDTH, half_height),
                color,
            );

            frame.fill_rectangle(
                Point::new(x, center_y),
                Size::new(theme::WAVEFORM_BAR_WIDTH, half_height),
                color,
            );
        }

        vec![frame.into_geometry()]
    }
}

struct DotCanvas {
    color: Color,
    glow_color: Color,
}

const GLOW_RING_COUNT: usize = 8;

impl<Message> canvas::Program<Message> for DotCanvas {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let center = frame.center();
        let radius = bounds.width.min(bounds.height) / 2.0;
        let dot_radius = radius * 0.5;
        let max_glow_radius = radius * 1.3;

        for ring in 0..GLOW_RING_COUNT {
            let fraction = 1.0 - (ring as f32 / GLOW_RING_COUNT as f32);
            let ring_radius = dot_radius + (max_glow_radius - dot_radius) * fraction;
            let alpha = self.glow_color.a * (1.0 - fraction) * (1.0 - fraction);
            let ring_color = Color::from_rgba(
                self.glow_color.r,
                self.glow_color.g,
                self.glow_color.b,
                alpha,
            );
            frame.fill(&canvas::Path::circle(center, ring_radius), ring_color);
        }

        frame.fill(&canvas::Path::circle(center, dot_radius), self.color);

        vec![frame.into_geometry()]
    }
}
