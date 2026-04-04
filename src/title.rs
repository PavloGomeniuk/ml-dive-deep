use bevy::prelude::*;
use crate::state::GameState;
use crate::components::{Z_OVERLAY, SCREEN_W, SCREEN_H};

pub struct TitlePlugin;
impl Plugin for TitlePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::Title), setup_title)
           .add_systems(Update, handle_title_input.run_if(in_state(GameState::Title)));
    }
}

fn setup_title(mut commands: Commands) {
    // Dark background
    commands.spawn((
        Sprite {
            color: Color::srgb(0.051, 0.051, 0.102),
            custom_size: Some(Vec2::new(SCREEN_W, SCREEN_H)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, Z_OVERLAY),
        StateScoped(GameState::Title),
    ));

    // Title
    commands.spawn((
        Text2d::new("MEDIEVAL RPG"),
        TextFont { font_size: 52.0, ..default() },
        TextColor(Color::srgb(0.831, 0.686, 0.216)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, 80.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Title),
    ));

    // Subtitle
    commands.spawn((
        Text2d::new("Built with Rust + WebAssembly · zero npm"),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::srgb(0.353, 0.227, 0.082)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, 42.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Title),
    ));

    // Divider
    commands.spawn((
        Sprite {
            color: Color::srgb(0.353, 0.227, 0.082),
            custom_size: Some(Vec2::new(400.0, 1.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 15.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Title),
    ));

    // NEW GAME button outline
    commands.spawn((
        Sprite {
            color: Color::srgba(0.831, 0.686, 0.216, 0.15),
            custom_size: Some(Vec2::new(200.0, 42.0)),
            ..default()
        },
        Transform::from_xyz(0.0, -20.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Title),
    ));

    // NEW GAME text
    commands.spawn((
        Text2d::new("⚔  NEW GAME"),
        TextFont { font_size: 20.0, ..default() },
        TextColor(Color::srgb(0.831, 0.686, 0.216)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, -20.0, Z_OVERLAY + 0.6),
        StateScoped(GameState::Title),
    ));

    // Key hint
    commands.spawn((
        Text2d::new("click · ENTER · SPACE"),
        TextFont { font_size: 11.0, ..default() },
        TextColor(Color::srgb(0.25, 0.25, 0.35)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, -55.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Title),
    ));

    // Flavour
    commands.spawn((
        Text2d::new("☠  ENTER IF YOU DARE  ☠"),
        TextFont { font_size: 11.0, ..default() },
        TextColor(Color::srgb(0.18, 0.18, 0.28)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, -170.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Title),
    ));
}

fn handle_title_input(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut next: ResMut<NextState<GameState>>,
) {
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::Space) {
        next.set(GameState::CharacterSelect);
        return;
    }
    if mouse.just_pressed(MouseButton::Left) {
        if let (Ok(win), Ok((cam, cam_tf))) = (windows.get_single(), camera_q.get_single()) {
            if let Some(cursor) = win.cursor_position() {
                if let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) {
                    // Button region: ±100 x, centred at y=-20
                    if world.x.abs() < 110.0 && (world.y + 20.0).abs() < 28.0 {
                        next.set(GameState::CharacterSelect);
                    }
                }
            }
        }
    }
}
