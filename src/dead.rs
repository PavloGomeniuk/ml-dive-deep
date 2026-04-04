use bevy::prelude::*;
use crate::state::GameState;
use crate::components::{Z_OVERLAY, SCREEN_W, SCREEN_H};
use crate::resources::RunStats;

pub struct DeadPlugin;
impl Plugin for DeadPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::Dead), setup_dead)
           .add_systems(Update, handle_dead_input.run_if(in_state(GameState::Dead)));
    }
}

fn setup_dead(mut commands: Commands, run_stats: Res<RunStats>) {
    // Dark overlay
    commands.spawn((
        Sprite {
            color: Color::srgba(0.0, 0.0, 0.0, 0.75),
            custom_size: Some(Vec2::new(SCREEN_W, SCREEN_H)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, Z_OVERLAY),
        StateScoped(GameState::Dead),
    ));

    // YOU DIED
    commands.spawn((
        Text2d::new("YOU DIED"),
        TextFont { font_size: 56.0, ..default() },
        TextColor(Color::srgb(0.800, 0.133, 0.133)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, 60.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Dead),
    ));

    // Stats
    let stats_text = format!(
        "Enemies slain: {}    Rooms cleared: {}    Gold: {}",
        run_stats.kills, run_stats.rooms_cleared, run_stats.gold_earned
    );
    commands.spawn((
        Text2d::new(stats_text),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::srgb(0.533, 0.533, 0.533)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, 15.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Dead),
    ));

    // Restart hint
    commands.spawn((
        Text2d::new("click or press ENTER to return to title"),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::srgb(0.533, 0.533, 0.533)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, -30.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::Dead),
    ));
}

fn handle_dead_input(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut next: ResMut<NextState<GameState>>,
) {
    if mouse.just_pressed(MouseButton::Left)
        || keys.just_pressed(KeyCode::Enter)
        || keys.just_pressed(KeyCode::Space)
    {
        next.set(GameState::Title);
    }
}
