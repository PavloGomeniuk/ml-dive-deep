use bevy::prelude::*;
use crate::state::GameState;
use crate::components::{Z_OVERLAY, SCREEN_W, SCREEN_H};
use crate::resources::{SelectedClass, CharSelectHighlight};
use crate::entities::PlayerClass;

pub struct CharSelectPlugin;
impl Plugin for CharSelectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::CharacterSelect), setup_char_select)
           .add_systems(
               Update,
               (handle_char_select_input, update_card_highlights)
                   .run_if(in_state(GameState::CharacterSelect)),
           );
    }
}

#[derive(Component)]
struct ClassCard(usize); // 0=Warrior, 1=Magician

#[derive(Component)]
struct CardBorder(usize);

fn card_x(idx: usize) -> f32 {
    match idx {
        0 => -160.0,
        _ => 160.0,
    }
}

fn setup_char_select(mut commands: Commands, mut highlight: ResMut<CharSelectHighlight>) {
    highlight.0 = None;

    // Background
    commands.spawn((
        Sprite {
            color: Color::srgb(0.051, 0.051, 0.102),
            custom_size: Some(Vec2::new(SCREEN_W, SCREEN_H)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, Z_OVERLAY),
        StateScoped(GameState::CharacterSelect),
    ));

    // Header
    commands.spawn((
        Text2d::new("CHOOSE YOUR FATE"),
        TextFont { font_size: 22.0, ..default() },
        TextColor(Color::srgb(0.831, 0.686, 0.216)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, 170.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::CharacterSelect),
    ));

    // Divider
    commands.spawn((
        Sprite {
            color: Color::srgb(0.353, 0.227, 0.082),
            custom_size: Some(Vec2::new(500.0, 1.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 148.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::CharacterSelect),
    ));

    let classes = [
        (0usize, "WARRIOR", "150 HP  ·  30 MP", "Cleave — AoE melee, 80px, 4s cd", Color::srgb(0.53, 0.47, 0.43)),
        (1usize, "MAGICIAN", "90 HP  ·  100 MP", "Frost Nova — freeze, 70px, 3s cd", Color::srgb(0.60, 0.53, 0.80)),
    ];

    for (idx, class_name, stats, ability, body_color) in classes {
        let x = card_x(idx);

        // Card background
        commands.spawn((
            Sprite {
                color: Color::srgb(0.059, 0.059, 0.118),
                custom_size: Some(Vec2::new(260.0, 290.0)),
                ..default()
            },
            Transform::from_xyz(x, 10.0, Z_OVERLAY + 0.5),
            StateScoped(GameState::CharacterSelect),
        ));

        // Card border (updated by highlight system)
        commands.spawn((
            Sprite {
                color: Color::srgb(0.353, 0.227, 0.082),
                custom_size: Some(Vec2::new(262.0, 292.0)),
                ..default()
            },
            Transform::from_xyz(x, 10.0, Z_OVERLAY + 0.4),
            CardBorder(idx),
            ClassCard(idx),
            StateScoped(GameState::CharacterSelect),
        ));

        // Sprite preview (coloured silhouette)
        commands.spawn((
            Sprite {
                color: body_color,
                custom_size: Some(Vec2::new(40.0, 70.0)),
                ..default()
            },
            Transform::from_xyz(x, 95.0, Z_OVERLAY + 0.6),
            StateScoped(GameState::CharacterSelect),
        ));

        // Class name
        commands.spawn((
            Text2d::new(class_name),
            TextFont { font_size: 18.0, ..default() },
            TextColor(Color::srgb(0.831, 0.686, 0.216)),
            TextLayout::new_with_justify(JustifyText::Center),
            Transform::from_xyz(x, 40.0, Z_OVERLAY + 0.6),
            StateScoped(GameState::CharacterSelect),
        ));

        // Stats
        commands.spawn((
            Text2d::new(stats),
            TextFont { font_size: 12.0, ..default() },
            TextColor(Color::srgb(0.533, 0.533, 0.667)),
            TextLayout::new_with_justify(JustifyText::Center),
            Transform::from_xyz(x, 15.0, Z_OVERLAY + 0.6),
            StateScoped(GameState::CharacterSelect),
        ));

        // Ability box background
        commands.spawn((
            Sprite {
                color: Color::srgb(0.227, 0.227, 0.353),
                custom_size: Some(Vec2::new(230.0, 36.0)),
                ..default()
            },
            Transform::from_xyz(x, -30.0, Z_OVERLAY + 0.6),
            StateScoped(GameState::CharacterSelect),
        ));

        // Ability text
        commands.spawn((
            Text2d::new(ability),
            TextFont { font_size: 11.0, ..default() },
            TextColor(Color::srgb(0.831, 0.686, 0.216)),
            TextLayout::new_with_justify(JustifyText::Center),
            Transform::from_xyz(x, -30.0, Z_OVERLAY + 0.7),
            StateScoped(GameState::CharacterSelect),
        ));
    }

    // Hint bar
    commands.spawn((
        Text2d::new("click to select  ·  ENTER to begin  ·  ESC for title"),
        TextFont { font_size: 12.0, ..default() },
        TextColor(Color::srgb(0.35, 0.35, 0.45)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, -170.0, Z_OVERLAY + 0.5),
        StateScoped(GameState::CharacterSelect),
    ));
}

fn handle_char_select_input(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut selected: ResMut<SelectedClass>,
    mut highlight: ResMut<CharSelectHighlight>,
    mut next: ResMut<NextState<GameState>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        next.set(GameState::Title);
        return;
    }

    if keys.just_pressed(KeyCode::Enter) {
        if let Some(idx) = highlight.0 {
            selected.0 = Some(if idx == 0 { PlayerClass::Warrior } else { PlayerClass::Magician });
            next.set(GameState::Playing);
        }
        return;
    }

    if mouse.just_pressed(MouseButton::Left) {
        if let (Ok(win), Ok((cam, cam_tf))) = (windows.get_single(), camera_q.get_single()) {
            if let Some(cursor) = win.cursor_position() {
                if let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) {
                    for idx in 0..2usize {
                        let cx = card_x(idx);
                        if (world.x - cx).abs() < 131.0 && (world.y - 10.0).abs() < 146.0 {
                            if highlight.0 == Some(idx) {
                                // Second click = confirm
                                selected.0 = Some(if idx == 0 { PlayerClass::Warrior } else { PlayerClass::Magician });
                                next.set(GameState::Playing);
                            } else {
                                highlight.0 = Some(idx);
                            }
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn update_card_highlights(
    highlight: Res<CharSelectHighlight>,
    mut borders: Query<(&CardBorder, &mut Sprite)>,
) {
    if !highlight.is_changed() { return; }
    for (border, mut sprite) in &mut borders {
        sprite.color = if highlight.0 == Some(border.0) {
            Color::srgb(0.831, 0.686, 0.216) // gold
        } else {
            Color::srgb(0.353, 0.227, 0.082) // bronze
        };
    }
}
