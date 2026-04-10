use shared::deck::Card;
use shared::durak::DurakGame;
use shared::messages::GameAction;

/// Return the bot's next Durak move. Returns None if no valid move found.
pub fn durak_move(game: &DurakGame, bot_idx: usize) -> Option<GameAction> {
    use shared::durak::DurakState;

    match &game.state {
        DurakState::PlayerAttacks if game.attacker == bot_idx => {
            // Attack with lowest non-trump card, or lowest trump if no other choice
            let hand = &game.hands[bot_idx];
            if hand.is_empty() {
                return None;
            }

            // If table is non-empty, must match an existing rank
            let card = if game.table.is_empty() {
                lowest_non_trump(hand, game.trump).or_else(|| lowest_card(hand, game.trump))
            } else {
                // Find a card that matches a rank already on the table
                let valid: Vec<Card> = hand
                    .iter()
                    .copied()
                    .filter(|c| rank_on_table(game, c.rank))
                    .collect();
                if valid.is_empty() {
                    // Bot can't attack further — end attack
                    return Some(GameAction::DurakEndAttack);
                }
                lowest_non_trump(&valid, game.trump).or_else(|| lowest_card(&valid, game.trump))
            };

            card.map(|c| GameAction::DurakAttack { card: c })
        }

        DurakState::PlayerDefends if game.defender() == bot_idx => {
            // Try to defend each undefended card
            let hand = &game.hands[bot_idx];
            let undefended: Vec<Card> = game
                .table
                .iter()
                .filter_map(|(atk, def)| if def.is_none() { Some(*atk) } else { None })
                .collect();

            if undefended.is_empty() {
                return None;
            }

            let attack = undefended[0]; // defend one at a time

            // Find cheapest card that beats attack
            let defend = cheapest_defense(hand, attack, game.trump);
            if let Some(dc) = defend {
                Some(GameAction::DurakDefend { attack_card: attack, defend_card: dc })
            } else {
                // Can't defend — take cards
                Some(GameAction::DurakTakeCards)
            }
        }

        DurakState::PlayerAttacks if game.attacker != bot_idx => {
            // All table cards defended — end attack
            if game.table.iter().all(|(_, d)| d.is_some()) && !game.table.is_empty() {
                Some(GameAction::DurakEndAttack)
            } else {
                None
            }
        }

        _ => None,
    }
}

fn rank_on_table(game: &DurakGame, rank: shared::deck::Rank) -> bool {
    game.table.iter().any(|(atk, def)| {
        atk.rank == rank || def.as_ref().map_or(false, |d| d.rank == rank)
    })
}

fn lowest_non_trump(hand: &[Card], trump: shared::deck::Suit) -> Option<Card> {
    hand.iter()
        .filter(|c| c.suit != trump)
        .min_by_key(|c| c.rank.value())
        .copied()
}

fn lowest_card(hand: &[Card], trump: shared::deck::Suit) -> Option<Card> {
    // Prefer non-trump, then trump
    hand.iter()
        .min_by_key(|c| {
            let base = c.rank.value() as u16;
            if c.suit == trump { base + 100 } else { base }
        })
        .copied()
}

/// Find the cheapest card in `hand` that beats `attack`.
fn cheapest_defense(hand: &[Card], attack: Card, trump: shared::deck::Suit) -> Option<Card> {
    hand.iter()
        .filter(|c| beats(attack, **c, trump))
        .min_by_key(|c| {
            let base = c.rank.value() as u16;
            if c.suit == trump { base + 100 } else { base }
        })
        .copied()
}

fn beats(attack: Card, defend: Card, trump: shared::deck::Suit) -> bool {
    if attack.suit == defend.suit {
        defend.rank.value() > attack.rank.value()
    } else {
        defend.suit == trump && attack.suit != trump
    }
}
