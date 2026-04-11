use shared::deck::Card;
use shared::durak::DurakGame;
use shared::messages::GameAction;
use shared::poker::{TexasPokerGame, evaluate_hand};

// ── Poker bot ─────────────────────────────────────────────────────────────────

pub enum PokerBotAction {
    Fold,
    Check,
    Call,
    Raise(u32),
}

/// Simple poker bot: pre-flop uses hole card strength; post-flop evaluates best hand.
pub fn poker_move(game: &TexasPokerGame) -> PokerBotAction {
    let bot_id = match game.action_player_id() {
        Some(id) => id,
        None => return PokerBotAction::Check,
    };

    let player = match game.players.iter().find(|p| p.id == bot_id) {
        Some(p) => p,
        None => return PokerBotAction::Check,
    };

    let community = &game.community_cards;
    let call_amount = game.current_bet.saturating_sub(player.bet);
    let chips = player.chips;

    // Evaluate hand strength (0.0 = worst, 1.0 = best)
    let strength = if community.is_empty() {
        pre_flop_strength(&player.hole_cards)
    } else {
        let mut all_cards = player.hole_cards.to_vec();
        all_cards.extend_from_slice(community);
        post_flop_strength(&player.hole_cards, community)
    };

    // Decision thresholds
    if strength > 0.75 {
        // Strong hand: raise
        let raise_amount = (game.big_blind * 3).min(chips / 4).max(game.big_blind);
        PokerBotAction::Raise(game.current_bet + raise_amount)
    } else if strength > 0.45 {
        // Medium hand: call or check
        if call_amount == 0 {
            PokerBotAction::Check
        } else if call_amount <= chips / 3 {
            PokerBotAction::Call
        } else {
            PokerBotAction::Fold
        }
    } else {
        // Weak hand: check if free, else fold
        if call_amount == 0 {
            PokerBotAction::Check
        } else {
            PokerBotAction::Fold
        }
    }
}

/// Pre-flop hand strength heuristic based on hole cards.
fn pre_flop_strength(hole: &[Card; 2]) -> f64 {
    let [a, b] = hole;
    let av = a.rank.value() as f64;
    let bv = b.rank.value() as f64;
    let high = av.max(bv);
    let low = av.min(bv);
    let suited = a.suit == b.suit;
    let pair = a.rank == b.rank;
    let gap = (high - low) as f64;

    // Pair: strong, especially high pairs
    if pair {
        return 0.5 + (high - 2.0) / 24.0; // 0.50 (22) to 0.96 (AA)
    }

    // Base from high card
    let mut score = (high - 2.0) / 12.0 * 0.5; // 0..0.5
    score += (low - 2.0) / 12.0 * 0.2;
    if suited { score += 0.08; }
    if gap <= 1.0 { score += 0.05; } // connectors

    score.min(0.95).max(0.05)
}

/// Post-flop strength: rank the best 5-card hand.
fn post_flop_strength(hole: &[Card; 2], community: &[Card]) -> f64 {
    let (rank, _) = evaluate_hand(hole, community);
    use shared::poker::HandRank;
    match rank {
        HandRank::HighCard => 0.10,
        HandRank::OnePair => 0.30,
        HandRank::TwoPair => 0.50,
        HandRank::ThreeOfAKind => 0.65,
        HandRank::Straight => 0.72,
        HandRank::Flush => 0.78,
        HandRank::FullHouse => 0.88,
        HandRank::FourOfAKind => 0.95,
        HandRank::StraightFlush => 0.98,
        HandRank::RoyalFlush => 1.00,
    }
}

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
