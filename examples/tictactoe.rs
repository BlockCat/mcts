use std::fmt::Display;

use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;
use rand::prelude::SliceRandom;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    pub fn other(&self) -> Self {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }
}

impl Default for Player {
    fn default() -> Self {
        Player::Player1
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TicTacToeAction {
    x: usize,
    y: usize,
}

#[derive(Debug, Default, PartialEq, Eq, std::hash::Hash, Clone)]
struct TicTacToeState {
    current_player: Player,
    board: [[Option<Player>; 3]; 3],
}

impl Display for TicTacToeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r1 = self.board[0]
            .iter()
            .map(|c| match c {
                Some(Player::Player1) => 'X',
                Some(Player::Player2) => 'O',
                None => ' ',
            })
            .collect::<String>();
        let r2 = self.board[1]
            .iter()
            .map(|c| match c {
                Some(Player::Player1) => 'X',
                Some(Player::Player2) => 'O',
                None => ' ',
            })
            .collect::<String>();
        let r3 = self.board[2]
            .iter()
            .map(|c| match c {
                Some(Player::Player1) => 'X',
                Some(Player::Player2) => 'O',
                None => ' ',
            })
            .collect::<String>();
        f.write_fmt(format_args!("Turn: {:?}\n", self.current_player))?;
        f.write_fmt(format_args!("{}\n", r1))?;
        f.write_fmt(format_args!("{}\n", r2))?;
        f.write_fmt(format_args!("{}\n", r3))?;

        Ok(())
    }
}

impl GameState for TicTacToeState {
    type Move = TicTacToeAction;
    type Player = Player;
    type MoveList = Vec<TicTacToeAction>;

    fn current_player(&self) -> Self::Player {
        self.current_player.clone()
    }

    fn available_moves(&self) -> Self::MoveList {
        if self.is_terminal() {
            Vec::new()
        } else {
            self.board
                .iter()
                .enumerate()
                .map(|(y, row)| {
                    row.iter()
                        .enumerate()
                        .filter_map(move |(x, cell)| match cell {
                            None => Some(TicTacToeAction { x: x, y: y }),
                            _ => None,
                        })
                })
                .flatten()
                .collect()
        }
    }

    fn make_move(&mut self, mov: &Self::Move) -> Result<(), ()> {
        self.board[mov.y][mov.x] = Some(self.current_player());
        self.current_player = self.current_player.other();

        Ok(())
    }

    fn is_terminal(&self) -> bool {
        if let Some(_) = self.get_winner() {
            true
        } else {
            self.board
                .iter()
                .flat_map(|r| r.iter())
                .all(|x| x.is_some())
        }
    }

    fn get_winner(&self) -> Option<Self::Player> {
        for line in &[
            // Rows
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            // Cols
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            // Diags
            [(0, 0), (1, 1), (2, 2)],
            [(2, 0), (1, 1), (0, 2)],
        ] {
            if line
                .into_iter()
                .all(|&(x, y)| self.board[y][x] == Some(Player::Player1))
            {
                return Some(Player::Player1);
            }
            if line
                .into_iter()
                .all(|&(x, y)| self.board[y][x] == Some(Player::Player2))
            {
                return Some(Player::Player2);
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
enum StateEval {
    Winner(Player),
    Draw,
}
struct MyEvaluator(Player);

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = StateEval;

    fn evaluate_new_state(
        &self,
        state: &TicTacToeState,
        moves: &MoveList<MyMCTS>,
        _: Option<SearchHandle<MyMCTS>>,
    ) -> (Vec<MoveEvaluation<MyMCTS>>, Self::StateEvaluation) {
        let mut node = state.clone();
        let mut rand = rand::thread_rng();
        while !node.is_terminal() {
            let moves = node.available_moves();
            let random = moves
                .choose(&mut rand)
                .expect("Could not sample random moves");
            node.make_move(random).expect("Could not");
        }

        let state = match node.get_winner() {
            Some(winner) => StateEval::Winner(winner),
            None => StateEval::Draw,
        };

        (vec![(); moves.len()], state)
    }

    fn evaluate_existing_state(
        &self,
        _state: &TicTacToeState,
        evaln: &StateEval,
        _handle: SearchHandle<MyMCTS>,
    ) -> StateEval {
        evaln.clone()
    }

    fn interpret_evaluation_for_player(
        &self,
        evaln: &StateEval,
        player: &mcts::Player<MyMCTS>,
    ) -> f64 {
        match evaln {
            StateEval::Winner(winner) if winner == player => 1.0,
            StateEval::Winner(_) => -1.0,
            StateEval::Draw => 0.0,
        }
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = TicTacToeState;
    type Eval = MyEvaluator;
    type TreePolicy = UCTPolicy<()>;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::PanicWhenCycleDetected
    }
}

fn main() {
    play_game(find_mcts_action, find_mcts_action);
}

fn play_game<F>(player_1: F, player_2: F)
where
    F: Fn(&TicTacToeState) -> TicTacToeAction,
{
    let mut game = TicTacToeState::default();
    println!("{}", game);

    while !game.is_terminal() {
        let action = player_1(&game);
        game.make_move(&action).expect("Could not make move");
        println!("{}", game);
        if game.is_terminal() {
            break;
        }

        let action = player_2(&game);
        game.make_move(&action).expect("Could not make move");
        println!("{}", game);
    }
}

fn find_mcts_action(game: &TicTacToeState) -> TicTacToeAction {
    let mut mcts = MCTSManager::new(
        game.clone(),
        MyMCTS,
        MyEvaluator(game.current_player()),
        UCTPolicy::new(4.4),
        ApproxTable::new(1024),
    );
    mcts.playout_n(100_000);

    mcts.tree().debug_moves();

    let resulting_action = mcts.principal_variation(1);
    resulting_action
        .first()
        .expect("Could not find action")
        .clone()
}

// fn find_player_action(game: &TicTacToeState) -> TicTacToeAction {
//     use std::io;

//     let find = || {
//         let mut buffer = String::new();
//         println!("Please enter coordinations: x,y");

//         io::stdin().read_line(&mut buffer).unwrap();

//         let x: usize = buffer[0..1].parse().ok()?;
//         let y: usize = buffer[2..3].parse().ok()?;

//         Some(TicTacToeAction { x: x, y: y })
//     };
//     loop {
//         match find() {
//             Some(action) => {
//                 break action;
//             }
//             None => {}
//         }
//     }
// }
