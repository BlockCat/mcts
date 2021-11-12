use std::fmt::Display;

use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;

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

impl TicTacToeState {
    pub fn is_terminal(&self) -> bool {
        if let Some(_) = self.get_winner() {
            true
        } else {
            self.board
                .iter()
                .flat_map(|r| r.iter())
                .all(|x| x.is_some())
        }
    }

    pub fn get_winner(&self) -> Option<Player> {
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

impl GameState for TicTacToeState {
    type Move = TicTacToeAction;
    type Player = Player;
    type MoveList = Vec<TicTacToeAction>;

    fn current_player(&self) -> Self::Player {
        self.current_player.clone()
    }

    fn available_moves(&self) -> Self::MoveList {
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

    fn make_move(&mut self, mov: &Self::Move) {
        self.board[mov.y][mov.x] = Some(self.current_player());
        self.current_player = self.current_player.other();
    }
}

struct MyEvaluator(Player);

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = f64;

    fn evaluate_new_state(
        &self,
        state: &TicTacToeState,
        moves: &MoveList<MyMCTS>,
        _: Option<SearchHandle<MyMCTS>>,
    ) -> (Vec<MoveEvaluation<MyMCTS>>, Self::StateEvaluation) {
        let progression = 9 - moves.len();
        if state.is_terminal() {
            if let Some(winner) = state.get_winner() {
                if winner == self.0 {
                    (vec![(); moves.len()], 2.0)
                } else {
                    (vec![(); moves.len()], -2.0)
                }
            } else {
                (vec![(); moves.len()], 1.0)
            }
        } else {
            (vec![(); moves.len()], 0.0)
        }
    }

    fn evaluate_existing_state(
        &self,
        _state: &TicTacToeState,
        evaln: &f64,
        _handle: SearchHandle<MyMCTS>,
    ) -> f64 {
        *evaln
    }

    fn interpret_evaluation_for_player(&self, evaln: &f64, player: &mcts::Player<MyMCTS>) -> i64 {
        let eval = (*evaln * 100f64) as i64;

        if player == &self.0 {
            eval
        } else {
            -eval
        }
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = TicTacToeState;
    type Eval = MyEvaluator;
    type TreePolicy = UCTPolicy;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();
}

fn main() {
    let game = TicTacToeState::default();
    let mut mcts = MCTSManager::new(
        game,
        MyMCTS,
        MyEvaluator(Player::Player1),
        UCTPolicy::new(5.0),
        ApproxTable::new(1024),
    );
    mcts.playout_n(10_000);
    let pv: Vec<_> = mcts.principal_variation_states(10);
    println!("Principal variation: {:?}", pv);
    println!("Evaluation of moves:");
    mcts.tree().debug_moves();

    play_game();
}

fn play_game() {
    let mut game = TicTacToeState::default();

    println!("{}", game);

    while !game.is_terminal() {
        let action = find_mcts_action(&game);
        game.make_move(&action);
        println!("{}", game);
        if game.is_terminal() {
            break;
        }

        let action = find_mcts_action(&game);
        // let action = loop {
        //     match find_player_action(&game) {
        //         Some(action) => {
        //             break action;
        //         }
        //         None => {}
        //     }
        // };

        game.make_move(&action);
        println!("{}", game);
    }
}

fn find_mcts_action(game: &TicTacToeState) -> TicTacToeAction {
    let mut mcts = MCTSManager::new(
        game.clone(),
        MyMCTS,
        MyEvaluator(game.current_player()),
        UCTPolicy::new(1.4),
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

fn find_player_action(game: &TicTacToeState) -> Option<TicTacToeAction> {
    use std::io;

    let mut buffer = String::new();
    println!("Please enter coordinations: x,y");

    io::stdin().read_line(&mut buffer).unwrap();

    let x: usize = buffer[0..1].parse().ok()?;
    let y: usize = buffer[2..3].parse().ok()?;

    Some(TicTacToeAction { x: x, y: y })
}
