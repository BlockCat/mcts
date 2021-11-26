extern crate rand;
use self::rand::{prelude::ThreadRng, Rng, SeedableRng};

use super::*;
use rand::prelude::StdRng;
use search_tree::*;
use std::{self, marker::PhantomData};

pub trait TreePolicy<Spec: MCTS<TreePolicy = Self>>: Sync + Sized {
    type MoveEvaluation: Sync + Send;
    type ThreadLocalData: Default + SelectionRng;

    fn choose_child<'a, MoveIter>(
        &self,
        moves: MoveIter,
        handle: SearchHandle<Spec>,
    ) -> &'a MoveInfo<Spec>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<Spec>> + Clone;
    fn validate_evaluations(&self, _evalns: &[Self::MoveEvaluation]) {}
}

#[derive(Clone, Debug)]
pub struct UCTPolicy<MV> {
    exploration_constant: f64,
    _phantom: PhantomData<MV>,
}

impl<MV> UCTPolicy<MV> {
    pub fn new(exploration_constant: f64) -> Self {
        assert!(
            exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant
        );
        Self {
            exploration_constant,
            _phantom: Default::default(),
        }
    }

    pub fn exploration_constant(&self) -> f64 {
        self.exploration_constant
    }
}

const RECIPROCAL_TABLE_LEN: usize = 128;

#[derive(Clone, Debug)]
pub struct AlphaGoPolicy {
    exploration_constant: f64,
    reciprocals: Vec<f64>,
}

impl AlphaGoPolicy {
    pub fn new(exploration_constant: f64) -> Self {
        assert!(
            exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant
        );
        let reciprocals = (0..RECIPROCAL_TABLE_LEN)
            .map(|x| if x == 0 { 2.0 } else { 1.0 / x as f64 })
            .collect();
        Self {
            exploration_constant,
            reciprocals,
        }
    }

    pub fn exploration_constant(&self) -> f64 {
        self.exploration_constant
    }

    fn reciprocal(&self, x: usize) -> f64 {
        if x < RECIPROCAL_TABLE_LEN {
            unsafe { *self.reciprocals.get_unchecked(x) }
        } else {
            1.0 / x as f64
        }
    }
}

impl<Spec: MCTS<TreePolicy = Self>, MV: Send + Sync> TreePolicy<Spec> for UCTPolicy<MV> {
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = MV;

    fn choose_child<'a, MoveIter>(
        &self,
        moves: MoveIter,
        mut handle: SearchHandle<Spec>,
    ) -> &'a MoveInfo<Spec>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<Spec>> + Clone,
    {
        let parent_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
        handle
            .thread_data()
            .policy_data
            .select_by_key(moves, |mov| {
                let sum_rewards = mov.sum_rewards();
                let child_visits = mov.visits();
                // http://mcts.ai/pubs/mcts-survey-master.pdf
                if child_visits == 0 {
                    std::f64::INFINITY
                } else {
                    let parent_visits = parent_visits as f64;
                    let child_visits = child_visits as f64;
                    let explore_term = (parent_visits.ln() / child_visits).sqrt();
                    let mean_action_value = sum_rewards as f64 / child_visits;
                    self.exploration_constant * explore_term + mean_action_value
                }
            })
            .unwrap()
    }
}

impl<Spec: MCTS<TreePolicy = Self>> TreePolicy<Spec> for AlphaGoPolicy {
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = f64;

    fn choose_child<'a, MoveIter>(
        &self,
        moves: MoveIter,
        mut handle: SearchHandle<Spec>,
    ) -> &'a MoveInfo<Spec>
    where
        MoveIter: Iterator<Item = &'a MoveInfo<Spec>> + Clone,
    {
        let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>() + 1;
        let sqrt_total_visits = (total_visits as f64).sqrt();
        let explore_coef = self.exploration_constant * sqrt_total_visits;

        handle
            .thread_data()
            .policy_data
            .select_by_key(moves, |mov| {
                let sum_rewards = mov.sum_rewards() as f64;
                let child_visits = mov.visits();
                let policy_evaln = *mov.move_evaluation() as f64;

                (sum_rewards + explore_coef * policy_evaln) * self.reciprocal(child_visits as usize)
            })
            .unwrap()
    }

    fn validate_evaluations(&self, evalns: &[f64]) {
        for &x in evalns {
            assert!(
                x >= -1e-6,
                "Move evaluation is {} (must be non-negative)",
                x
            );
        }
        if evalns.len() >= 1 {
            let evaln_sum: f64 = evalns.iter().sum();
            assert!(
                (evaln_sum - 1.0).abs() < 0.1,
                "Sum of evaluations is {} (should sum to 1)",
                evaln_sum
            );
        }
    }
}

pub trait SelectionRng {
    fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, key_fn: KeyFn) -> Option<T>
    where
        Iter: Iterator<Item = T>,
        KeyFn: Fn(&T) -> f64,
        T: Clone;
}

#[derive(Clone)]
pub struct WeightedRng {
    rng: StdRng,
}

#[derive(Clone)]
pub struct PolicyRng {
    rng: StdRng,
}

impl PolicyRng {
    pub fn new(seed: u64) -> Self {
        let rng = SeedableRng::seed_from_u64(seed);
        Self { rng }
    }
}

impl WeightedRng {
    pub fn new(seed: u64) -> Self {
        let rng = SeedableRng::seed_from_u64(seed);
        Self { rng }
    }
}

impl SelectionRng for PolicyRng {
    fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, key_fn: KeyFn) -> Option<T>
    where
        Iter: Iterator<Item = T>,
        KeyFn: Fn(&T) -> f64,
        T: Clone,
    {
        let mut choice = None;
        let mut num_optimal: u32 = 0;
        let mut best_so_far: f64 = std::f64::NEG_INFINITY;
        for elt in elts {
            let score = key_fn(&elt);
            if score > best_so_far {
                choice = Some(elt);
                num_optimal = 1;
                best_so_far = score;
            } else if score == best_so_far {
                num_optimal += 1;
                if self.rng.gen_bool(1.0 / (num_optimal as f64)) {
                    choice = Some(elt);
                }
            }
        }
        choice
    }
}

impl SelectionRng for WeightedRng {
    fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, key_fn: KeyFn) -> Option<T>
    where
        Iter: Iterator<Item = T>,
        KeyFn: Fn(&T) -> f64,
        T: Clone,
    {
        use rand::seq::SliceRandom;

        let options = elts.collect::<Vec<_>>();

        let minimal = options
            .iter()
            .map(&key_fn)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let minimal = if minimal < 0.0 { -minimal } else { 0.01 };

        options
            .choose_weighted(&mut self.rng, |v| key_fn(v) + minimal)
            .ok()
            .or_else(|| {
                println!(
                    "No weighted found, {} moves found, choosing random. {:?}",
                    options.len(),
                    options.iter().map(&key_fn).collect::<Vec<_>>()
                );
                options.choose(&mut self.rng)
            })
            .cloned()
    }
}

impl Default for WeightedRng {
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl Default for PolicyRng {
    fn default() -> Self {
        Self::new(rand::random())
    }
}
