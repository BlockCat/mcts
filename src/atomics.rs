use std;

#[cfg(not(any(target_pointer_width = "64", feature = "nightly")))]
compile_error!("If you aren't compiling for 64-bit, you must use the nightly compiler.");

pub type AtomicF64 = atomic_float::AtomicF64;

#[cfg(target_pointer_width = "64")]
pub type AtomicU64 = std::sync::atomic::AtomicUsize;
#[cfg(not(target_pointer_width = "64"))]
pub type AtomicU64 = std::sync::atomic::AtomicU64;

#[cfg(target_pointer_width = "64")]
pub type FakeU64 = usize;
#[cfg(not(target_pointer_width = "64"))]
pub type FakeU64 = u64;

pub type AtomicPtr<T> = std::sync::atomic::AtomicPtr<T>;
pub type AtomicBool = std::sync::atomic::AtomicBool;
pub type AtomicIsize = std::sync::atomic::AtomicIsize;
pub type AtomicUsize = std::sync::atomic::AtomicUsize;
pub use std::sync::atomic::Ordering;
