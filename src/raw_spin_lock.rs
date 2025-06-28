use lock_api::{RawMutex, Mutex, GuardSend};
use core::sync::atomic::{AtomicBool, Ordering};

pub struct RawSpinlock(AtomicBool);

unsafe impl RawMutex for RawSpinlock {
    const INIT: RawSpinlock = RawSpinlock(AtomicBool::new(false));
    type GuardMarker = GuardSend;

    fn lock(&self) {
        while !self.try_lock() {}
    }

    fn try_lock(&self) -> bool {
        self.0.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
    }

    unsafe fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}