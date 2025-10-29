use core::sync::atomic::{AtomicBool, Ordering};
use lock_api::{GuardSend, RawMutex};

pub struct RawSpinlock(AtomicBool);

unsafe impl RawMutex for RawSpinlock {
    const INIT: RawSpinlock = RawSpinlock(AtomicBool::new(false));
    type GuardMarker = GuardSend;

    fn lock(&self) {
        while !self.try_lock() {
            // Hint to CPU that we're spinning - reduces power consumption
            // and improves performance on hyperthreaded CPUs
            core::hint::spin_loop();
        }
    }

    fn try_lock(&self) -> bool {
        self.0
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    unsafe fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}
