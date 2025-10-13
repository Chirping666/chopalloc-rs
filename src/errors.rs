use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuddyAllocatorError {
    /// Bitmap storage is too small
    BitmapStorageTooSmall {
        required_words: usize,
        provided_words: usize,
    },
    /// Requested allocation size is too large
    AllocationTooLarge {
        requested_bytes: usize,
        max_block_size: usize,
    },
    /// Out of memory - no free blocks available
    OutOfMemory {
        requested_order: usize,
        largest_available_order: Option<usize>,
    },
    /// Invalid deallocation - pointer not within managed region
    InvalidPointer {
        ptr: NonNull<u8>,
        base_addr: NonNull<u8>,
        region_size: usize,
    },
    /// Invalid deallocation - block not properly aligned
    InvalidAlignment {
        ptr: NonNull<u8>,
        block_size: usize,
        required_alignment: usize,
    },
    /// Invalid deallocation - block already free (double-free)
    DoubleFree { ptr: NonNull<u8>, order: usize },
    /// Memory region parameters are invalid
    InvalidMemoryRegion {
        base_addr: NonNull<u8>,
        size: usize,
        reason: &'static str,
    },
    /// Invalid layout parameters
    InvalidLayout {
        size: usize,
        align: usize,
        reason: &'static str,
    },
}

impl core::fmt::Display for BuddyAllocatorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BitmapStorageTooSmall {
                required_words,
                provided_words,
            } => {
                write!(
                    f,
                    "Bitmap storage too small: need {} words, got {}",
                    required_words, provided_words
                )
            }
            Self::OutOfMemory {
                requested_order,
                largest_available_order,
            } => match largest_available_order {
                Some(largest) => write!(
                    f,
                    "Out of memory: requested 2^{} bytes, largest available 2^{}",
                    requested_order, largest
                ),
                None => write!(f, "Out of memory: no free blocks available"),
            },
            _ => {
                write!(f, "Buddy Allocator Error: {:?}", self)
            } // ... other variants
        }
    }
}

pub type Result<T> = core::result::Result<T, BuddyAllocatorError>;
