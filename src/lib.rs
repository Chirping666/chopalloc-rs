#![no_std]
mod raw_spin_lock;
#[cfg(test)]
mod tests;
mod errors;

use core::ptr::NonNull;
use core::alloc::Layout;
// use anyhow::Result;
use lock_api::Mutex;
use raw_spin_lock::*;
use errors::*;

// Free block node stored directly in free memory
#[repr(C)]
struct FreeBlock {
    next: Option<NonNull<FreeBlock>>,
}

// Inner allocator state protected by mutex
struct BuddyAllocatorInner<const MAX_ORDER: usize> {
    free_lists: [Option<NonNull<FreeBlock>>; MAX_ORDER],
    // TODO: Add bitmap storage here
    bitmap_storage: &'static mut [u64],
    bitmap_offsets: [usize; MAX_ORDER],
}

// Main allocator structure
pub struct BuddyAllocator<const MAX_ORDER: usize> {
    inner: Mutex<RawSpinlock, BuddyAllocatorInner<MAX_ORDER>>,
    base_addr: NonNull<u8>,
    total_size: usize,
}

impl<const MAX_ORDER: usize> BuddyAllocator<MAX_ORDER> {
    pub fn new(
        memory_region: NonNull<u8>,
        memory_size: usize,
        bitmap_storage: &'static mut [u64]
    ) -> Result<Self> {
        let mut bitmap_offsets = [0; MAX_ORDER];
        let mut current_offset = 0;

        // Calculate the starting offset for each order's bitmap
        for order in 0..MAX_ORDER {
            // This order's bitmap starts at the current offset
            bitmap_offsets[order] = current_offset;

            // Calculate how many words this order needs
            let block_size = 1 << order;
            let blocks_count = memory_size / block_size;
            let words_for_this_order = (blocks_count + 63) / 64;

            // Next order starts after this order's words
            current_offset += words_for_this_order;
        }

        // Validate total storage
        let total_words_needed = current_offset;
        if bitmap_storage.len() < total_words_needed {
            return Err(BuddyAllocatorError::BitmapStorageTooSmall {
                required_words: total_words_needed,
                provided_words: bitmap_storage.len(),
            });
        }

        // Initialize all bits to 0 (allocated)
        bitmap_storage.fill(0);

        Ok(Self {
            inner: Mutex::new(BuddyAllocatorInner {
                free_lists: [None; MAX_ORDER],
                bitmap_storage,
                bitmap_offsets,
            }),
            base_addr: memory_region,
            total_size: memory_size,
        })
    }

    pub fn calculate_bitmap_words_needed(memory_size: usize) -> usize {
        let mut total_words = 0;

        for order in 0..MAX_ORDER {

            let block_size = 1 << order; // 2^order bytes per block

            // How many blocks of this size fit in our memory?
            let blocks_count = memory_size / block_size;

            // How many u64 words do we need to store this many bits?
            // Each u64 can store 64 bits, so we round up
            let words_for_this_order = (blocks_count + 63) / 64;

            total_words += words_for_this_order;
        }

        total_words
    }

    /// Try to allocate memory matching the given layout
    pub fn try_allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let mut guard = self.inner.lock();

        let size = layout.size().max(layout.align()).next_power_of_two();
        let order = size.trailing_zeros() as usize;

        // Try to find a block of the exact size
        if let Some(block) = Self::pop_free_block(&mut guard.free_lists, order) {
            return Ok(block);
        }

        // Find a larger block and split it down
        for larger_order in (order + 1)..MAX_ORDER {
            if let Some(large_block) = Self::pop_free_block(&mut guard.free_lists, larger_order) {
                return Ok(Self::split_block_down_to(&mut guard.free_lists, large_block, larger_order, order));
            }
        }

        todo!()
    }

    /// Try to deallocate memory at the given pointer
    pub fn try_deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        let _guard = self.inner.lock();

        let size = layout.size().max(layout.align()).next_power_of_two();
        let order = size.trailing_zeros() as usize;

        // TODO: Implement deallocation with buddy merging
        // 1. Mark block as free
        // 2. Find buddy using XOR
        // 3. If buddy is free, merge and repeat
        // 4. Add final block to appropriate free list

        Ok(())
    }

    /// Remove and return a block from the free list of given order
    fn pop_free_block(
        free_lists: &mut [Option<NonNull<FreeBlock>>; MAX_ORDER],
        order: usize
    ) -> Option<NonNull<u8>> {
        if order >= MAX_ORDER {
            return None;
        }

        // Get the head of the free list
        let head = free_lists[order].take()?;

        unsafe {
            // Update free list to point to next block
            let free_block = head.cast::<FreeBlock>().as_ref();
            free_lists[order] = free_block.next;
        }

        Some(head.cast::<u8>())
    }

    /// Add a block to the free list of given order
    fn push_free_block(
        free_lists: &mut [Option<NonNull<FreeBlock>>; MAX_ORDER],
        ptr: NonNull<u8>,
        order: usize
    ) {
        if order >= MAX_ORDER {
            return;
        }

        unsafe {
            // Convert the memory into a FreeBlock node
            let free_block = ptr.cast::<FreeBlock>().as_mut();
            free_block.next = free_lists[order];
            free_lists[order] = Some(ptr.cast::<FreeBlock>());
        }
    }

    /// Split a large block down to the target order
    fn split_block_down_to(
        free_lists: &mut [Option<NonNull<FreeBlock>>; MAX_ORDER],
        mut block: NonNull<u8>,
        from_order: usize,
        to_order: usize
    ) -> NonNull<u8> {
        let mut current_order = from_order;
        let current_block = block;

        // Split repeatedly until we reach the target size
        while current_order > to_order {
            current_order -= 1;
            let block_size = 1 << current_order; // 2^current_order

            // Calculate buddy address using XOR magic!
            let buddy_addr = current_block.as_ptr() as usize ^ block_size;
            let buddy_ptr = NonNull::new(buddy_addr as *mut u8).unwrap();

            // Put the buddy on the appropriate free list
            Self::push_free_block(free_lists, buddy_ptr, current_order);

            // Keep the first half for further splitting
        }

        current_block
    }

    /// Convert address to bit index for a given order
    fn get_bit_index(&self, addr: NonNull<u8>, order: usize) -> usize {
        let offset_from_base = addr.as_ptr() as usize - self.base_addr.as_ptr() as usize;
        let block_size = 1 << order; // 2^order
        offset_from_base / block_size
    }

    // TODO: Bitmap management functions
    // fn is_block_free(&self, bitmaps: &[???], addr: NonNull<u8>, order: usize) -> bool
    // fn set_block_free(&self, bitmaps: &mut [???], addr: NonNull<u8>, order: usize, is_free: bool)

    /// Check if a block at given address and order is marked as free
    fn is_block_free(
        &self,
        inner: &BuddyAllocatorInner<MAX_ORDER>,
        addr: NonNull<u8>,
        order: usize
    ) -> bool {
        let bit_index = self.get_bit_index(addr, order);
        let word_offset = inner.bitmap_offsets[order] + bit_index / 64;
        let bit_position = bit_index % 64;

        // Extract the bit: shift right, then mask with 1
        (inner.bitmap_storage[word_offset] >> bit_position) & 1 == 1
    }

    /// Mark a block at given address and order as free or allocated
    fn set_block_free(
        &self,
        inner: &mut BuddyAllocatorInner<MAX_ORDER>,
        addr: NonNull<u8>,
        order: usize,
        is_free: bool,
    ) {
        let bit_index = self.get_bit_index(addr, order);
        let word_offset = inner.bitmap_offsets[order] + bit_index / 64;
        let bit_position = bit_index % 64;

        if is_free {
            // Set the bit: OR with a mask that has 1 in the target position
            inner.bitmap_storage[word_offset] |= 1u64 << bit_position;
        } else {
            // Clear the bit: AND with a mask that has 0 in target position, 1s elsewhere
            inner.bitmap_storage[word_offset] &= !(1u64 << bit_position);
        }
    }
}