#![no_std]
mod raw_spin_lock;
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

        let mut inner = BuddyAllocatorInner {
            free_lists: [None; MAX_ORDER],
            bitmap_storage,
            bitmap_offsets,
        };

        let allocator = Self {
            inner: Mutex::new(inner),
            base_addr: memory_region,
            total_size: memory_size,
        };

        // Initialize with one giant free block
        allocator.initialize_free_memory()?;

        Ok(allocator)
    }

    fn initialize_free_memory(&self) -> Result<()> {
        let mut guard = self.inner.lock();
        let mut remaining_size = self.total_size;
        let mut current_addr = self.base_addr.as_ptr() as usize;

        while remaining_size > 0 {
            // Find largest power-of-2 block that fits
            let block_order = remaining_size.trailing_zeros() as usize;
            let block_order = block_order.min(MAX_ORDER - 1);
            let block_size = 1 << block_order;

            let block_ptr = NonNull::new(current_addr as *mut u8).unwrap();

            self.set_block_free(&mut guard, block_ptr, block_order, true);
            Self::push_free_block(&mut guard.free_lists, block_ptr, block_order);

            current_addr += block_size;
            remaining_size -= block_size;
        }

        Ok(())
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

        // Check if order is too large
        if order >= MAX_ORDER {
            return Err(BuddyAllocatorError::AllocationTooLarge {
                requested_bytes: size,
                max_block_size: 1 << (MAX_ORDER - 1),
            });
        }

        // Try to find a block of the exact size
        if let Some(block) = Self::pop_free_block(&mut guard.free_lists, order) {
            self.set_block_free(&mut guard, block, order, false); // Mark as allocated!
            return Ok(block);
        }

        // Find a larger block and split it down
        for larger_order in (order + 1)..MAX_ORDER {
            if let Some(large_block) = Self::pop_free_block(&mut guard.free_lists, larger_order) {
                let allocated_block = self.split_block_down_to(&mut guard, large_block, larger_order, order);
                self.set_block_free(&mut guard, allocated_block, order, false); // Mark as allocated!
                return Ok(allocated_block);
            }
        }

        // No free blocks available
        Err(BuddyAllocatorError::OutOfMemory {
            requested_order: order,
            largest_available_order: None, // TODO: Could scan for largest available
        })
    }

    /// Try to deallocate memory at the given pointer
    pub fn try_deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        let mut guard = self.inner.lock();

        let size = layout.size().max(layout.align()).next_power_of_two();
        let order = size.trailing_zeros() as usize;

        // Validation
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base_addr.as_ptr() as usize;

        if ptr_addr < base_addr || ptr_addr >= base_addr + self.total_size {
            return Err(BuddyAllocatorError::InvalidPointer {
                ptr, base_addr: self.base_addr, region_size: self.total_size,
            });
        }

        let block_size = 1 << order;
        if (ptr_addr - base_addr) % block_size != 0 {
            return Err(BuddyAllocatorError::InvalidAlignment {
                ptr, block_size, required_alignment: block_size,
            });
        }

        if self.is_block_free(&guard, ptr, order) {
            return Err(BuddyAllocatorError::DoubleFree { ptr, order });
        }

        // Mark block as free
        self.set_block_free(&mut guard, ptr, order, true);

        // Buddy merging loop
        let mut current_ptr = ptr;
        let mut current_order = order;

        while current_order < MAX_ORDER - 1 {
            let buddy_addr = current_ptr.as_ptr() as usize ^ (1 << current_order);
            let buddy_ptr = NonNull::new(buddy_addr as *mut u8).unwrap();

            if !self.is_block_free(&guard, buddy_ptr, current_order) {
                break;
            }

            let removed = Self::remove_from_free_list(&mut guard.free_lists, buddy_ptr, current_order);
            debug_assert!(removed, "Bitmap-freelist inconsistency");

            self.set_block_free(&mut guard, buddy_ptr, current_order, false);

            current_ptr = if current_ptr.as_ptr() < buddy_ptr.as_ptr() {
                current_ptr
            } else {
                buddy_ptr
            };
            current_order += 1;
        }

        Self::push_free_block(&mut guard.free_lists, current_ptr, current_order);

        Ok(())
    }

    fn remove_from_free_list(
        free_lists: &mut [Option<NonNull<FreeBlock>>; MAX_ORDER],
        target: NonNull<u8>,
        order: usize
    ) -> bool {
        if order >= MAX_ORDER {
            return false;
        }

        let target_block = target.cast::<FreeBlock>();

        // Handle empty list case
        let head = match free_lists[order] {
            Some(h) => h,
            None => return false, // Can't remove from empty list
        };

        unsafe {
            // Handle removing the head of the list
            if head == target_block {
                free_lists[order] = head.as_ref().next;
                return true;
            }

            // Search through the rest of the list
            let mut current = head;
            loop {
                let next_ptr = match current.as_ref().next {
                    Some(next) => next,
                    None => break, // End of list, target not found
                };

                if next_ptr == target_block {
                    // Found it! Update current's next to skip over target
                    let target_next = next_ptr.as_ref().next;
                    current.as_mut().next = target_next;
                    return true;
                }

                current = next_ptr;
            }
        }

        false // Not found in list
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
        &self,
        inner: &mut BuddyAllocatorInner<MAX_ORDER>,
        mut block: NonNull<u8>,
        from_order: usize,
        to_order: usize
    ) -> NonNull<u8> {
        let mut current_order = from_order;
        let current_block = block;

        while current_order > to_order {
            current_order -= 1;
            let block_size = 1 << current_order;

            let buddy_addr = current_block.as_ptr() as usize ^ block_size;
            let buddy_ptr = NonNull::new(buddy_addr as *mut u8).unwrap();

            // ✅ NOW we can mark the buddy as free in the bitmap!
            self.set_block_free(inner, buddy_ptr, current_order, true);
            Self::push_free_block(&mut inner.free_lists, buddy_ptr, current_order);
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::alloc::Layout;
    use core::ptr;

    #[test]
    fn test_buddy_merging_cascade() {
        static mut MEMORY: [u8; 1024] = [0; 1024];
        static mut BITMAP: [u64; 100] = [0; 100];

        unsafe {
            // Use addr_of_mut! to get raw pointers, then convert to NonNull
            let memory_ptr = NonNull::new(ptr::addr_of_mut!(MEMORY).cast::<u8>()).unwrap();
            let bitmap_slice = core::slice::from_raw_parts_mut(
                ptr::addr_of_mut!(BITMAP).cast::<u64>(),
                100
            );

            let allocator = BuddyAllocator::<10>::new(memory_ptr, 1024, bitmap_slice).unwrap();

            // Test the buddy merging cascade
            let layout_64 = Layout::from_size_align(64, 64).unwrap();
            let ptr1 = allocator.try_allocate(layout_64).unwrap();
            let ptr2 = allocator.try_allocate(layout_64).unwrap();

            // Free in reverse order to trigger merging
            allocator.try_deallocate(ptr2, layout_64).unwrap();
            allocator.try_deallocate(ptr1, layout_64).unwrap();

            // Should be able to allocate full block again
            let layout_1024 = Layout::from_size_align(1024, 1024).unwrap();
            let big_ptr = allocator.try_allocate(layout_1024);
            assert!(big_ptr.is_ok(), "Should be able to allocate 1024 bytes after merging");
        }
    }
}