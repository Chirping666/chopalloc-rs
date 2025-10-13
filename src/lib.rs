#![no_std]
mod errors;
mod raw_spin_lock;

use core::alloc::Layout;
use core::ptr::NonNull;
// use anyhow::Result;
use errors::*;
use lock_api::Mutex;
use raw_spin_lock::*;

// Free block node stored directly in free memory
#[repr(C)]
struct FreeBlock {
    next: Option<NonNull<FreeBlock>>,
}

impl FreeBlock {
    #[inline]
    fn min_block_size() -> usize {
        core::mem::size_of::<FreeBlock>().next_power_of_two()
    }

    #[inline]
    fn min_order() -> usize {
        FreeBlock::min_block_size().trailing_zeros() as usize
    }
}

// Inner allocator state protected by mutex
// `MAX_ORDER` represents the number of order levels managed by the allocator.
// Valid block orders therefore range from 0 up to `MAX_ORDER - 1`.
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
        bitmap_storage: &'static mut [u64],
    ) -> Result<Self> {
        Self::validate_configuration(memory_region, memory_size)?;
        if MAX_ORDER == 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "MAX_ORDER must be at least 1",
            });
        }

        if memory_size == 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "memory size must be greater than zero",
            });
        }

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

        let bitmap_offsets = Self::build_bitmap_offsets(memory_size, bitmap_storage.len())?;

        // Initialize all bits to 0 (allocated)
        bitmap_storage.fill(0);

        let inner = BuddyAllocatorInner {
            free_lists: [None; MAX_ORDER],
            bitmap_storage,
            bitmap_offsets,
        };

        let allocator = Self {
            inner: Mutex::new(inner),
            base_addr: memory_region,
            total_size: memory_size,
        };

        // Initialize with top-level free blocks
        allocator.initialize_free_memory()?;

        Ok(allocator)
    }

    fn validate_configuration(memory_region: NonNull<u8>, memory_size: usize) -> Result<()> {
        if MAX_ORDER == 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "MAX_ORDER must be at least 1",
            });
        }

        if MAX_ORDER > usize::BITS as usize {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "MAX_ORDER exceeds native pointer width",
            });
        }

        if memory_size == 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "memory size must be greater than zero",
            });
        }

        let min_order = Self::min_order();
        if min_order >= MAX_ORDER {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "MAX_ORDER is too small for allocator metadata",
            });
        }

        let min_block_size = Self::min_block_size();
        if memory_size % min_block_size != 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "memory size must be a multiple of the minimum block size",
            });
        }

        let largest_block_size = Self::largest_block_size();
        if memory_size % largest_block_size != 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "memory size must be a multiple of the largest block size",
            });
        }

        if (memory_region.as_ptr() as usize) % core::mem::align_of::<FreeBlock>() != 0 {
            return Err(BuddyAllocatorError::InvalidMemoryRegion {
                base_addr: memory_region,
                size: memory_size,
                reason: "memory region is not aligned for allocator metadata",
            });
        }

        Ok(())
    }

    fn build_bitmap_offsets(memory_size: usize, bitmap_words: usize) -> Result<[usize; MAX_ORDER]> {
        let mut bitmap_offsets = [0; MAX_ORDER];
        let mut current_offset = 0;
        let largest_block_size = Self::largest_block_size();
        let min_order = Self::min_order();

        for order in min_order..MAX_ORDER {
            let block_size = 1 << order;
            if block_size > largest_block_size {
                break;
            }

            bitmap_offsets[order] = current_offset;
            let blocks_count = memory_size / block_size;
            let words_for_this_order = (blocks_count + 63) / 64;
            current_offset += words_for_this_order;
        }

        if bitmap_words < current_offset {
            return Err(BuddyAllocatorError::BitmapStorageTooSmall {
                required_words: current_offset,
                provided_words: bitmap_words,
            });
        }

        Ok(bitmap_offsets)
    }

    fn initialize_free_memory(&self) -> Result<()> {
        let mut guard = self.inner.lock();
        let largest_block_order = Self::largest_order();
        let largest_block_size = Self::largest_block_size();
        debug_assert!(self.total_size % largest_block_size == 0);

        let mut current_addr = self.base_addr.as_ptr() as usize;
        let end_addr = current_addr + self.total_size;

        while current_addr < end_addr {
            let block_ptr = NonNull::new(current_addr as *mut u8).unwrap();
            self.set_block_free(&mut guard, block_ptr, largest_block_order, true);
            Self::push_free_block(&mut guard.free_lists, block_ptr, largest_block_order);
            current_addr += largest_block_size;
        }

        Ok(())
    }

    pub fn calculate_bitmap_words_needed(memory_size: usize) -> usize {
        if memory_size == 0 {
            return 0;
        }

        if memory_size % Self::min_block_size() != 0 {
            return 0;
        }

        let min_order = Self::min_order();
        let mut total_words = 0;
        for order in min_order..MAX_ORDER {
            let block_size = 1 << order;
            if block_size > memory_size {
                break;
            }
        for order in 0..MAX_ORDER {
            let block_size = 1 << order; // 2^order bytes per block

            let blocks_count = memory_size / block_size;
            let words_for_this_order = (blocks_count + 63) / 64;
            total_words += words_for_this_order;
        }

        total_words
    }

    /// Try to allocate memory matching the given layout
    pub fn try_allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let (_, order) = self.normalize_layout(layout)?;
        let mut guard = self.inner.lock();

        // Try to find a block of the exact size
        if let Some(block) = Self::pop_free_block(&mut guard.free_lists, order) {
            self.set_block_free(&mut guard, block, order, false); // Mark as allocated!
            return Ok(block);
        }

        // Find a larger block and split it down
        for larger_order in (order + 1)..MAX_ORDER {
            if let Some(large_block) = Self::pop_free_block(&mut guard.free_lists, larger_order) {
                let allocated_block =
                    self.split_block_down_to(&mut guard, large_block, larger_order, order);
                self.set_block_free(&mut guard, allocated_block, order, false); // Mark as allocated!
                return Ok(allocated_block);
            }
        }

        // No free blocks available
        Err(BuddyAllocatorError::OutOfMemory {
            requested_order: order,
            largest_available_order: Self::largest_available_order(&guard),
        })
    }

    /// Try to deallocate memory at the given pointer
    pub fn try_deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        let (_, order) = self.normalize_layout(layout)?;
        let mut guard = self.inner.lock();

        // Validation
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base_addr.as_ptr() as usize;

        if ptr_addr < base_addr || ptr_addr >= base_addr + self.total_size {
            return Err(BuddyAllocatorError::InvalidPointer {
                ptr,
                base_addr: self.base_addr,
                region_size: self.total_size,
            });
        }

        let block_size = 1 << order;
        if (ptr_addr - base_addr) % block_size != 0 {
            return Err(BuddyAllocatorError::InvalidAlignment {
                ptr,
                block_size,
                required_alignment: block_size,
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
            let block_size = 1 << current_order;
            let base_addr = self.base_addr.as_ptr() as usize;
            let current_offset = current_ptr.as_ptr() as usize - base_addr;
            debug_assert!(current_offset < self.total_size);
            let buddy_offset = current_offset ^ block_size;
            debug_assert!(buddy_offset < self.total_size);
            let buddy_addr = base_addr + buddy_offset;
            let buddy_ptr = NonNull::new(buddy_addr as *mut u8).unwrap();

            if !self.is_block_free(&guard, buddy_ptr, current_order) {
                break;
            }

            let removed =
                Self::remove_from_free_list(&mut guard.free_lists, buddy_ptr, current_order);
            debug_assert!(removed, "Bitmap-freelist inconsistency");
            if !removed {
                break;
            }

            self.set_block_free(&mut guard, current_ptr, current_order, false);
            self.set_block_free(&mut guard, buddy_ptr, current_order, false);

            current_ptr = if current_ptr.as_ptr() < buddy_ptr.as_ptr() {
                current_ptr
            } else {
                buddy_ptr
            };
            current_order += 1;

            self.set_block_free(&mut guard, current_ptr, current_order, true);
        }

        Self::push_free_block(&mut guard.free_lists, current_ptr, current_order);

        Ok(())
    }

    fn remove_from_free_list(
        free_lists: &mut [Option<NonNull<FreeBlock>>; MAX_ORDER],
        target: NonNull<u8>,
        order: usize,
    ) -> bool {
        if order >= MAX_ORDER || order < Self::min_order() {
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
        order: usize,
    ) -> Option<NonNull<u8>> {
        if order >= MAX_ORDER || order < Self::min_order() {
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
        order: usize,
    ) {
        if order >= MAX_ORDER || order < Self::min_order() {
            return;
        }

        unsafe {
            debug_assert_eq!(
                (ptr.as_ptr() as usize) % core::mem::align_of::<FreeBlock>(),
                0
            );
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
        block: NonNull<u8>,
        from_order: usize,
        to_order: usize,
    ) -> NonNull<u8> {
        let mut current_order = from_order;
        let current_block = block;

        self.set_block_free(inner, current_block, current_order, false);

        while current_order > to_order {
            current_order -= 1;
            let block_size = 1 << current_order;

            let buddy_ptr =
                unsafe { NonNull::new_unchecked(current_block.as_ptr().add(block_size)) };
            let buddy_offset = buddy_ptr.as_ptr() as usize - self.base_addr.as_ptr() as usize;
            debug_assert!(buddy_offset < self.total_size);

            // ✅ NOW we can mark the buddy as free in the bitmap!
            self.set_block_free(inner, buddy_ptr, current_order, true);
            Self::push_free_block(&mut inner.free_lists, buddy_ptr, current_order);
            self.set_block_free(inner, current_block, current_order, false);
        }

        current_block
    }

    /// Convert address to bit index for a given order
    fn get_bit_index(&self, addr: NonNull<u8>, order: usize) -> usize {
        let offset_from_base = addr.as_ptr() as usize - self.base_addr.as_ptr() as usize;
        let block_size = 1 << order; // 2^order
        debug_assert!(offset_from_base < self.total_size);
        debug_assert_eq!(offset_from_base % block_size, 0);
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
        order: usize,
    ) -> bool {
        if order >= MAX_ORDER || order < Self::min_order() {
        if order >= MAX_ORDER {
            return false;
        }

        let bit_index = self.get_bit_index(addr, order);
        let word_offset = inner.bitmap_offsets[order] + bit_index / 64;
        let bit_position = bit_index % 64;
        debug_assert!(word_offset < inner.bitmap_storage.len());

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
        if order >= MAX_ORDER || order < Self::min_order() {
        if order >= MAX_ORDER {
            return;
        }

        let bit_index = self.get_bit_index(addr, order);
        let word_offset = inner.bitmap_offsets[order] + bit_index / 64;
        let bit_position = bit_index % 64;
        debug_assert!(word_offset < inner.bitmap_storage.len());

        if is_free {
            // Set the bit: OR with a mask that has 1 in the target position
            inner.bitmap_storage[word_offset] |= 1u64 << bit_position;
        } else {
            // Clear the bit: AND with a mask that has 0 in target position, 1s elsewhere
            inner.bitmap_storage[word_offset] &= !(1u64 << bit_position);
        }
    }

    fn min_order() -> usize {
        FreeBlock::min_order()
    }

    fn min_block_size() -> usize {
        FreeBlock::min_block_size()
    }

    fn largest_order() -> usize {
        MAX_ORDER - 1
    }

    fn largest_block_size() -> usize {
        1 << Self::largest_order()
    }

    fn normalize_layout(&self, layout: Layout) -> Result<(usize, usize)> {
        if layout.size() == 0 {
            return Err(BuddyAllocatorError::InvalidLayout {
                size: layout.size(),
                align: layout.align(),
                reason: "zero-sized allocations are not supported",
            });
        }

        let mut requested = layout.size().max(layout.align());
        requested = requested.max(Self::min_block_size());

        if requested > Self::largest_block_size() {
            return Err(BuddyAllocatorError::AllocationTooLarge {
                requested_bytes: requested,
                max_block_size: Self::largest_block_size(),
            });
        }

        let size = requested.next_power_of_two();
        let order = size.trailing_zeros() as usize;

        if order >= MAX_ORDER {
            return Err(BuddyAllocatorError::AllocationTooLarge {
                requested_bytes: size,
                max_block_size: Self::largest_block_size(),
            });
        }

        Ok((size, order))
    }

    fn largest_available_order(inner: &BuddyAllocatorInner<MAX_ORDER>) -> Option<usize> {
        (Self::min_order()..MAX_ORDER)
            .rev()
            .find(|&order| inner.free_lists[order].is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::alloc::Layout;
    use core::ptr;

    #[test]
    fn test_buddy_merging_cascade() {
        static mut MEMORY: [usize; 1024 / core::mem::size_of::<usize>()] =
            [0; 1024 / core::mem::size_of::<usize>()];
        static mut BITMAP: [u64; 64] = [0; 64];

        unsafe {
            // Use addr_of_mut! to get raw pointers, then convert to NonNull
            let memory_ptr = NonNull::new(ptr::addr_of_mut!(MEMORY).cast::<u8>()).unwrap();
            let bitmap_words = BuddyAllocator::<11>::calculate_bitmap_words_needed(1024);
            assert!(bitmap_words <= 64);
            let bitmap_slice = core::slice::from_raw_parts_mut(
                ptr::addr_of_mut!(BITMAP).cast::<u64>(),
                bitmap_words,
            );
            let bitmap_slice =
                core::slice::from_raw_parts_mut(ptr::addr_of_mut!(BITMAP).cast::<u64>(), 100);

            let allocator = BuddyAllocator::<11>::new(memory_ptr, 1024, bitmap_slice).unwrap();

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
            assert!(
                big_ptr.is_ok(),
                "Should be able to allocate 1024 bytes after merging"
            );
        }
    }
}
