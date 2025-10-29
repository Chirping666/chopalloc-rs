// Demonstration of memory fragmentation and coalescing in the buddy allocator
// Shows how buddy merging works when deallocating adjacent blocks

use chopalloc::BuddyAllocator;
use core::alloc::Layout;
use core::ptr::{self, NonNull};

#[repr(align(4096))]
struct AlignedMemory([u8; 4096]);

static mut MEMORY: AlignedMemory = AlignedMemory([0; 4096]);
static mut BITMAP: [u64; 256] = [0; 256];

fn main() {
    unsafe {
        let bitmap_words = BuddyAllocator::<13>::calculate_bitmap_words_needed(4096);
        let memory_ptr = NonNull::new(ptr::addr_of_mut!(MEMORY.0).cast::<u8>()).unwrap();
        let bitmap_slice =
            core::slice::from_raw_parts_mut(ptr::addr_of_mut!(BITMAP).cast::<u64>(), bitmap_words);

        let allocator = BuddyAllocator::<13>::new(memory_ptr, 4096, bitmap_slice)
            .expect("Failed to create allocator");

        println!("=== Buddy Allocator Fragmentation Demo ===\n");

        // Phase 1: Allocate many small blocks
        println!("Phase 1: Allocating 32 small blocks (64 bytes each)");
        let layout_small = Layout::from_size_align(64, 64).unwrap();
        let mut blocks = Vec::new();

        for i in 0..32 {
            match allocator.try_allocate(layout_small) {
                Ok(ptr) => {
                    blocks.push(ptr);
                    if i % 8 == 7 {
                        println!("  Allocated blocks 0-{}", i);
                    }
                }
                Err(e) => {
                    println!("  Failed at block {}: {}", i, e);
                    break;
                }
            }
        }
        println!("  Total allocated: {} blocks\n", blocks.len());

        // Phase 2: Free every other block to create fragmentation
        println!("Phase 2: Freeing every other block (creating fragmentation)");
        let mut freed_count = 0;
        for i in (1..blocks.len()).step_by(2) {
            allocator.try_deallocate(blocks[i], layout_small).unwrap();
            freed_count += 1;
        }
        println!("  Freed {} blocks, {} remain allocated\n", freed_count, blocks.len() - freed_count);

        // Phase 3: Try to allocate a large block (might succeed if buddy pairs coalesced)
        println!("Phase 3: Attempting to allocate large block (512 bytes)");
        let layout_large = Layout::from_size_align(512, 512).unwrap();
        match allocator.try_allocate(layout_large) {
            Ok(ptr) => {
                println!("  ✓ Succeeded (some buddy pairs were already coalesced)");
                allocator.try_deallocate(ptr, layout_large).unwrap();
            }
            Err(e) => println!("  ✗ Failed: {}", e),
        }
        println!("  Note: Whether this succeeds depends on the specific allocation pattern\n");

        // Phase 4: Free remaining blocks to enable coalescing
        println!("Phase 4: Freeing remaining blocks (enabling coalescing)");
        for i in (0..blocks.len()).step_by(2) {
            allocator.try_deallocate(blocks[i], layout_small).unwrap();
        }
        println!("  All blocks freed - buddy merging should occur\n");

        // Phase 5: Now the large allocation should succeed
        println!("Phase 5: Retrying large block allocation");
        match allocator.try_allocate(layout_large) {
            Ok(ptr) => {
                println!("  ✓ Successfully allocated 512 bytes at {:p}", ptr.as_ptr());
                println!("  (Buddies were coalesced into larger blocks)\n");

                // Write pattern to verify memory is usable
                let slice = core::slice::from_raw_parts_mut(ptr.as_ptr(), 512);
                for (i, byte) in slice.iter_mut().enumerate() {
                    *byte = (i % 256) as u8;
                }
                println!("  Wrote test pattern to memory");
                println!("  Sample values: [{}, {}, {}, ...]", slice[0], slice[1], slice[2]);

                // Clean up
                allocator.try_deallocate(ptr, layout_large).unwrap();
            }
            Err(e) => println!("  ✗ Still failed: {}", e),
        }

        // Phase 6: Demonstrate buddy merging cascade
        println!("\nPhase 6: Demonstrating buddy merging cascade");
        let layout_128 = Layout::from_size_align(128, 128).unwrap();

        println!("  Allocating 4x 128-byte blocks...");
        let b1 = allocator.try_allocate(layout_128).unwrap();
        let b2 = allocator.try_allocate(layout_128).unwrap();
        let b3 = allocator.try_allocate(layout_128).unwrap();
        let b4 = allocator.try_allocate(layout_128).unwrap();

        println!("  Blocks at: {:p}, {:p}, {:p}, {:p}",
                 b1.as_ptr(), b2.as_ptr(), b3.as_ptr(), b4.as_ptr());

        println!("  Freeing in order: b2, b1 (should merge), then b4, b3 (should merge)");
        allocator.try_deallocate(b2, layout_128).unwrap();
        println!("    Freed b2");
        allocator.try_deallocate(b1, layout_128).unwrap();
        println!("    Freed b1 -> b1+b2 should merge into 256-byte block");

        allocator.try_deallocate(b4, layout_128).unwrap();
        println!("    Freed b4");
        allocator.try_deallocate(b3, layout_128).unwrap();
        println!("    Freed b3 -> b3+b4 should merge into 256-byte block");
        println!("             -> then both 256-byte blocks merge into 512-byte block");

        println!("\n=== Demo completed successfully ===");
    }
}
