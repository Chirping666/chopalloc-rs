// Basic usage example of the chopalloc buddy allocator
// This demonstrates fundamental allocation and deallocation operations

use chopalloc::BuddyAllocator;
use core::alloc::Layout;
use core::ptr::{self, NonNull};

// Static memory backing for the allocator (1 KB, aligned to 1024)
#[repr(align(1024))]
struct AlignedMemory([u8; 1024]);

static mut MEMORY: AlignedMemory = AlignedMemory([0; 1024]);
static mut BITMAP: [u64; 64] = [0; 64];

fn main() {
    unsafe {
        // Calculate required bitmap size
        let bitmap_words = BuddyAllocator::<11>::calculate_bitmap_words_needed(1024);
        println!("Required bitmap words: {}", bitmap_words);

        // Create memory pointers
        let memory_ptr = NonNull::new(ptr::addr_of_mut!(MEMORY.0).cast::<u8>()).unwrap();
        let bitmap_slice =
            core::slice::from_raw_parts_mut(ptr::addr_of_mut!(BITMAP).cast::<u64>(), bitmap_words);

        // Initialize the allocator
        let allocator = match BuddyAllocator::<11>::new(memory_ptr, 1024, bitmap_slice) {
            Ok(alloc) => {
                println!("✓ Allocator initialized successfully");
                alloc
            }
            Err(e) => {
                eprintln!("✗ Failed to initialize allocator: {}", e);
                return;
            }
        };

        // Example 1: Allocate a 64-byte block
        println!("\n--- Example 1: Basic Allocation ---");
        let layout_64 = Layout::from_size_align(64, 64).unwrap();
        match allocator.try_allocate(layout_64) {
            Ok(ptr) => {
                println!("✓ Allocated 64 bytes at address: {:p}", ptr.as_ptr());

                // Use the memory (write some data)
                let slice = core::slice::from_raw_parts_mut(ptr.as_ptr(), 64);
                slice[0] = 42;
                slice[63] = 99;
                println!("  Wrote data: [{}, ..., {}]", slice[0], slice[63]);

                // Deallocate
                match allocator.try_deallocate(ptr, layout_64) {
                    Ok(_) => println!("✓ Deallocated 64 bytes"),
                    Err(e) => eprintln!("✗ Deallocation failed: {}", e),
                }
            }
            Err(e) => eprintln!("✗ Allocation failed: {}", e),
        }

        // Example 2: Multiple allocations
        println!("\n--- Example 2: Multiple Allocations ---");
        let mut allocations = Vec::new();

        for i in 0..5 {
            let layout = Layout::from_size_align(32, 32).unwrap();
            match allocator.try_allocate(layout) {
                Ok(ptr) => {
                    println!("  [{}] Allocated 32 bytes at {:p}", i, ptr.as_ptr());
                    allocations.push(ptr);
                }
                Err(e) => {
                    eprintln!("  [{}] Allocation failed: {}", i, e);
                    break;
                }
            }
        }

        // Deallocate in reverse order
        println!("\nDeallocating...");
        for (i, ptr) in allocations.iter().enumerate().rev() {
            let layout = Layout::from_size_align(32, 32).unwrap();
            match allocator.try_deallocate(*ptr, layout) {
                Ok(_) => println!("  [{}] Deallocated", i),
                Err(e) => eprintln!("  [{}] Deallocation failed: {}", i, e),
            }
        }

        // Example 3: Alignment requirements
        println!("\n--- Example 3: High Alignment ---");
        let layout_aligned = Layout::from_size_align(8, 256).unwrap();
        match allocator.try_allocate(layout_aligned) {
            Ok(ptr) => {
                let addr = ptr.as_ptr() as usize;
                println!("✓ Allocated with 256-byte alignment at: {:p}", ptr.as_ptr());
                println!("  Alignment check: {} % 256 = {}", addr, addr % 256);
                allocator.try_deallocate(ptr, layout_aligned).unwrap();
            }
            Err(e) => eprintln!("✗ Allocation failed: {}", e),
        }

        // Example 4: Out of memory
        println!("\n--- Example 4: Out of Memory ---");
        let layout_large = Layout::from_size_align(2048, 1).unwrap();
        match allocator.try_allocate(layout_large) {
            Ok(_) => println!("✗ Unexpected success - should have failed!"),
            Err(e) => println!("✓ Expected failure: {}", e),
        }

        println!("\n--- All examples completed successfully ---");
    }
}
