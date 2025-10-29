// Example demonstrating the allocator without bitmap optimization
// Compile with: cargo run --example no_bitmap --no-default-features

#![cfg(not(feature = "bitmap"))]

use chopalloc::BuddyAllocator;
use core::alloc::Layout;
use core::ptr::{self, NonNull};

#[repr(align(1024))]
struct AlignedMemory([u8; 1024]);

static mut MEMORY: AlignedMemory = AlignedMemory([0; 1024]);

fn main() {
    #[cfg(feature = "bitmap")]
    {
        println!("ERROR: This example must be run with --no-default-features");
        println!("Usage: cargo run --example no_bitmap --no-default-features");
        return;
    }

    #[cfg(not(feature = "bitmap"))]
    {
        unsafe {
            println!("=== Buddy Allocator WITHOUT Bitmap ===\n");
            println!("In this mode:");
            println!("  - No bitmap storage needed");
            println!("  - Buddy checking is O(n) (walks free lists)");
            println!("  - Simpler, less memory overhead");
            println!("  - Suitable for small heaps or memory-constrained systems\n");

            // Create memory pointer
            let memory_ptr = NonNull::new(ptr::addr_of_mut!(MEMORY.0).cast::<u8>()).unwrap();

            // Initialize the allocator (note: no bitmap parameter!)
            let allocator = BuddyAllocator::<11>::new(memory_ptr, 1024)
                .expect("Failed to initialize allocator");

            println!("✓ Allocator initialized (no bitmap needed)\n");

            // Allocate some blocks
            println!("--- Testing Allocations ---");
            let layout = Layout::from_size_align(64, 64).unwrap();

            let ptr1 = allocator.try_allocate(layout).unwrap();
            println!("✓ Allocated block 1 at {:p}", ptr1.as_ptr());

            let ptr2 = allocator.try_allocate(layout).unwrap();
            println!("✓ Allocated block 2 at {:p}", ptr2.as_ptr());

            let ptr3 = allocator.try_allocate(layout).unwrap();
            println!("✓ Allocated block 3 at {:p}", ptr3.as_ptr());

            // Deallocate middle block
            println!("\n--- Testing Deallocation ---");
            allocator.try_deallocate(ptr2, layout).unwrap();
            println!("✓ Deallocated block 2");

            // Try double-free detection (uses free list walking)
            println!("\n--- Testing Double-Free Detection ---");
            match allocator.try_deallocate(ptr2, layout) {
                Err(_) => println!("✓ Double-free correctly detected via free list walking"),
                Ok(_) => println!("✗ ERROR: Double-free not detected!"),
            }

            // Cleanup
            println!("\n--- Cleanup ---");
            allocator.try_deallocate(ptr1, layout).unwrap();
            allocator.try_deallocate(ptr3, layout).unwrap();
            println!("✓ All blocks freed\n");

            println!("=== Demo completed successfully ===");
            println!("\nNote: Buddy merging works the same with or without bitmap,");
            println!("but free/allocated checking is slower without bitmap.");
        }
    }
}
