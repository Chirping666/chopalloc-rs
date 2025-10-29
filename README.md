# chopalloc
A minimal, no_std buddy allocator for embedded systems and bare-metal environments.

## Features

- **`bitmap` (default)**: Enables O(1) buddy checking using a bitmap. Disable for pure free-list implementation (O(n) checks, less memory overhead).

## Choosing MAX_ORDER

The `MAX_ORDER` const generic parameter determines how many block size levels the allocator manages.

**Formula**: `MAX_ORDER = log2(memory_size_in_bytes) + 1`

Examples:
- 1 KB (1024 bytes) → `BuddyAllocator::<11>` (2^10 = 1024)
- 4 KB (4096 bytes) → `BuddyAllocator::<13>` (2^12 = 4096)
- 64 KB (65536 bytes) → `BuddyAllocator::<17>` (2^16 = 65536)

The allocator will manage blocks from 2^0 up to 2^(MAX_ORDER-1) bytes.

## Usage

```rust
use chopalloc::BuddyAllocator;
use core::alloc::Layout;

// With bitmap (default)
#[repr(align(1024))]
static mut MEMORY: [u8; 1024] = [0; 1024];
static mut BITMAP: [u64; 64] = [0; 64];

let allocator = unsafe {
    let memory_ptr = NonNull::new(MEMORY.as_mut_ptr()).unwrap();
    let bitmap_words = BuddyAllocator::<11>::calculate_bitmap_words_needed(1024);
    let bitmap_slice = &mut BITMAP[..bitmap_words];
    BuddyAllocator::<11>::new(memory_ptr, 1024, bitmap_slice).unwrap()
};

let layout = Layout::from_size_align(64, 64).unwrap();
let ptr = allocator.try_allocate(layout).unwrap();
allocator.try_deallocate(ptr, layout).unwrap();
```

### Without Bitmap

```rust
// In Cargo.toml: chopalloc = { version = "1.0", default-features = false }

let allocator = unsafe {
    let memory_ptr = NonNull::new(MEMORY.as_mut_ptr()).unwrap();
    BuddyAllocator::<11>::new(memory_ptr, 1024).unwrap() // No bitmap parameter
};
```

## Alignment Requirements

The backing memory region must be aligned to the allocator's largest block size
(`BuddyAllocator::largest_block_size()`). Use `#[repr(align(...))]` for static buffers:

```rust
#[repr(align(1024))]  // Align to largest block size
static mut MEMORY: [u8; 1024] = [0; 1024];
```

## Examples

See `examples/` for:
- `basic_usage.rs` - Fundamental operations
- `fragmentation_demo.rs` - Buddy merging behavior
- `embedded_simulation.rs` - Simulated embedded system
- `no_bitmap.rs` - Free-list-only mode (compile with `--no-default-features`)

