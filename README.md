# Chop-Chop Allocator
A buddy allocator created with the help of generative AI

## Alignment requirements

When constructing a [`BuddyAllocator`](src/lib.rs), ensure the backing memory region you
provide is aligned to the allocator's largest block size (see
`BuddyAllocator::largest_block_size`). Passing an unaligned region will be rejected as an
invalid memory configuration. Preparing statically allocated buffers with `#[repr(align(...))]`
or using an allocator that can honor the requested alignment is sufficient.

