// Simulated embedded system using chopalloc for dynamic memory management
// This example shows how you might use the allocator in a no_std environment

use chopalloc::BuddyAllocator;
use core::alloc::Layout;
use core::ptr::{self, NonNull};

// Simulate different memory regions in an embedded system
#[repr(align(2048))]
struct SramRegion([u8; 2048]);

static mut HEAP_MEMORY: SramRegion = SramRegion([0; 2048]);
static mut HEAP_BITMAP: [u64; 128] = [0; 128];

// Simulated task control block
#[repr(C)]
struct TaskControlBlock {
    task_id: u32,
    stack_ptr: *mut u8,
    priority: u8,
    state: u8,
}

impl TaskControlBlock {
    fn new(task_id: u32, priority: u8) -> Self {
        Self {
            task_id,
            stack_ptr: core::ptr::null_mut(),
            priority,
            state: 0,
        }
    }
}

// Simulated packet buffer
#[repr(C)]
struct PacketBuffer {
    header: [u8; 16],
    payload: [u8; 48],
    checksum: u32,
}

impl PacketBuffer {
    fn new(packet_id: u8) -> Self {
        let mut buf = Self {
            header: [0; 16],
            payload: [0; 48],
            checksum: 0,
        };
        buf.header[0] = packet_id;
        buf
    }
}

fn main() {
    unsafe {
        println!("=== Embedded System Memory Management Simulation ===\n");

        // Initialize the heap allocator
        let bitmap_words = BuddyAllocator::<12>::calculate_bitmap_words_needed(2048);
        println!("System Info:");
        println!("  Heap size: 2048 bytes");
        println!("  Bitmap words: {}", bitmap_words);
        println!("  Minimum block: {} bytes", BuddyAllocator::<12>::min_block_size());
        println!("  Maximum block: {} bytes\n", BuddyAllocator::<12>::largest_block_size());

        let heap_ptr = NonNull::new(ptr::addr_of_mut!(HEAP_MEMORY.0).cast::<u8>()).unwrap();
        let bitmap_slice =
            core::slice::from_raw_parts_mut(ptr::addr_of_mut!(HEAP_BITMAP).cast::<u64>(), bitmap_words);

        let allocator = BuddyAllocator::<12>::new(heap_ptr, 2048, bitmap_slice)
            .expect("Failed to initialize allocator");

        println!("✓ Heap allocator initialized\n");

        // Task 1: Allocate task control blocks
        println!("--- Task Management ---");
        let tcb_layout = Layout::new::<TaskControlBlock>();
        println!("  TCB size: {} bytes, align: {}", tcb_layout.size(), tcb_layout.align());

        let mut tasks = Vec::new();
        for i in 0..3 {
            match allocator.try_allocate(tcb_layout) {
                Ok(ptr) => {
                    let tcb_ptr = ptr.as_ptr() as *mut TaskControlBlock;
                    tcb_ptr.write(TaskControlBlock::new(i, (i * 10) as u8));
                    println!("  ✓ Task {} created at {:p}", i, tcb_ptr);
                    tasks.push(ptr);
                }
                Err(e) => {
                    eprintln!("  ✗ Failed to create task {}: {}", i, e);
                    break;
                }
            }
        }

        // Task 2: Allocate packet buffers
        println!("\n--- Network Packet Buffers ---");
        let packet_layout = Layout::new::<PacketBuffer>();
        println!("  Packet size: {} bytes, align: {}", packet_layout.size(), packet_layout.align());

        let mut packets = Vec::new();
        for i in 0..4 {
            match allocator.try_allocate(packet_layout) {
                Ok(ptr) => {
                    let pkt_ptr = ptr.as_ptr() as *mut PacketBuffer;
                    pkt_ptr.write(PacketBuffer::new(i as u8));
                    println!("  ✓ Packet {} allocated at {:p}", i, pkt_ptr);
                    packets.push(ptr);
                }
                Err(e) => {
                    eprintln!("  ✗ Failed to allocate packet {}: {}", i, e);
                    break;
                }
            }
        }

        // Task 3: Simulate processing and freeing packets
        println!("\n--- Processing Packets ---");
        if packets.len() >= 2 {
            let pkt0 = packets[0];
            let pkt1 = packets[1];

            println!("  Processing packets 0 and 1...");
            allocator.try_deallocate(pkt0, packet_layout).unwrap();
            println!("  ✓ Packet 0 processed and freed");

            allocator.try_deallocate(pkt1, packet_layout).unwrap();
            println!("  ✓ Packet 1 processed and freed");
            println!("  (Adjacent packets may have been coalesced)");
        }

        // Task 4: Allocate a larger DMA buffer
        println!("\n--- DMA Buffer Allocation ---");
        let dma_layout = Layout::from_size_align(256, 256).unwrap();
        match allocator.try_allocate(dma_layout) {
            Ok(ptr) => {
                println!("  ✓ DMA buffer allocated at {:p}", ptr.as_ptr());
                println!("  Alignment: {} bytes (required: {})",
                         ptr.as_ptr() as usize % 256,
                         dma_layout.align());

                // Simulate DMA transfer
                let buffer = core::slice::from_raw_parts_mut(ptr.as_ptr(), 256);
                for (i, byte) in buffer.iter_mut().enumerate() {
                    *byte = (i & 0xFF) as u8;
                }
                println!("  Simulated DMA transfer of 256 bytes");

                allocator.try_deallocate(ptr, dma_layout).unwrap();
                println!("  ✓ DMA buffer freed");
            }
            Err(e) => {
                println!("  ✗ DMA buffer allocation failed: {}", e);
            }
        }

        // Cleanup: Free remaining resources
        println!("\n--- System Shutdown ---");
        println!("  Cleaning up remaining tasks...");
        for (i, task_ptr) in tasks.iter().enumerate() {
            allocator.try_deallocate(*task_ptr, tcb_layout).unwrap();
            println!("    Task {} terminated", i);
        }

        println!("  Cleaning up remaining packets...");
        for (i, pkt_ptr) in packets.iter().enumerate().skip(2) {
            allocator.try_deallocate(*pkt_ptr, packet_layout).unwrap();
            println!("    Packet {} freed", i);
        }

        println!("\n✓ All resources freed - heap clean");
        println!("=== Simulation completed successfully ===");
    }
}
