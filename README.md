# stencil op

lol i tried this then realized its already perfectly coalesced. third times the charm (gonna try a matrix transpose)

baseline kernel
(main) root@C.28510276:/workspace/stencil/build$ ./bin/stencil 
--- Stencil Op Stable Timing Test ---
Array Size N: 16777216 (0.0625 GB)
Grid: 65536 blocks, 256 threads/block.
Warming up the GPU and Caches (10 runs)...
Starting Stable Timing Loop (100 runs)...

--- Timing Results ---
Total execution time for 100 stable runs: 11.0019 ms
**Average kernel execution time:** 110.019 us

Verification Check: **PASSED** (midpoint value: 3)

calculation:
read 3, write 1 so 4 * 4 bytes cuz float = 16 bytes per thread * N threads = 16 * 16,777,216 = 268,435,456 bytes
time it takes = 0.000109281 s

268,435,456 / 0.000109281 = 2,456,378,107,813.8011182182 ~= 2.456 trillion bytes / second 

how is this possible when A100 theoretical max is 1.555 GB/s? Because it seems like the gpu driver makes good utilization of the L2 cache.

Bruh lol this is already perfectly coalesced as well
