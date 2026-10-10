# Native expert forward connection

The source starts from cx-flash's integration baseline a35d0b93. Apply packed preparation commit 687e76db first. This track changes recipe.rs and barrier.ptx; packed.ptx remains the packed track's source.

The target/head graphs replace each gate/up/activation/product/down region with the native ExpertOut dispatch operation. Its source remains live until dispatch. The ordinary tape reserves only the descriptor slot and omits every external expert matrix. One shared ExpertExecution retains exact packed expert bytes, worker tables, pinned spill, and packet sequences for both models.

Each forward queues one persistent worker grid on each nonmain die before launching the main forward. The main kernel reads actual native TopK count/ID records and expert-major FP32 coefficients, packs position-major hidden columns, calls split_send, runs local main experts, calls split_combine, and restores native channel/position layout. The worker performs gate/up, SiLU product, down, router-weighted reduction, and return inside that invocation. The selected expert union shares all 16 SMs by logical CTA rank/count; identical expert matrix sizes within a GGUF layer give equal byte shares. Larger unions use waves. Whole-die preparation handles the groups' compact products with a completion barrier before down.

Every local completion stage has a distinct tag. p2p_publish_tagged separates that tag from the peer protocol sequence. The existing p2p_publish interface is preserved. Integrated status records identify die, layer, phase, and CTA. A failed invocation poisons the execution and preserves allocations and trace; it is not replayed.

Main is physical die 4. Physical die 2 is capped at 5798205440 bytes before placement; die 0 keeps its existing cap, and die 6 never receives a context. The worker pointers may refer to resident VRAM or mapped pinned RAM. RECIPE_EXPERT_PICK_COUNTS can supply model,layer,expert,picks CSV records. Placement keeps higher-frequency experts resident first; absent history, zero-frequency ties use deterministic tensor order. Real forward selections update main-die counters, and the CSV is written after a completed invocation. Cold-start ties are not measured frequency evidence.

Operation reports include each worker's selected expert union and packed weight bytes. Report memory counts include worker allocations once across the shared target/head placement. Pinned-RAM bytes are separate in the placement table.

The actual packed_matvec PTX uses 14 arguments and an extern dynamic shared buffer, despite the earlier 15-argument prose contract. Workers launch with 40960 dynamic shared bytes. Logical packed_prepare uses eight arguments after 687e76db.

## Evidence

- Library and CLI build pass on Archy.
- The composed worker is assembled separately with ptxas 11.4, PTX 7.4, sm_52, -c -O1, then linked by nvlink. Device input names retain the .cubin extension; an .o extension produced an empty linked image and is not used.
- The retained worker symbol reports 252 registers, a 656-byte stack, 16 static shared bytes, and no local-memory bytes. Its runtime occupancy check uses 256 threads and 40960 dynamic shared bytes.
- A monolithic -O3 assembler attempt exited 137 after exhausting RAM. Separate compilation avoids that failure. The native main compiler uses the same separate-link mode when it emits split_main_forward.
- placement.md uses actual target/head headers and the actual lowered five-position layouts at context 128, on the a35d0b93 precision configuration. Main reserve is 2227327718 bytes, including request buffers and the 32 MiB execution reserve. It returns zero unplaced bytes and 1441382400 pinned-RAM bytes. The saved free-VRAM snapshot has no model allocations. The current root flash profile must recompute this table at startup.

No GPU model invocation is claimed from these build and planning results. The die-memory sweep, combined root build, actual forward trace, timeout resolution, router-selected overhead, whole-model rate, and final-logit agreement remain runtime gates. cx-flash owns the coordinated restart. No GPU kernel is launched by this source track before that window.

## Current integration port

The current branch ports the connection onto d1a186a, preserving root's Result-based inference window, BF16 handling, mapped allocation reuse, and flash profile. It consumes the published 431203e 256-thread library and uses measured kind/row-lane tuples for the real expert types and capacities. Main and worker grids both use 256 threads; the expert module uses the forward entry for a single position as well as prefill. A separate 512-thread step is not compiled for this module.

The selector may include capped die 2 or explicitly omit it. Omission creates no die-2 context, buffer, packet, or worker. It does not automatically recover a failed GPU or change the selected topology. Root retains that decision. Die 6 stays excluded.

RECIPE_EXPERT_PICK_COUNTS is a read-only ranking seed. It accepts the root reference TSV (`layer, expert, picks, router_positions`, separated by tabs) or a target/head CSV. Reference seed counts are not added to the live Recipe counters. RECIPE_EXPERT_PICK_LOG selects a separate output path; absent that setting, actual Recipe picks go to router-picks.csv in the worker artifact directory. The reference file is preserved.

`current/` records the current flash-profile CPU plans, using the real target/head lowering at context 128 and five positions. The main reserve is 1313660806 bytes. The seven-die plan spills 527462400 bytes; the plan explicitly excluding die 2 spills 6288128000 bytes. Both leave zero unplaced bundles/bytes. These are computed plans; live startup recomputes CUDA free bytes after context creation.

The current library/CLI build and composed worker assembly/link pass. The retained worker reports 252 registers, 488 stack bytes, 16 static shared bytes, and zero local-memory bytes. No real GPU forward or rate is claimed. The original receipts remain historical snapshots, separate from these current receipts.
