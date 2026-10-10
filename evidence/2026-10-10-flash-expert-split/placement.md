Real headers: 1224 target tensors, 34 head tensors, 46084915200 expert bytes, 6987059712 main-die bytes including the head's own Q8 output.

## Live free VRAM, nv0 cap, KV/scratch reserved

| Physical die | Role | Free bytes | KV/scratch reserve | Assigned weight bytes | Expert bundles | Remaining bytes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | experts | 6186598400 | 268435456 | 5916774400 | 3124 | 1388544 |
| 1 | experts | 6846152704 | 268435456 | 6576460800 | 3477 | 1256448 |
| 2 | main | 7988051968 | 536870912 | 6987064320 | 0 | 464116736 |
| 3 | experts | 2055208960 | 268435456 | 1786572800 | 956 | 200704 |
| 4 | experts | 2591031296 | 268435456 | 2321049600 | 1242 | 1546240 |
| 5 | experts | 321912832 | 268435456 | 52326400 | 28 | 1150976 |

Unplaced expert bundles: 15749; unplaced packed bytes: 29431731200; complete resident fit: false.


## Physical capacity upper bound after service release, nv0 cap, reserves

| Physical die | Role | Free bytes | KV/scratch reserve | Assigned weight bytes | Expert bundles | Remaining bytes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | experts | 6186598400 | 268435456 | 5916620800 | 3166 | 1542144 |
| 1 | experts | 8523874304 | 268435456 | 8254566400 | 4396 | 872448 |
| 2 | main | 7988051968 | 536870912 | 6987064320 | 0 | 464116736 |
| 3 | experts | 7988051968 | 268435456 | 7718220800 | 4109 | 1395712 |
| 4 | experts | 8523874304 | 268435456 | 8254566400 | 4396 | 872448 |
| 5 | experts | 8523874304 | 268435456 | 8254566400 | 4396 | 872448 |

Unplaced expert bundles: 4113; unplaced packed bytes: 7686374400; complete resident fit: false.

Planner checks PASS: byte conservation, 24,576 complete expert bundles, same-die matrices, alignment, selected-slot mapping, invalid selection rejection, and forbidden main dies. The roomy budget is a CPU fixture, not an Archy fit claim.
