# Full-corpus Rust evaluator run

- Policy: `full`, one epoch, 770,493 source rows.
- Started: 2026-09-09 11:12:03 UTC.
- Finished: 2026-09-09 11:14:31 UTC.
- Total elapsed: 148 seconds, including setup and model updates.
- Epoch elapsed: 144.764884 seconds.
- Exit status: 0; systemd result: success.
- All 770,493 scores were returned in input order.
- All 9 GPUs and both CPU pools processed records.
- Compared 377,877 overlapping records against the original SQL run: identical proposal records and metric differences below 1e-9 relative tolerance. Maximum absolute metric difference: 3.1039615322470127e-12.
- Best observed quota R2: -0.03217716606037002, record 371,839.
- Surrogate R2: -0.2506037755199426.
- Saved model: `quota-model.ogdl`.

The measured quota scores belong to the proposal batch before the final proposer update. The saved model was not externally rescored after that update.

| Device | Scored proposals |
| --- | ---: |
| Engi amd0 | 258,049 |
| Engi CPU pool | 111,617 |
| Archy nv0 | 49,153 |
| Archy nv1 | 49,153 |
| Archy nv2 | 49,153 |
| Archy nv3 | 49,153 |
| Archy nv4 | 49,153 |
| Archy nv5 | 49,153 |
| Archy nv6 | 45,057 |
| Archy nv7 | 47,793 |
| Archy CPU pool | 13,059 |

Live observation during scoring: Archy's eight NVIDIA GPUs each reported 100% utilization, Engi AMD reported 99%, and both CPU pools were active. All remote GPU processes exited after completion.

Reproduce verification with `ruby verify.rb` in this directory. The previous SQL run's logs were preserved at `../full-vector.iDFL4FYi`.
