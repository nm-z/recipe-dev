# Branch and pull request policy

- Development pull requests target `minimal`.
- `minimal`, `main`, `legacy`, `experimental/AOT`, and `experimental/RAT` are persistent branches and cannot be deleted.
- GitHub automatically deletes other pull request branches after merge.
- A pull request that conflicts with `minimal` is converted to draft and receives one conflict notice. Automation never marks it ready.
- If GitHub reports mergeability as unknown while it calculates, automation retries and then defers without changing the pull request.
- If a branch is stacked on another development branch, rebase or restack it as needed before changing the base to `minimal`. Automation does not retarget it.
