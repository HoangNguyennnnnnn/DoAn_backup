# Quality Gate Closure

## Checklist

- README updated to reflect the actual notebook, script, and checkpoint layout.
- Kaggle runbook now points Stage 1 evaluation at the real decode runner.
- Stage 1 recovery guidance uses `latest_step.ckpt` first, then `interrupt.ckpt`, `latest.ckpt`, and `best.ckpt`.
- Stage 2 export/evaluation guidance prefers `best.ckpt` first, then `latest.ckpt`.
- Checkpointing guidance now matches the trainer and export script behavior.
- Publication gate language remains aligned with `DATA_LICENSE.md`.

## Unresolved Risks

- Empirical Kaggle P100 and T4x2 benchmark numbers are still pending.
- Final validation still depends on Kaggle/Linux runtime execution for authoritative GPU timing and memory behavior.
- Public checkpoint release remains conditional on license-chain review approval.

## Post-v1 Backlog

- Broader Stage 2 expansion beyond smoke scope.
- TPU work and future branch support.
- Public-release automation beyond the current publication gate.
