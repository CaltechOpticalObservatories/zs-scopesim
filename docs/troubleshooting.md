# Troubleshooting

## PALACE Data Import Errors

If PALACE reports missing `palace.data` or `palace.config`, reinstall the local
PALACE package:

```bash
pip install -e ./PALACE
zs-sim doctor
```

## Editable Checkout Drift

If a notebook result differs between machines, compare `zs-sim doctor` output.
Use `scripts/sync_stack.sh` to checkout the refs recorded in the stack
manifest.
