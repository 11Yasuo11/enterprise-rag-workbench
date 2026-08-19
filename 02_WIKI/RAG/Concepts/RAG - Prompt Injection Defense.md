# RAG - Prompt Injection Defense

Retrieved text は untrusted DATA として扱い、system policy として解釈しない。

## Controls

| Control | Behavior |
|---|---|
| Prompt-injection boundary | Retrieved text ≠ system instructions |
| ACL filtering | Before ranking |
| Support validation | Judge supporting IDs must be in authorized Top-5 |
| Content identity validation | Supporting chunk text must match stored content |

## Measured

Final V2: prompt-injection boundary 100% (4/4 cases).

Test corpus includes `security-training-untrusted.md` with injection patterns.

## Related

- [[RAG - Safety Pipeline]]
- [`README.md`](../../../README.md) — Security section
