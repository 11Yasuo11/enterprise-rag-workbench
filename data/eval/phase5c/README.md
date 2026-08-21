# V3 Clean 120-Case Dataset Draft

Dataset ID: `acmeai-enterprise-rag-v3-clean-e2e-final-120`

- Cases: 120
- Answerable: 102
- Should abstain: 18
- Dataset SHA-256: `595d8033ab059ce110696b938109afcf8a4c4716c0ae9ab23df1637464352d64`
- Source audit ZIP SHA-256: `8f776dbd4f5e1c8686161db133785777433643a8c6e5d145226df4d8c8c49efe`

## Distribution

```json
{
  "single_document": 4,
  "multidoc_two": 20,
  "multidoc_three": 30,
  "near_duplicate": 16,
  "same_document_multi_chunk": 10,
  "exact_identifier": 8,
  "version_region": 8,
  "semantic_paraphrase": 6,
  "acl_sensitive": 4,
  "partial_no_answer": 4,
  "prompt_injection": 10
}
```

## QA already applied

- Every answerable case has explicit required facts and required chunk IDs.
- Required facts were checked against the exported chunk text.
- Two-document cases require exactly two distinct documents.
- Three-document cases require exactly three distinct documents.
- Same-document multi-chunk cases require at least two chunks from one document.
- Exact-identifier questions do not contain the expected identifier in the question text.
- ACL, partial/no-answer, and prompt-injection cases are labeled `should_abstain=true`.
- The largest simple token-Jaccard overlap with the previous 120 questions was 0.471, below 0.50.

## Important

This is a **dataset draft for final validation**, not automatically a valid unseen benchmark.
Before any inference, Cursor must run the repository's canonical independence checker and a second dataset-QA gate. If either fails, stop before results are visible.
