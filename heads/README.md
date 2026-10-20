# The head weights

The trained head is **626M**, too large for git, so it ships as a release asset
rather than a committed file. It is attached to release **`v1.0.0`**. Place it in
this directory, beside `draft_head.config.json`.

While the repository is private, the plain download link will not work without
credentials -- use the API form below, or copy the file over directly.

```bash
# public repo:
curl -L -o heads/draft_head.pt \
  https://github.com/importZL/AIBuildAI-SpecDec/releases/download/v1.0.0/draft_head.pt

# private repo (token needs Contents: read):
curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/octet-stream" \
     -o heads/draft_head.pt \
  "$(curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" \
      https://api.github.com/repos/importZL/AIBuildAI-SpecDec/releases/tags/v1.0.0 \
      | python3 -c 'import json,sys; print(next(a["url"] for a in json.load(sys.stdin)["assets"] if a["name"]=="draft_head.pt"))')"
```

```
file    draft_head.pt
size    655398915 bytes
sha256  7a744703b50f49f9d5cb6ad04256e6a5ee81746b70d4426e3bd900980e371343
```

Then verify it:

```bash
sha256sum -c <<'SUMS'
7a744703b50f49f9d5cb6ad04256e6a5ee81746b70d4426e3bd900980e371343  draft_head.pt
SUMS
```

## Geometry

The companion `draft_head.config.json` must sit beside the `.pt` with the
same stem -- `run_eval.sh` reads `${HEAD%.pt}.config.json` to recover the draft
tree. Both files are needed.

```json
{
  "n_layers": 1,
  "inter": 13824,
  "hidden": 5120,
  "n_heads": 40,
  "kv_heads": 8,
  "head_dim": 128,
  "rope_theta": 1000000.0,
  "sglang_overrides": {
    "speculative_num_steps": 4,
    "speculative_eagle_topk": 24,
    "speculative_num_draft_tokens": 64
  }
}
```

`inter 13824`, ~328M parameters, one layer, and the tree
`4 steps / 24 topk / 64 draft tokens`. See RECIPE.md for why those values.

## Provenance

Step 10 of a 10-step automated search over self-distilled data from
`Qwen/Qwen2.5-14B-Instruct`. Internal harness score 2.2116x; MT-Bench 2.0089x.
The weights contain no training data -- they are a 1-layer draft head over the
target's hidden states, and the target's own embedding and LM head are reused at
load time rather than copied in.
