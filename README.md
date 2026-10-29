# chat-sentinel

Near-realtime moderation for short, informal chat messages in Rioplatense Spanish. CPU only, fully
local: messages never leave the server and no external AI API is called.

Moderation never blocks sending. Your backend publishes the message, hands it to chat-sentinel, and
gets a webhook a few seconds later if the message should be reviewed or hidden.

```
backend ──POST /messages──▶ api ──▶ Redis stream ──▶ worker (micro-batches: 32 msgs or 2 s)
   ▲                                                   │ 1. rules (regex, wordlists)      ~0 ms
   │                                                   │ 2. encoders on the whole batch   ONNX int8
   └──────── webhook (HMAC-signed) ◀── flag / hide ────┤ 3. LLM for doubtful cases (off in the POC)
                                                       ▼
                                                   SQLite (messages, verdicts, reviews)
```

## Verdicts

| verdict | meaning |
|---|---|
| `ok` | nothing to do |
| `flag` | goes to the review queue, message stays visible |
| `hide` | hide it; it also goes to the review queue |

Guiding rule: **when in doubt, flag, never hide.** False positives are costly in a professional
setting, so `hide` needs a `hide` rule (slurs) or a very high encoder risk (≥ 0.98), and the LLM can
only flag by default (`llm.can_hide: false`).

## Pipeline

1. **Rules** (`config.yaml` → `rules`): phones, emails (also "arroba … punto com"), social handles,
   contact requests ("pasame tu wsp"), external links (meeting/docs domains allowed), repeated
   messages, spam phrases, slurs, threats, strong insults. Each rule has `action: flag | hide | ignore`.
   Everyday words like "boludo" are `ignore`: recorded and passed to the LLM as hints, never acted on.
2. **Encoders**, on the whole batch, combined into a `risk_score` (noisy-OR of weighted scores):
   - [`pysentimiento/robertuito-hate-speech`](https://huggingface.co/pysentimiento/robertuito-hate-speech) — `hateful` probability, weight 1.0
   - [`textdetox/xlmr-base-toxicity-classifier`](https://huggingface.co/textdetox/xlmr-base-toxicity-classifier) — `toxic` probability, weight 0.9
   - [`pysentimiento/robertuito-sentiment-analysis`](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis) — `NEG` probability, weight 0.25
3. **Doubtful band** (`llm_min ≤ risk ≤ llm_max`): the message is flagged with `doubtful: true`.
   **The POC runs without an LLM** (`llm.enabled: false`): doubtful messages go to the review queue,
   and each human review labels them. That measures how many doubtful cases real traffic has before
   paying ~2 GB of RAM and seconds per message for an LLM:

   ```sql
   SELECT doubtful, verdict, COUNT(*) FROM verdicts GROUP BY doubtful, verdict;
   -- how reviewers resolved the doubtful ones
   SELECT r.final_verdict, COUNT(*) FROM verdicts v JOIN reviews r ON r.message_key = v.message_key
   WHERE v.doubtful = 1 GROUP BY r.final_verdict;
   ```

   With `llm.enabled: true`, the doubtful ones go to `qwen3:1.7b` through Ollama with thinking disabled
   (`think: false`), the last 4 messages of the conversation as context and a JSON-schema-constrained
   answer `{verdict, category, reason}`. If Ollama fails, the message stays `flag`.

Model revisions are pinned in `config.yaml`. Choices checked on 2026-10-28, sized for a VPS that
shares its 4 vCPU / 8 GB with another production stack:

- Toxicity uses XLM-R **base**. The larger `textdetox/xlmr-large-toxicity-classifier-v2` (2025
  dataset, Spanish F1 0.71) is a config swap away when memory allows.
- `qwen3:1.7b` (1.4 GB) instead of `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB, better with irony and
  friendly insults). Qwen3 thinks before answering by default, which costs tens of seconds on CPU,
  hence `think: false`. `qwen3.5:2b` (3.1 GB) and `qwen3.5:4b` (4.0 GB) do not fit.
- Llama-Guard-3-1B was considered. Its output is a fixed `safe` / `unsafe S<n>` taxonomy, not our
  schema, so it would need an adapter. Not implemented.

> **License caveat.** pysentimiento states that its models are trained on third-party datasets "many
> of which are for non-commercial use" (the sentiment model uses TASS). Review those licenses before
> using the RoBERTuito models in a commercial deployment. The XLM-R model is OpenRAIL++ and Qwen3 is
> Apache 2.0.

## Quick start (local)

Requirements: Docker with ~4 GB of memory free for containers.

```bash
cp .env.example .env
./scripts/download_models.sh          # ~2 GB: encoders -> ONNX fp32 + int8 (+ the Ollama model if llm.enabled)
docker compose up -d --build

# Create a client (its webhook) and an API key. Both secrets are printed once.
docker compose exec api python -m sentinel.cli create-client sanjuan --webhook-url https://backend.example/hooks/moderation
docker compose exec api python -m sentinel.cli create-key sanjuan
```

Send a message and read the verdict (the API listens on `127.0.0.1:8100`, `SENTINEL_PORT` in `.env`):

```bash
KEY=sk_...
curl http://127.0.0.1:8100/messages -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"message_id":"m-1","conversation_id":"c-1","author_id":"u-1","text":"pasame tu wsp","created_at":"2026-10-28T12:00:00Z"}'
curl http://127.0.0.1:8100/moderation/m-1 -H "Authorization: Bearer $KEY"
```

Load test through the real queue (from inside the network):

```bash
docker compose exec -e SENTINEL_API_KEY=$KEY api python scripts/simulate_chat.py --rate 5 --count 200
```

## API

Every endpoint except `/healthz` needs `Authorization: Bearer <api key>`. A key belongs to one client
and only sees that client's messages.

| method | path | scope | |
|---|---|---|---|
| `POST` | `/messages` | `messages:write` | `{message_id, conversation_id, author_id, text, created_at}` → `202 {status: queued \| duplicate}` |
| `GET` | `/moderation/{message_id}` | `moderation:read` | `status: pending \| done`, verdict, category, reason, risk, scores, rule hits, latency, latest review |
| `GET` | `/review-queue?limit=50` | `moderation:read` | flagged/hidden messages without a human review, oldest first |
| `POST` | `/review/{message_id}` | `review:write` | `{decision: confirm \| reject, final_verdict?, category?, reviewer_id?, notes?}` |

`message_id` is yours and idempotent per client: resending it returns `duplicate` and is not
re-moderated. Errors: `401` bad key, `403` missing scope, `404` unknown message (or another client's),
`429` over `api.rate_limit_per_minute`.

### API keys

```bash
docker compose exec api python -m sentinel.cli create-key sanjuan --scopes messages:write,moderation:read
docker compose exec api python -m sentinel.cli list-keys
docker compose exec api python -m sentinel.cli revoke-key <key-id>
```

Keys look like `sk_…`; only their SHA-256 is stored. Give each backend the narrowest scopes it needs:
the chat service usually needs `messages:write` only.

### Webhook

Sent for `flag` and `hide` verdicts, up to 3 attempts:

```json
{"event": "message.flagged", "message_id": "m-1", "conversation_id": "c-1", "verdict": "flag",
 "category": "contact_sharing", "reason": "rule contact_request: pasame tu wsp", "risk_score": null, "stage": "rules"}
```

Verify it with the client's webhook secret. Reject requests whose timestamp is older than ~5 minutes:

```ts
const expected = "sha256=" + createHmac("sha256", secret).update(`${timestamp}.${rawBody}`).digest("hex");
// timestamp = header X-Sentinel-Timestamp, compare with X-Sentinel-Signature using timingSafeEqual
```

## Configuration

Everything lives in `config.yaml` (mounted read-only; restart `worker` and `api` after editing):
thresholds, batch size and window, context size, models and weights, encoder runtime and threads,
LLM model and prompt platform text, webhook retries and rules.

## Evaluation and benchmarks

`data/test_messages.jsonl` holds 96 synthetic, labeled Rioplatense messages in 20 conversations:
coordination, hard but valid feedback, friendly banter, insults, harassment, threats, slurs, spam and
contact sharing.

Stop the worker first on the VPS: a second copy of the encoders does not fit next to it.

```bash
docker compose stop worker
docker compose run --rm worker python scripts/eval.py              # full pipeline -> reports/eval-onnx-int8.md
docker compose run --rm worker python scripts/eval.py --no-llm     # rules + encoders only
# with llm.enabled: true and `docker compose --profile llm up -d ollama`:
docker compose exec ollama ollama pull qwen3:4b-instruct-2507-q4_K_M   # to compare against the bigger LLM
docker compose run --rm worker python scripts/eval.py --llm-model qwen3:4b-instruct-2507-q4_K_M
docker compose --profile tools run --rm tools python scripts/bench_encoders.py   # torch vs onnx vs onnx-int8
docker compose --profile tools run --rm tools pytest
```

## Deploying to the VPS

Target: 4 vCPU (AVX2, no VNNI), 8 GB RAM, no swap, 100 GB SSD, no GPU, Ubuntu 24.04, **shared with
another production stack** (~1.7 GB used) behind a host nginx on 80/443.

1. **Take the VPS snapshot first.** The plan allows a single snapshot; use it before the first deploy
   so you can roll back the whole machine, production included.
2. `sudo ./scripts/setup_server.sh` adds a 2 GB swap file if there is none (safety net against the
   OOM killer picking the other stack's database). It does not touch nginx or the firewall.
3. `./scripts/download_models.sh` (a few minutes at 200 Mbit/s), then `docker compose up -d --build`.
4. Add the nginx site from `deploy/nginx.conf.example`, reload nginx and run certbot for the domain.
5. Create a client and an API key as above.

The API is published on `127.0.0.1:8100` only; nginx is the public entry. Redis and Ollama live on
the compose network. Models live in `./models` (a bind mount), so rebuilds never download them again.

### Resource budget (container limits)

| service | memory | CPU | holds |
|---|---|---|---|
| ollama (only with `--profile llm`) | 2 GB | vCPU 2-3 | Qwen3 1.7B Q4_K_M (~1.4 GB) + 2k context, kept loaded |
| worker | 1.25 GB | vCPU 2-3 | three ONNX int8 encoders + runtime |
| api | 256 MB | vCPU 2-3 | FastAPI |
| redis | 128 MB | any | stream, rate-limit counters |

About 1.6 GB without the LLM (3.6 GB with it) of the 6.1 GB available. The worker and Ollama never compute
at the same time (the worker waits for each LLM answer), so they share the same two pinned vCPUs with
2 threads each, and production always keeps vCPUs 0-1 free of moderation load.

## Human reviews as training data

Every `POST /review` is stored in `reviews` with the model verdict and the human verdict, ready to
export for a future SetFit fine-tune:

```sql
SELECT m.text, r.final_verdict, r.category FROM reviews r JOIN messages m ON m.id = r.message_key;
```

## Out of scope

Admin UI, end-user auth, fine-tuning, horizontal scaling (one worker; SQLite).
