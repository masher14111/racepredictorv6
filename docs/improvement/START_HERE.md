# Start here — 24 prompts with shared memory

Original steps01-18 have recorded dispositions. The user authorized follow-up19-24 on 2026-09-19;
see [Daily paper follow-up](DAILY_PAPER_FOLLOWUP.md). Read STATE.md for actual execution progress.
For automatic CLI execution with terminal updates, use [Automatic runner](AUTOMATIC_RUNNER.md).
The current manifest contains18 original work orders and6 follow-up work orders.
Prompts01..10 establish a trustworthy baseline. Optional research follows, then readiness and paper collection.
New races and settled observations still take real elapsed time; completing code cannot complete an eight-week trial.

## How to use this
1. Open a fresh Claude Code or Codex chat with this repository available.
2. Choose the model and thinking level shown above the selected prompt.
3. Open [NEXT_PROMPT.md](../../memory/improvement/NEXT_PROMPT.md) and paste its complete text block.
4. Let it implement, verify and update memory before switching chats.
5. For the next chat, reopen the newly updated NEXT_PROMPT.md; do not reuse a stale copy.
6. The reusable [NEW_CHAT.md](NEW_CHAT.md) starter can also resume the next ready step from memory.

Actual Git root: C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
The parent folder has entrypoint files directing assistants into this nested repository.
The selected model/effort is an application setting; writing “High” in text does not guarantee changing it.
Settings are recommendations, not prerequisites: a capable available model can complete the same work order.
Use Extra high/xhigh for the two independent audits; avoid Max by default.
Low is suitable only for simple follow-up formatting/handoff work, not these correctness-sensitive stages.
If Fable5.1 is absent, use Opus5 at Extra high for step17; keep the exact requested role.

## Prompt sequence
| Prompt | Work | Recommended model | Thinking |
|---|---|---|---|
| [01](prompts/01-checkpoint-and-contracts.md) | Checkpoint the project and agree data/evaluation contracts | GPT-6 Astra | Medium |
| [02](prompts/02-whole-race-splits.md) | Repair chronological splits throughout training and tuning | GPT-6 Astra | High |
| [03](prompts/03-canonical-markets.md) | Preserve WIN and PLACE market association when fusing data | GPT-6 Astra | High |
| [04](prompts/04-point-in-time-features.md) | Repair independent features and historical feature invariance | GPT-6 Astra | High |
| [05](prompts/05-current-declarations.md) | Acquire current race declarations with source provenance | Claude Sonnet 5 | High |
| [06](prompts/06-coverage-and-results-refresh.md) | Refresh results and repair measurable feature coverage | Claude Sonnet 5 | High |
| [07](prompts/07-as-of-odds-capture.md) | Verify timestamped odds capture and executable-price replay | GPT-6 Astra | High |
| [08](prompts/08-settlement-and-forward-report.md) | Repair settlement identity and forward-report accounting | GPT-6 Astra | High |
| [09](prompts/09-independent-foundation-review.md) | Independently audit the repaired data and evaluation foundations | Claude Opus 5 | Extra high (xhigh) |
| [10](prompts/10-corrected-frozen-baselines.md) | Build comparable baselines and freeze the experiment protocol | GPT-6 Astra | High |
| [11](prompts/11-speed-and-sectionals.md) | Pilot measured speed, sectionals and normalized performance | Claude Opus 5 | High |
| [12](prompts/12-market-blend-and-calibration.md) | Learn market combination and calibrate race probabilities | Claude Opus 5 | High |
| [13](prompts/13-ensemble-and-uncertainty.md) | Test XGBoost as a third numerical model and measure ensemble gain | GPT-6 Astra | High |
| [14](prompts/14-text-extraction-and-archive.md) | Repair text extraction and preserve dated source comments | Claude Sonnet 5 | Medium |
| [15](prompts/15-local-llm-benchmark.md) | Benchmark local and hosted DeepSeek extraction against repaired regex | Claude Sonnet 5 | High |
| [16](prompts/16-text-feature-ablation.md) | Measure whether dated text features improve future forecasts | GPT-6 Astra | High |
| [17](prompts/17-integrated-readiness-review.md) | Review the integrated candidate for paper operation | Claude Fable 5.1 | Extra high (xhigh) |
| [18](prompts/18-paper-forward-start.md) | Start frozen paper collection and hand over daily operation | GPT-6 Astra | Medium |
| [19](prompts/19-deepseek-shadow-integration.md) | Integrate capped DeepSeek daily shadow extraction | Claude Sonnet 5 | High |
| [20](prompts/20-daily-identity-cutoff-provenance.md) | Repair identity, cutoff and provenance | Claude Sonnet 5 | High |
| [21](prompts/21-paper-stakes-settlement-observations.md) | Repair stakes, settlement and PASS observations | Claude Sonnet 5 | High |
| [22](prompts/22-daily-source-health.md) | Make current declarations and source health trustworthy | Claude Sonnet 5 | High |
| [23](prompts/23-complete-shadow-bundle.md) | Build complete shadow bundle and freeze fresh evaluation | Claude Sonnet 5 | High |
| [24](prompts/24-fresh-paper-readiness-audit.md) | Audit readiness and hand off daily operation | Claude Opus 5 | Extra high (xhigh) |

## Memory files
| File | Purpose |
|---|---|
| [AGENTS.md](../../AGENTS.md) | Shared instructions for all assistants |
| [CLAUDE.md](../../CLAUDE.md) | Compact Claude entrypoint |
| [DESIGN.md](../../DESIGN.md) | Architecture and preserved UI contract |
| [STATE.md](../../memory/improvement/STATE.md) | Current status, evidence, unresolved gaps and next step |
| [DECISIONS.md](../../memory/improvement/DECISIONS.md) | Durable choices and their reasons |
| [HANDOFF.md](../../memory/improvement/HANDOFF.md) | Immediate resume instructions |
| [CHAT_CONTEXT.md](../../memory/improvement/CHAT_CONTEXT.md) | Generated portable context for a chat without shared history |
| [NEXT_PROMPT.md](../../memory/improvement/NEXT_PROMPT.md) | Generated next work order with model/effort |
| [Stage notes](../../memory/improvement/stages/) | Dated evidence per step; each below 200 lines |

Every active memory file must contain fewer than 200 physical lines; the checker enforces 199 maximum.
The [full original UI design](../archive/memory-setup-20260918/DESIGN.md) is preserved as a reference archive.
Long logs/reports and the [all-prompts collection](ALL_PROMPTS.md) are references, not compact startup memory.
The historical root HANDOFF.md and memory/MEMORY.md entries remain available, but may describe superseded behavior.

## Chats without repository access
A web chat does not automatically remember local files. Paste the latest CHAT_CONTEXT.md and the selected
prompt; attach requested source files as needed. Such a chat cannot edit or verify your local project merely
from these notes. Return its proposed changes through an assistant with repository access and update the
canonical memory there. For direct implementation, fresh Claude Code/Codex chats in this project are simpler.

## Progress and parallel work
Default: run prompts sequentially and finish their checks before moving on.
Step14 text work may run alongside modelling after its dependencies, using separate code/output ownership.
A reviewer may inspect a frozen snapshot while another worker changes unrelated files.
Do not have two chats edit shared STATE/HANDOFF or train into the same model/report paths.
Only the integration owner updates shared memory after merging a parallel worker's dated stage note.
An isolated worktree can lack ignored raw data/credentials; verify inputs without exposing secrets.
A DEFERRED_DATA optional step remains visible; it does not count as validated or force other experiments on.
Required correctness failures return to the owning stage; do not paper over them to reach step18.
Step18 may finish collection setup while model verdict stays NO-GO and forward evidence stays pending.

## Memory verification
Run with the existing project Python:
```powershell
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
```
Each numbered prompt already requires these commands before handoff.
The exporter reads STATE's Next step and generates a fresh prompt/context; it does not advance status.
The checker verifies line limits, prompt labels, dependency order, state consistency and original archives.

See [model settings and official sources](MODELS.md), [research review](../research/2026-09-18-predictor-review.md),
and [setup verification](VERIFICATION.md).
