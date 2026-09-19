# Recommended models and thinking settings
Checked 2026-09-18. Assignments are project judgments, not comparative benchmark claims.
Select the model and effort in the app before pasting; prompt text is not a selector API.

| Model | Assigned stages | Thinking |
|---|---|---|
| Codex — GPT-6 Astra | 01,02,03,04,07,08,10,13,16,18 | High; Medium for01 and18 |
| Claude Sonnet 5 | 05,06,14,15 | High; Medium for14 |
| Claude Opus 5 | 09,11,12 | High; Extra high/xhigh for09 |
| Claude Fable 5.1 | 17 | Extra high/xhigh |

- Codex owns most code integration and reproducible checks.
- Sonnet handles bounded source/text components; existing interfaces and tests define completion.
- Opus handles methodology and one independent foundation review.
- Fable provides a second independent integrated review; use Opus5 if Fable5.1 is unavailable.
- Each prompt includes a practical fallback; unavailable branding must not stop an otherwise capable setup.
- High is the default for core correctness. Medium covers bounded work; Low is only for routine follow-up notes.
- Reserve Extra high for the two audits. Max is not needed by default.
- The effort scale is model-specific: High is not identical compute across providers.
- Claude Code supports /effort; choose the actual displayed option if another client labels it differently.
- These settings do not concern the local Qwen/Hermes extractor, which is a separate research component.

## Verified sources
- [Claude models](https://platform.claude.com/docs/en/models/fable-5-1/overview)
- [Claude effort controls](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Claude Code model configuration](https://code.claude.com/docs/en/model-config)
- [OpenAI reasoning effort](https://developers.openai.com/api/docs/guides/reasoning)
- [Codex project instructions](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Codex worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees)

Official Claude documentation checked for this pack documents Opus5, Sonnet5 and Fable5.1 with
low/medium/high/xhigh/max effort. This Codex app explicitly exposes High and Medium for GPT-6 Astra.
Choose an available supported setting; account access and picker contents can differ.
