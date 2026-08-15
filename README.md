# InterviewPilot

Adaptive Multi-Agent AI Interview System — LangChain + LangGraph + LangServe + FastAPI + OpenRouter.

## Architecture

Everything lives in a single `server.py`. Four real LangGraph nodes plus a human-in-the-loop
interrupt node:

```
START -> question_agent -> human_input -> evaluator_agent -> difficulty_agent
                                                                     |
                                                    +----------------+----------------+
                                                    |                                 |
                                             question_agent                    feedback_agent -> END
```

- **question_agent** — generates the next question (own prompt, own node)
- **human_input** — calls `interrupt()` to pause the graph and wait for the candidate's answer
- **evaluator_agent** — scores the answer (own prompt, own node)
- **difficulty_agent** — decides `strategy` / `next_difficulty` (own prompt, own node); the
  score→strategy rubric and the Easy/Medium/Hard bounds are enforced in plain Python as a
  safety net, so the LLM can never violate them
- **feedback_agent** — writes the final Markdown report after the last question
- Question counting (`current_question_number` vs `total_questions`) is decided by a plain
  Python conditional edge, never by the LLM — it is not possible for the graph to generate a
  4th question when `total_questions = 3`

State persistence and resume use `InMemorySaver` + `thread_id`. `POST /api/answer` resumes the
**same** graph execution with `Command(resume=answer)` — the graph is never restarted.

## Structured output

Each agent asks for structured Pydantic output via `prompt | llm.with_structured_output(Model)`.
Because free OpenRouter models don't all support native tool-calling reliably, there's a fallback
in `invoke_structured()`: if native structured output fails, it re-prompts with an explicit
"return only this JSON shape" instruction, parses/validates the response, retries up to twice,
and — only if every attempt fails — returns a deterministic default so one bad model response
never crashes the interview.

## LangServe

`add_routes()` exposes the graph at `/interview`, but wrapped behind a narrow
`LangServeStartInput` Pydantic model (`target_role`, `interview_type`, `selected_difficulty`,
`total_questions`) instead of the raw internal `InterviewState` TypedDict. This keeps the
auto-generated LangServe Playground from asking for internal fields like `question_history` or
`evaluation_history`. The actual candidate-facing UI never touches `/interview` — it only calls
`/api/start` and `/api/answer`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your OpenRouter key
python server.py
```

Open: **http://localhost:8080**

## Environment variables

```
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
MODEL_NAME=openrouter/free
```

`openrouter/free` is OpenRouter's auto-router — it currently picks a live free model that
supports the capabilities the request needs, so it keeps working as the free catalog rotates.
If you want a fixed model instead, set `MODEL_NAME` to something like
`meta-llama/llama-3.3-70b-instruct:free` or `openai/gpt-oss-20b:free`.
**Always double-check current free/paid status at https://openrouter.ai/models before
relying on any model name — the free catalog changes without notice.**

## Testing a full 3-question interview

1. Open http://localhost:8080
2. Fill in Target Role (e.g. "AI Engineer"), Interview Type ("Technical"), Starting Difficulty
   ("Easy"), Number of Questions (3)
3. Click **Start Interview** — Question 1 appears
4. Type an answer, click **Submit Answer** — see the score/feedback card, then Question 2 loads
   automatically after ~1s
5. Repeat for Question 2 and Question 3
6. After the 3rd answer, the **Final Interview Report** panel replaces the interview panel

You can also drive it directly:

```bash
curl -X POST http://localhost:8080/api/start \
  -H "Content-Type: application/json" \
  -d '{"target_role":"AI Engineer","interview_type":"Technical","selected_difficulty":"Easy","total_questions":3}'

curl -X POST http://localhost:8080/api/answer \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<paste thread_id from above>","answer":"My answer..."}'

curl http://localhost:8080/api/state/<thread_id>
```

Other endpoints: `GET /docs` (Swagger), `GET /health`, `GET /interview/playground` (LangServe
playground — dev/demo tool only, not the candidate UI).

## Notes

- No API key is ever hardcoded or returned in error messages.
- `/api/start`, `/api/answer` return structured JSON errors (400 for bad input, 404 for unknown
  `thread_id`, 502 if the LLM/graph call itself fails) instead of crashing the process.
- Two interviews never share state — each gets its own UUID `thread_id`.