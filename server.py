"""
InterviewPilot - Adaptive Multi-Agent AI Interview System
============================================================
LangChain + LangGraph + LangServe + FastAPI + OpenRouter

Four real LangGraph agent nodes:
  1. question_agent   - generates the next interview question
  2. human_input       - pauses the graph (interrupt) and waits for the candidate's answer
  3. evaluator_agent   - scores the answer
  4. difficulty_agent  - decides next difficulty / strategy
  5. feedback_agent    - produces the final report

Run:
    pip install -r requirements.txt
    python server.py

Open:
    http://localhost:8080
"""

import json
import os
import re
import uuid
import logging
from typing import Optional, Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, ValidationError

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from langserve import add_routes
                
# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("interviewpilot")

# ============================================================
# ENVIRONMENT / LLM SETUP  (Google Gemini)
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env. "
        "Please provide your Google Gemini API key."
    )

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    max_retries=2,
)

DIFFICULTIES = ["Easy", "Medium", "Hard"]

# ============================================================
# PYDANTIC STRUCTURED OUTPUT MODELS
# ============================================================


class QuestionResult(BaseModel):
    question: str = Field(description="The interview question text.")
    topic: str = Field(description="Short topic/skill area this question covers.")


class EvaluationResult(BaseModel):
    score: int = Field(ge=0, le=10, description="Overall score from 0 to 10.")
    correctness: str = Field(description="Brief assessment of factual correctness.")
    relevance: str = Field(description="Brief assessment of relevance to the question.")
    depth: str = Field(description="Brief assessment of depth of understanding.")
    clarity: str = Field(description="Brief assessment of communication clarity.")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    weak_topics: list[str] = Field(default_factory=list)
    feedback: str = Field(description="1-3 sentence actionable feedback for the candidate.")


class DifficultyDecision(BaseModel):
    performance_level: Literal["strong", "average", "weak"]
    next_difficulty: Literal["Easy", "Medium", "Hard"]
    strategy: Literal["increase_difficulty", "maintain_difficulty", "target_weakness", "follow_up"]
    reason: str = Field(description="One sentence explanation of the decision.")


# ============================================================
# ROBUST STRUCTURED OUTPUT HELPER
# ============================================================
#
# Free OpenRouter models don't all reliably support native tool-calling based
# structured output. We try llm.with_structured_output() first (works cleanly
# on models that support it), and fall back to an explicit "return JSON only"
# prompt + manual parse + Pydantic validation + retry if that fails. A single
# malformed model response never crashes the interview - after N retries we
# fall back to a safe deterministic default.


def _extract_json_block(text: str) -> str:
    """Pull the first {...} JSON object out of a raw LLM text response."""
    text = text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    # Otherwise find the first balanced-looking { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def invoke_structured(
    prompt: ChatPromptTemplate,
    variables: dict,
    model: type[BaseModel],
    fallback: BaseModel,
    max_retries: int = 2,
):
    """
    Invoke `prompt` against the shared `llm`, coercing the response into
    `model`. Tries native structured output first, falls back to
    instructed-JSON + manual parsing, and finally returns `fallback` if every
    attempt fails so a single bad model response never crashes the interview.
    """
    # --- Attempt 1: native structured output ---
    try:
        chain = prompt | llm.with_structured_output(model)
        result = chain.invoke(variables)
        if isinstance(result, model):
            return result
        if isinstance(result, dict):
            return model.model_validate(result)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a fallback boundary
        logger.warning("Native structured output failed (%s), falling back to JSON mode.", exc)

    # --- Attempt 2..N: explicit JSON instruction + manual parse ---
    schema_hint = json.dumps(model.model_json_schema().get("properties", {}), indent=2)
    json_instruction = (
        "\n\nRespond with ONLY a single valid JSON object matching this shape "
        f"(no markdown, no commentary, no code fences):\n{schema_hint}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            messages = prompt.invoke(variables).to_messages()
            messages[-1].content = str(messages[-1].content) + json_instruction
            response = llm.invoke(messages)
            raw = _extract_json_block(str(response.content))
            data = json.loads(raw)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
            logger.warning("JSON-mode structured output attempt %s/%s failed: %s", attempt, max_retries, exc)

    logger.error("All structured output attempts failed for %s. Using deterministic fallback.", model.__name__)
    return fallback


# ============================================================
# INTERVIEW STATE
# ============================================================


class InterviewState(TypedDict):
    target_role: str
    interview_type: str
    selected_difficulty: str
    current_difficulty: str
    total_questions: int
    current_question_number: int
    current_question: str
    current_topic: str
    current_answer: str
    current_evaluation: Optional[dict]
    current_strategy: str
    question_history: list[str]
    answer_history: list[str]
    evaluation_history: list[dict]
    weak_topics: list[str]
    difficulty_history: list[str]
    interview_complete: bool
    final_report: str


def make_initial_state(target_role: str, interview_type: str, selected_difficulty: str, total_questions: int) -> InterviewState:
    return InterviewState(
        target_role=target_role,
        interview_type=interview_type,
        selected_difficulty=selected_difficulty,
        current_difficulty=selected_difficulty,
        total_questions=total_questions,
        current_question_number=0,
        current_question="",
        current_topic="",
        current_answer="",
        current_evaluation=None,
        current_strategy="maintain_difficulty",
        question_history=[],
        answer_history=[],
        evaluation_history=[],
        weak_topics=[],
        difficulty_history=[],
        interview_complete=False,
        final_report="",
    )


# ============================================================
# AGENT 1: QUESTION AGENT
# ============================================================

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "System Instruction:\n"
            "You are the Question Agent in a multi-agent adaptive interview system. "
            "You generate one realistic, practical interview question at a time. "
            "CRITICAL: You MUST NEVER repeat or rephrase a previously asked question. "
            "Respect the target role, interview type, and difficulty exactly. "
            "Make each question distinct and cover different aspects of the required skills.",
        ),
        (
            "human",
            "Target role: {target_role}\n"
            "Interview type: {interview_type}\n"
            "Current difficulty: {current_difficulty}\n"
            "Strategy for this question: {current_strategy}\n"
            "Weak topics identified so far: {weak_topics}\n"
            "Previously asked questions (DO NOT REPEAT THESE):\n{previous_questions}\n\n"
            "Strategy meaning:\n"
            "- increase_difficulty: ask a harder question than before\n"
            "- maintain_difficulty: ask a question at the same difficulty\n"
            "- target_weakness: focus the question on one of the weak topics above\n"
            "- follow_up: ask a related follow-up probing the previous topic more deeply\n\n"
            "Generate a highly unique, varied, and specific interview question now.",
        ),
    ]
)


def question_agent(state: InterviewState) -> InterviewState:
    fallback = QuestionResult(
        question=f"Tell me about a challenging {state['target_role']} problem you solved recently.",
        topic="general experience",
    )
    result: QuestionResult = invoke_structured(
        QUESTION_PROMPT,
        {
            "target_role": state["target_role"],
            "interview_type": state["interview_type"],
            "current_difficulty": state["current_difficulty"],
            "current_strategy": state["current_strategy"],
            "weak_topics": ", ".join(state["weak_topics"]) or "none yet",
            "previous_questions": "; ".join(state["question_history"]) or "none yet",
        },
        QuestionResult,
        fallback,
    )

    next_number = state["current_question_number"] + 1
    state["current_question_number"] = next_number
    state["current_question"] = result.question
    state["current_topic"] = result.topic
    state["question_history"] = state["question_history"] + [result.question]
    logger.info("Question %s generated: %s", next_number, result.question)
    return state


# ============================================================
# HUMAN-IN-THE-LOOP NODE
# ============================================================


def human_input_node(state: InterviewState) -> InterviewState:
    answer = interrupt(
        {
            "question_number": state["current_question_number"],
            "question": state["current_question"],
            "topic": state["current_topic"],
            "difficulty": state["current_difficulty"],
        }
    )
    state["current_answer"] = answer
    state["answer_history"] = state["answer_history"] + [answer]
    return state


# ============================================================
# AGENT 2: EVALUATOR AGENT
# ============================================================

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Evaluator Agent in a multi-agent adaptive interview system. "
            "Evaluate the candidate's answer strictly but fairly. Do not invent "
            "information the candidate did not provide. Base weak_topics only on "
            "genuine gaps you observed in this specific answer.",
        ),
        (
            "human",
            "Target role: {target_role}\n"
            "Interview type: {interview_type}\n"
            "Question: {question}\n"
            "Candidate's answer: {answer}\n\n"
            "Evaluate this answer now.",
        ),
    ]
)


def evaluator_agent(state: InterviewState) -> InterviewState:
    fallback = EvaluationResult(
        score=5,
        correctness="Unable to automatically verify correctness.",
        relevance="Answer addressed the question.",
        depth="Moderate depth.",
        clarity="Reasonably clear.",
        strengths=["Provided a response"],
        weaknesses=["Could not be automatically scored in detail"],
        weak_topics=[],
        feedback="Automated evaluation was unavailable; a moderate score was assigned by default.",
    )
    result: EvaluationResult = invoke_structured(
        EVALUATOR_PROMPT,
        {
            "target_role": state["target_role"],
            "interview_type": state["interview_type"],
            "question": state["current_question"],
            "answer": state["current_answer"] or "(no answer provided)",
        },
        EvaluationResult,
        fallback,
    )

    evaluation_dict = result.model_dump()
    state["current_evaluation"] = evaluation_dict
    state["evaluation_history"] = state["evaluation_history"] + [evaluation_dict]
    # Accumulate weak topics without duplicates
    merged_weak_topics = list(state["weak_topics"])
    for wt in result.weak_topics:
        if wt not in merged_weak_topics:
            merged_weak_topics.append(wt)
    state["weak_topics"] = merged_weak_topics
    logger.info("Evaluation complete: score=%s", result.score)
    return state


# ============================================================
# AGENT 3: DIFFICULTY AGENT
# ============================================================

DIFFICULTY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Difficulty Agent in a multi-agent adaptive interview system. "
            "You decide how the interview should adapt based on the candidate's last "
            "evaluation. Follow the scoring rubric exactly: "
            "score 8-10 = strong -> increase_difficulty (unless already at Hard, then "
            "maintain_difficulty); score 5-7 = average -> maintain_difficulty; "
            "score 0-4 = weak -> target_weakness if weak topics exist, otherwise follow_up. "
            "Never suggest a difficulty beyond Hard or below Easy.",
        ),
        (
            "human",
            "Current difficulty: {current_difficulty}\n"
            "Last evaluation score: {score}\n"
            "Last evaluation feedback: {feedback}\n"
            "Weak topics so far: {weak_topics}\n\n"
            "Decide the next difficulty and strategy now.",
        ),
    ]
)


def _bounded_next_difficulty(current: str, direction: Literal["up", "down", "same"]) -> str:
    idx = DIFFICULTIES.index(current) if current in DIFFICULTIES else 0
    if direction == "up":
        idx = min(idx + 1, len(DIFFICULTIES) - 1)
    elif direction == "down":
        idx = max(idx - 1, 0)
    return DIFFICULTIES[idx]


def difficulty_agent(state: InterviewState) -> InterviewState:
    evaluation = state["current_evaluation"] or {}
    score = evaluation.get("score", 5)

    # Deterministic default in case the LLM fallback path is also used.
    if score >= 8:
        default_strategy = "increase_difficulty"
        default_difficulty = _bounded_next_difficulty(state["current_difficulty"], "up")
        default_level = "strong"
    elif score >= 5:
        default_strategy = "maintain_difficulty"
        default_difficulty = state["current_difficulty"]
        default_level = "average"
    else:
        default_strategy = "target_weakness" if state["weak_topics"] else "follow_up"
        default_difficulty = state["current_difficulty"]
        default_level = "weak"

    fallback = DifficultyDecision(
        performance_level=default_level,
        next_difficulty=default_difficulty,
        strategy=default_strategy,
        reason="Deterministic rubric-based fallback decision.",
    )

    result: DifficultyDecision = invoke_structured(
        DIFFICULTY_PROMPT,
        {
            "current_difficulty": state["current_difficulty"],
            "score": score,
            "feedback": evaluation.get("feedback", ""),
            "weak_topics": ", ".join(state["weak_topics"]) or "none",
        },
        DifficultyDecision,
        fallback,
    )

    # SAFETY: enforce the rubric and difficulty bounds in Python regardless of
    # what the LLM said, so it can never break the deterministic rules.
    if score >= 8:
        strategy = "increase_difficulty"
        next_difficulty = _bounded_next_difficulty(state["current_difficulty"], "up")
    elif score >= 5:
        strategy = "maintain_difficulty"
        next_difficulty = state["current_difficulty"]
    else:
        strategy = "target_weakness" if state["weak_topics"] else "follow_up"
        next_difficulty = state["current_difficulty"]

    if next_difficulty not in DIFFICULTIES:
        next_difficulty = state["current_difficulty"]

    state["current_strategy"] = strategy
    state["current_difficulty"] = next_difficulty
    state["difficulty_history"] = state["difficulty_history"] + [next_difficulty]
    logger.info("Difficulty decision: strategy=%s next_difficulty=%s (llm reason: %s)", strategy, next_difficulty, result.reason)
    return state


# ============================================================
# CONDITIONAL ROUTER  (deterministic - never LLM controlled)
# ============================================================


def route_after_difficulty(state: InterviewState) -> str:
    if state["current_question_number"] >= state["total_questions"]:
        return "feedback_agent"
    return "question_agent"


# ============================================================
# AGENT 4: FEEDBACK AGENT
# ============================================================

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Feedback Agent in a multi-agent adaptive interview system. "
            "Write the final interview performance report in Markdown using exactly "
            "the section headers requested. Be specific, reference the actual questions "
            "and answers given, and do not expose any chain-of-thought reasoning.",
        ),
        (
            "human",
            "Target role: {target_role}\n"
            "Interview type: {interview_type}\n"
            "Questions asked: {questions}\n"
            "Answers given: {answers}\n"
            "Evaluations: {evaluations}\n"
            "Difficulty progression: {difficulty_history}\n"
            "Weak topics: {weak_topics}\n\n"
            "Write the final report now using this exact structure:\n"
            "# Interview Performance Report\n"
            "## Overall Assessment\n"
            "## Overall Score\n"
            "## Strengths\n"
            "## Weaknesses\n"
            "## Weak Topics\n"
            "## Question-by-Question Review\n"
            "## Difficulty Progression\n"
            "## Recommendations\n"
            "## Final Assessment",
        ),
    ]
)


def feedback_agent(state: InterviewState) -> InterviewState:
    try:
        chain = FEEDBACK_PROMPT | llm
        response = chain.invoke(
            {
                "target_role": state["target_role"],
                "interview_type": state["interview_type"],
                "questions": json.dumps(state["question_history"]),
                "answers": json.dumps(state["answer_history"]),
                "evaluations": json.dumps(state["evaluation_history"]),
                "difficulty_history": json.dumps(state["difficulty_history"]),
                "weak_topics": json.dumps(state["weak_topics"]),
            }
        )
        report = str(response.content).strip()
        if not report:
            raise ValueError("empty report")
    except Exception as exc:  # noqa: BLE001
        logger.error("Feedback agent failed (%s); generating deterministic fallback report.", exc)
        avg_score = (
            sum(e.get("score", 0) for e in state["evaluation_history"]) / len(state["evaluation_history"])
            if state["evaluation_history"]
            else 0
        )
        report = (
            "# Interview Performance Report\n\n"
            "## Overall Assessment\nReport generation encountered an issue; showing a summary instead.\n\n"
            f"## Overall Score\n{avg_score:.1f} / 10\n\n"
            "## Strengths\nSee individual question evaluations below.\n\n"
            "## Weaknesses\nSee individual question evaluations below.\n\n"
            f"## Weak Topics\n{', '.join(state['weak_topics']) or 'None identified'}\n\n"
            "## Question-by-Question Review\n"
            + "\n".join(
                f"- Q{i+1}: {q}\n  - Answer: {a}\n  - Score: {e.get('score', 'N/A')}"
                for i, (q, a, e) in enumerate(
                    zip(state["question_history"], state["answer_history"], state["evaluation_history"])
                )
            )
            + "\n\n## Difficulty Progression\n"
            + " -> ".join(state["difficulty_history"])
            + "\n\n## Recommendations\nReview weak topics listed above.\n\n"
            "## Final Assessment\nInterview completed."
        )

    state["final_report"] = report
    state["interview_complete"] = True
    logger.info("Final report generated (%s chars).", len(report))
    return state


# ============================================================
# BUILD THE GRAPH
# ============================================================

graph_builder = StateGraph(InterviewState)
graph_builder.add_node("question_agent", question_agent)
graph_builder.add_node("human_input", human_input_node)
graph_builder.add_node("evaluator_agent", evaluator_agent)
graph_builder.add_node("difficulty_agent", difficulty_agent)
graph_builder.add_node("feedback_agent", feedback_agent)

graph_builder.add_edge(START, "question_agent")
graph_builder.add_edge("question_agent", "human_input")
graph_builder.add_edge("human_input", "evaluator_agent")
graph_builder.add_edge("evaluator_agent", "difficulty_agent")
graph_builder.add_conditional_edges(
    "difficulty_agent",
    route_after_difficulty,
    {"question_agent": "question_agent", "feedback_agent": "feedback_agent"},
)
graph_builder.add_edge("feedback_agent", END)

checkpointer = InMemorySaver()
interview_graph = graph_builder.compile(checkpointer=checkpointer)

# ============================================================
# INTERVIEW ORCHESTRATION HELPERS  (used by both /api and LangServe)
# ============================================================


def _state_to_response(thread_id: str, values: dict, completed: bool) -> dict:
    return {
        "thread_id": thread_id,
        "status": "completed" if completed else "waiting_for_answer",
        "question_number": values.get("current_question_number", 0),
        "question": None if completed else values.get("current_question"),
        "difficulty": values.get("current_difficulty"),
        "evaluation": values.get("current_evaluation"),
        "strategy": values.get("current_strategy"),
        "final_report": values.get("final_report") if completed else None,
    }


def start_interview_logic(target_role: str, interview_type: str, selected_difficulty: str, total_questions: int) -> dict:
    if total_questions < 1 or total_questions > 20:
        raise ValueError("total_questions must be between 1 and 20.")
    if selected_difficulty not in DIFFICULTIES:
        raise ValueError(f"selected_difficulty must be one of {DIFFICULTIES}.")
    if not target_role.strip() or not interview_type.strip():
        raise ValueError("target_role and interview_type are required.")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = make_initial_state(target_role, interview_type, selected_difficulty, total_questions)

    try:
        interview_graph.invoke(initial_state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph execution failed on start.")
        raise RuntimeError(f"Interview graph failed to start: {exc}") from exc

    snapshot = interview_graph.get_state(config)
    return _state_to_response(thread_id, snapshot.values, completed=False)


def answer_interview_logic(thread_id: str, answer: str) -> dict:
    if not answer or not answer.strip():
        raise ValueError("answer must not be empty.")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = interview_graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise KeyError(f"No interview found for thread_id={thread_id}")
    if snapshot.values.get("interview_complete"):
        raise ValueError("This interview has already been completed.")

    try:
        interview_graph.invoke(Command(resume=answer), config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph execution failed on resume.")
        raise RuntimeError(f"Interview graph failed to resume: {exc}") from exc

    snapshot = interview_graph.get_state(config)
    completed = bool(snapshot.values.get("interview_complete"))
    return _state_to_response(thread_id, snapshot.values, completed=completed)


def get_state_logic(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = interview_graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise KeyError(f"No interview found for thread_id={thread_id}")
    return dict(snapshot.values)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="InterviewPilot", description="Adaptive Multi-Agent AI Interview System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    target_role: str
    interview_type: str
    selected_difficulty: str = "Easy"
    total_questions: int = 3


class AnswerRequest(BaseModel):
    thread_id: str
    answer: str


@app.post("/api/start")
def api_start(req: StartRequest):
    try:
        return start_interview_logic(req.target_role, req.interview_type, req.selected_difficulty, req.total_questions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/answer")
def api_answer(req: AnswerRequest):
    try:
        return answer_interview_logic(req.thread_id, req.answer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/state/{thread_id}")
def api_get_state(thread_id: str):
    try:
        return get_state_logic(thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "service": "InterviewPilot"}


# ============================================================
# LANGSERVE  (separate simple schema - never exposes InterviewState)
# ============================================================


class LangServeStartInput(BaseModel):
    target_role: str
    interview_type: str
    selected_difficulty: str = "Easy"
    total_questions: int = 3


def _langserve_start(payload: LangServeStartInput) -> dict:
    return start_interview_logic(
        payload.target_role, payload.interview_type, payload.selected_difficulty, payload.total_questions
    )


langserve_runnable = RunnableLambda(_langserve_start).with_types(input_type=LangServeStartInput, output_type=dict)

add_routes(
    app,
    langserve_runnable,
    path="/interview",
)

# ============================================================
# FRONTEND  (embedded HTML/CSS/JS)
# ============================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>InterviewPilot</title>
<style>
  :root {
    --bg-deep: #0b0e1a;
    --bg-panel: #131829;
    --bg-panel-2: #1a2036;
    --accent: #7c5cff;
    --accent-2: #4f8cff;
    --text: #e6e8f5;
    --text-dim: #8b93b8;
    --border: #2a3151;
    --good: #4ade80;
    --warn: #fbbf24;
    --bad: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: radial-gradient(circle at 20% 0%, #1c2444 0%, var(--bg-deep) 55%);
    color: var(--text);
    display: flex;
    justify-content: center;
    padding: 40px 16px;
  }
  .app { width: 100%; max-width: 760px; }
  header { text-align: center; margin-bottom: 32px; }
  header h1 {
    font-size: 2.1rem;
    margin: 0;
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  header p { color: var(--text-dim); margin-top: 6px; }
  .panel {
    background: linear-gradient(180deg, var(--bg-panel), var(--bg-panel-2));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
  }
  label { display: block; font-size: 0.85rem; color: var(--text-dim); margin: 16px 0 6px; }
  label:first-of-type { margin-top: 0; }
  input, select, textarea {
    width: 100%;
    background: #0e1324;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.95rem;
    outline: none;
    transition: border-color .15s;
  }
  input:focus, select:focus, textarea:focus { border-color: var(--accent); }
  textarea { min-height: 140px; resize: vertical; font-family: inherit; }
  button {
    margin-top: 22px;
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    color: #0b0e1a;
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    transition: transform .1s, opacity .15s;
  }
  button:hover { transform: translateY(-1px); }
  button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(124,92,255,0.15);
    color: var(--accent);
    border: 1px solid rgba(124,92,255,0.4);
  }
  .meta-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
  .progress-bar { height: 6px; border-radius: 999px; background: #0e1324; overflow: hidden; margin-bottom: 20px; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent-2), var(--accent)); transition: width .3s; }
  .question-box { font-size: 1.05rem; line-height: 1.5; margin-bottom: 18px; }
  .topic-tag { color: var(--text-dim); font-size: 0.8rem; margin-bottom: 8px; }
  .eval-card {
    margin-top: 24px;
    padding: 18px;
    border-radius: 12px;
    background: #0e1324;
    border: 1px solid var(--border);
  }
  .score { font-size: 1.6rem; font-weight: 700; }
  .score.good { color: var(--good); }
  .score.warn { color: var(--warn); }
  .score.bad { color: var(--bad); }
  .eval-card ul { margin: 6px 0 0; padding-left: 18px; color: var(--text-dim); font-size: 0.9rem; }
  .eval-card h4 { margin: 14px 0 4px; font-size: 0.85rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: .05em; }
  .report { white-space: pre-wrap; line-height: 1.6; font-size: 0.95rem; }
  .error-box {
    margin-top: 16px;
    padding: 12px 16px;
    border-radius: 10px;
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.4);
    color: var(--bad);
    font-size: 0.9rem;
  }
  .hidden { display: none; }
  .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(11,14,26,0.3); border-top-color: #0b0e1a;
    border-radius: 50%; animation: spin .7s linear infinite; margin-right: 8px; vertical-align: -3px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>&#129504; InterviewPilot</h1>
    <p>Adaptive Multi-Agent AI Interview System</p>
  </header>

  <div class="panel" id="startPanel">
    <label for="targetRole">Target Role</label>
    <input id="targetRole" type="text" placeholder="e.g. AI Engineer" value="AI Engineer" />

    <div class="row">
      <div>
        <label for="interviewType">Interview Type</label>
        <select id="interviewType">
          <option>Technical</option>
          <option>Behavioral</option>
          <option>System Design</option>
          <option>Mixed</option>
        </select>
      </div>
      <div>
        <label for="startDifficulty">Starting Difficulty</label>
        <select id="startDifficulty">
          <option>Easy</option>
          <option>Medium</option>
          <option>Hard</option>
        </select>
      </div>
    </div>

    <label for="totalQuestions">Number of Questions</label>
    <input id="totalQuestions" type="number" min="1" max="20" value="3" />

    <button id="startBtn" onclick="startInterview()">&#128640; Start Interview</button>
    <div id="startError" class="error-box hidden"></div>
  </div>

  <div class="panel hidden" id="interviewPanel">
    <div class="meta-row">
      <span id="questionCounter"></span>
      <span class="badge" id="difficultyBadge"></span>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>

    <div class="topic-tag" id="topicTag"></div>
    <div class="question-box" id="questionText"></div>
    <textarea id="answerInput" placeholder="Type your answer here..."></textarea>
    <button id="submitBtn" onclick="submitAnswer()">Submit Answer</button>
    <div id="answerError" class="error-box hidden"></div>

    <div class="eval-card hidden" id="evalCard"></div>
  </div>

  <div class="panel hidden" id="reportPanel">
    <h2 style="margin-top:0;">&#127942; Final Interview Report</h2>
    <div class="report" id="reportText"></div>
  </div>
</div>

<script>
let threadId = null;
let totalQuestions = 3;

function setLoading(btn, loading, label) {
  btn.disabled = loading;
  btn.innerHTML = loading ? '<span class="spinner"></span>Working...' : label;
}

function showError(elId, message) {
  const el = document.getElementById(elId);
  el.textContent = message;
  el.classList.remove('hidden');
}

function hideError(elId) {
  document.getElementById(elId).classList.add('hidden');
}

function scoreClass(score) {
  if (score >= 8) return 'good';
  if (score >= 5) return 'warn';
  return 'bad';
}

async function startInterview() {
  hideError('startError');
  const btn = document.getElementById('startBtn');
  setLoading(btn, true, '&#128640; Start Interview');

  const payload = {
    target_role: document.getElementById('targetRole').value.trim(),
    interview_type: document.getElementById('interviewType').value,
    selected_difficulty: document.getElementById('startDifficulty').value,
    total_questions: parseInt(document.getElementById('totalQuestions').value, 10),
  };

  try {
    const res = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to start interview');

    threadId = data.thread_id;
    totalQuestions = payload.total_questions;

    document.getElementById('startPanel').classList.add('hidden');
    document.getElementById('interviewPanel').classList.remove('hidden');
    renderQuestion(data);
  } catch (err) {
    showError('startError', err.message);
  } finally {
    setLoading(btn, false, '&#128640; Start Interview');
  }
}

function renderQuestion(data) {
  document.getElementById('questionCounter').textContent =
    `Question ${data.question_number} of ${totalQuestions}`;
  document.getElementById('difficultyBadge').textContent = data.difficulty;
  document.getElementById('progressFill').style.width =
    `${Math.min(100, (data.question_number / totalQuestions) * 100)}%`;
  document.getElementById('questionText').textContent = data.question || '';
  document.getElementById('answerInput').value = '';
  document.getElementById('evalCard').classList.add('hidden');
  hideError('answerError');
}

function renderEvaluation(evaluation, strategy) {
  const card = document.getElementById('evalCard');
  const cls = scoreClass(evaluation.score);
  card.innerHTML = `
    <div class="score ${cls}">${evaluation.score} / 10</div>
    <div style="color:var(--text-dim); font-size:0.85rem; margin-top:4px;">Next strategy: ${strategy}</div>
    <h4>Feedback</h4>
    <div>${evaluation.feedback || ''}</div>
    <h4>Strengths</h4>
    <ul>${(evaluation.strengths || []).map(s => `<li>${s}</li>`).join('') || '<li>None noted</li>'}</ul>
    <h4>Weaknesses</h4>
    <ul>${(evaluation.weaknesses || []).map(s => `<li>${s}</li>`).join('') || '<li>None noted</li>'}</ul>
  `;
  card.classList.remove('hidden');
}

async function submitAnswer() {
  hideError('answerError');
  const answer = document.getElementById('answerInput').value.trim();
  if (!answer) {
    showError('answerError', 'Please enter an answer before submitting.');
    return;
  }

  const btn = document.getElementById('submitBtn');
  setLoading(btn, true, 'Submit Answer');

  try {
    const res = await fetch('/api/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId, answer }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to submit answer');

    if (data.evaluation) {
      renderEvaluation(data.evaluation, data.strategy);
    }

    if (data.status === 'completed') {
      setTimeout(() => {
        document.getElementById('interviewPanel').classList.add('hidden');
        document.getElementById('reportPanel').classList.remove('hidden');
        document.getElementById('reportText').textContent = data.final_report || 'No report generated.';
      }, 900);
    } else {
      setTimeout(() => renderQuestion(data), 300);
    }
  } catch (err) {
    showError('answerError', err.message);
  } finally {
    setLoading(btn, false, 'Submit Answer');
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return FRONTEND_HTML


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )