import os
import uuid
from typing import Any, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langserve import add_routes

from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

# Load .env from the SAME folder as this server.py.
# This avoids problems when the process is started from another
# working directory (locally or on a deployment platform).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite").strip()

# Accept both common variable names.
# Preferred name: GOOGLE_API_KEY
# Also accepted: GEMINI_API_KEY
GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if GOOGLE_API_KEY:
    GOOGLE_API_KEY = GOOGLE_API_KEY.strip()

# Do not print the actual secret.
print(f"[InterviewPilot] .env path: {ENV_PATH}")
print(f"[InterviewPilot] .env exists: {os.path.exists(ENV_PATH)}")
print(f"[InterviewPilot] Gemini API key loaded: {bool(GOOGLE_API_KEY)}")
print(f"[InterviewPilot] Gemini model: {MODEL_NAME}")


def create_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Gemini API key was not found. "
            f"Create a .env file beside server.py: {ENV_PATH} "
            "and add GOOGLE_API_KEY=your_key_here "
            "(GEMINI_API_KEY=your_key_here is also accepted)."
        )

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4,
        max_retries=2,
    )


llm = create_llm()

# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class QuestionResult(BaseModel):
    question: str
    topic: str


class EvaluationResult(BaseModel):
    score: int = Field(ge=0, le=10)
    correctness: str
    relevance: str
    depth: str
    clarity: str
    strengths: list[str]
    weaknesses: list[str]
    weak_topics: list[str]
    feedback: str


class DifficultyDecision(BaseModel):
    performance_level: Literal[
        "strong",
        "average",
        "weak"
    ]

    next_difficulty: Literal[
        "Easy",
        "Medium",
        "Hard"
    ]

    strategy: Literal[
        "increase_difficulty",
        "maintain_difficulty",
        "target_weakness",
        "follow_up"
    ]

    reason: str


# ============================================================
# API MODELS
# ============================================================

class StartRequest(BaseModel):
    target_role: str = Field(min_length=2)
    interview_type: str = Field(min_length=2)

    selected_difficulty: Literal[
        "Easy",
        "Medium",
        "Hard"
    ]

    total_questions: int = Field(
        default=5,
        ge=1,
        le=20
    )


class AnswerRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class InterviewResponse(BaseModel):
    thread_id: str

    status: Literal[
        "waiting_for_answer",
        "completed"
    ]

    question_number: Optional[int] = None
    question: Optional[str] = None
    difficulty: Optional[str] = None

    evaluation: Optional[dict[str, Any]] = None

    strategy: Optional[str] = None

    final_report: Optional[str] = None


# ============================================================
# LANGGRAPH STATE
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

    current_evaluation: dict[str, Any]

    current_strategy: str

    question_history: list[str]
    answer_history: list[str]

    evaluation_history: list[
        dict[str, Any]
    ]

    weak_topics: list[str]

    difficulty_history: list[str]

    interview_complete: bool

    final_report: str


# ============================================================
# AGENT 1 — QUESTION AGENT
# ============================================================


# ============================================================
# ROLE-SPECIFIC QUESTION GUIDANCE + FALLBACK BANK
# ============================================================

QUESTION_BANK = {
    "python": {
        "easy": [
            ("Python Fundamentals", "What is the difference between a list and a tuple in Python, and when would you use each?"),
            ("Python Functions", "What is the difference between a function parameter and an argument in Python? Give a simple example."),
            ("Python Collections", "How do dictionaries work in Python, and what makes dictionary key lookup efficient?"),
            ("Python Exceptions", "What is the purpose of try, except, and finally in Python? When would you use them?"),
        ],
        "medium": [
            ("Python Performance", "What is the difference between shallow copy and deep copy in Python, and when can it matter in a backend application?"),
            ("Python Concurrency", "What is the difference between threading, multiprocessing, and asyncio in Python? When would you choose each?"),
            ("Python Design", "How would you structure a Python service so that business logic is separated from API and database code?"),
            ("Python APIs", "How would you validate and handle errors for incoming JSON requests in a Python REST API?"),
        ],
        "hard": [
            ("Python Architecture", "You need to build a high-throughput Python API. How would you design its concurrency, connection pooling, caching, and error handling?"),
            ("Python Performance", "A Python API becomes slow under heavy traffic even though CPU usage is low. How would you investigate and improve it?"),
            ("Python Systems", "How would you design a resilient asynchronous Python service that calls several external APIs with timeouts, retries, and rate limits?"),
        ],
    },
    "ai": {
        "easy": [
            ("Machine Learning Fundamentals", "What is the difference between supervised and unsupervised learning? Give one practical example of each."),
            ("Model Evaluation", "Why can classification accuracy be misleading on an imbalanced dataset, and how does F1-score help?"),
            ("Overfitting", "What is overfitting in machine learning, and name two practical ways to reduce it."),
            ("Embeddings", "What are embeddings, and why are they useful when building an AI application?"),
        ],
        "medium": [
            ("Model Evaluation", "You have a fraud detection model with very few fraud examples. Which metrics would you monitor and why?"),
            ("RAG", "Walk through the main steps of a Retrieval-Augmented Generation system and explain where retrieval quality can fail."),
            ("LLM Applications", "How would you design an evaluation strategy for an LLM application when there is no single exact correct answer?"),
            ("ML Deployment", "What would you monitor after deploying a machine learning model to production, and why?"),
            ("Agents", "What is tool calling in an AI agent, and how would you prevent an agent from repeatedly calling the same tool?"),
        ],
        "hard": [
            ("RAG Architecture", "Design a production RAG system for a large enterprise knowledge base. Discuss chunking, embeddings, retrieval, reranking, latency, and evaluation."),
            ("Agent Architecture", "Design a multi-agent AI system that can research a question, call external tools, verify results, and produce a final answer. How would you control failures?"),
            ("LLM Reliability", "An LLM application gives correct answers most of the time but occasionally produces confident hallucinations. How would you diagnose and reduce the problem?"),
            ("Production AI", "You need to reduce latency and cost in a production LLM application without significantly reducing answer quality. What would you change and how would you measure the impact?"),
        ],
    },
    "ml": {
        "easy": [
            ("Supervised Learning", "What is the difference between classification and regression? Give a practical example of each."),
            ("Data Preprocessing", "Why do we split a dataset into training and testing sets?"),
            ("Model Evaluation", "What is the difference between precision and recall?"),
            ("Overfitting", "What is overfitting and how can cross-validation help detect it?"),
        ],
        "medium": [
            ("Feature Engineering", "How would you handle missing values and categorical variables before training a machine learning model?"),
            ("Model Selection", "How would you decide between a tree-based model and a linear model for a new classification problem?"),
            ("Imbalanced Data", "What techniques can you use when the minority class is much smaller than the majority class?"),
            ("ML Pipelines", "How would you build a reproducible machine learning pipeline from raw data through model evaluation?"),
        ],
        "hard": [
            ("Production ML", "Design an end-to-end ML system that trains periodically, serves predictions online, monitors drift, and supports model rollback."),
            ("Model Debugging", "A model performs well offline but poorly after deployment. Walk through the investigation you would perform."),
            ("ML System Design", "How would you design a scalable recommendation system and decide which parts should be batch versus real-time?"),
        ],
    },
}

def _role_category(role: str) -> str:
    r = role.lower()
    if any(x in r for x in ["ai engineer", "genai", "generative ai", "llm", "agentic"]):
        return "ai"
    if any(x in r for x in ["ml engineer", "machine learning", "machine-learning"]):
        return "ml"
    if any(x in r for x in ["python", "backend", "software engineer", "developer"]):
        return "python"
    return "ai"


def _fallback_question(state: InterviewState) -> tuple[str, str]:
    category = _role_category(state["target_role"])
    difficulty = state["current_difficulty"].lower()
    pool = QUESTION_BANK[category].get(difficulty, QUESTION_BANK[category]["medium"])

    previous = {q.strip().lower() for q in state["question_history"]}
    weak = {w.strip().lower() for w in state["weak_topics"]}

    # Prefer a question that targets an identified weakness.
    if weak:
        for topic, question in pool:
            if topic.lower() in weak or any(w in topic.lower() for w in weak):
                if question.lower() not in previous:
                    return topic, question

    for topic, question in pool:
        if question.lower() not in previous:
            return topic, question

    # If the small fallback bank is exhausted, produce a role-specific
    # non-repeating variant rather than the old generic question.
    topic = pool[0][0]
    return topic, f"Explain a practical {topic.lower()} problem you might face as a {state['target_role']} and describe how you would solve it."

question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Question Agent of InterviewPilot, acting like a real technical interviewer.

Your ONLY responsibility is to generate ONE realistic interview question.

The question MUST be specific to the target role and interview type.
Do NOT use generic questions such as "Tell me about a challenging problem you solved recently" unless the interview type is HR/behavioral.

For Technical interviews:
- Ask concrete technical/conceptual/problem-solving questions.
- Match the selected role.
- Easy = fundamentals and straightforward practical concepts.
- Medium = applied concepts, debugging, trade-offs, and implementation reasoning.
- Hard = architecture, production trade-offs, scalability, failure handling, or advanced problem solving.

For System Design:
- Ask design/architecture/trade-off questions appropriate to the role and difficulty.

For HR:
- Ask behavioral questions relevant to the candidate's role.

For Technical + HR:
- Alternate naturally between technical and behavioral questions.

Strategies:
increase_difficulty:
Ask a clearly harder question than the previous one.

maintain_difficulty:
Ask a different question at the same level.

target_weakness:
Test one identified weak topic with a new question.

follow_up:
Ask a focused question that probes the candidate's previous weakness or incomplete answer. Do not repeat the exact previous question.

IMPORTANT:
- Never repeat a previous question.
- Do not merely paraphrase a previous question.
- Do not always ask experience-based questions.
- Vary topics and question styles.
- Use the candidate's role, not a generic "AI Engineer" label.
- Return only one question and its topic.

Return:
1. question
2. topic
""",
        ),
        (
            "human",
            """
Target Role:
{target_role}

Interview Type:
{interview_type}

Required Difficulty:
{current_difficulty}

Strategy:
{strategy}

Previous Questions:
{previous_questions}

Weak Topics:
{weak_topics}

Previous Evaluations:
{previous_evaluations}
""",
        ),
    ]
)

question_agent = (
    question_prompt
    | llm.with_structured_output(QuestionResult)
)


def run_question_agent(
    state: InterviewState
):

    next_number = (
        state["current_question_number"] + 1
    )

    if next_number > state["total_questions"]:
        return {
            "interview_complete": True
        }

    try:
        result = question_agent.invoke(
            {
                "target_role":
                    state["target_role"],

                "interview_type":
                    state["interview_type"],

                "current_difficulty":
                    state["current_difficulty"],

                "strategy":
                    state["current_strategy"],

                "previous_questions":
                    "\n".join(
                        f"{i + 1}. {q}"
                        for i, q in enumerate(
                            state["question_history"]
                        )
                    ) or "None",

                "weak_topics":
                    ", ".join(
                        state["weak_topics"]
                    ) or "None",

                "previous_evaluations":
                    "\n".join(
                        str(e)
                        for e in state[
                            "evaluation_history"
                        ]
                    ) or "None",
            }
        )

        question = result.question.strip()
        topic = result.topic.strip() or "General"

        previous_normalized = {
            q.strip().lower()
            for q in state["question_history"]
        }

        # Reject exact or near-obvious repeats.
        if not question or question.lower() in previous_normalized:
            raise ValueError("Question Agent generated a repeated/empty question.")

    except Exception:
        topic, question = _fallback_question(state)

    return {
        "current_question":
            question,

        "current_topic":
            topic,

        "current_question_number":
            next_number,

        "question_history":
            state["question_history"]
            + [question],

        "difficulty_history":
            state["difficulty_history"]
            + [state["current_difficulty"]],

        "interview_complete":
            False
    }


# ============================================================
# HUMAN INPUT
# ============================================================

def human_input_node(
    state: InterviewState
):

    answer = interrupt(
        {
            "type":
                "candidate_answer_required",

            "question":
                state["current_question"],

            "question_number":
                state["current_question_number"],

            "difficulty":
                state["current_difficulty"],

            "topic":
                state["current_topic"],
        }
    )

    return {
        "current_answer":
            str(answer),

        "answer_history":
            state["answer_history"]
            + [str(answer)]
    }


# ============================================================
# AGENT 2 — EVALUATOR
# ============================================================

evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Evaluator Agent of InterviewPilot.

Your ONLY responsibility is evaluating the candidate's
answer.

Score from 0 to 10.

Evaluate:

- correctness
- relevance
- depth
- clarity

Identify:

- strengths
- weaknesses
- weak topics
- useful feedback

Be strict but fair.

Do not generate a question.

Do not decide difficulty.

Do not expose hidden reasoning.
""",
        ),
        (
            "human",
            """
Target Role:
{target_role}

Interview Type:
{interview_type}

Question:
{question}

Candidate Answer:
{answer}
""",
        ),
    ]
)

evaluator_agent = (
    evaluator_prompt
    | llm.with_structured_output(
        EvaluationResult
    )
)


def run_evaluator_agent(
    state: InterviewState
):

    result = evaluator_agent.invoke(
        {
            "target_role":
                state["target_role"],

            "interview_type":
                state["interview_type"],

            "question":
                state["current_question"],

            "answer":
                state["current_answer"],
        }
    )

    evaluation = result.model_dump()

    weak_topics = list(
        dict.fromkeys(
            state["weak_topics"]
            + result.weak_topics
        )
    )

    return {
        "current_evaluation":
            evaluation,

        "evaluation_history":
            state["evaluation_history"]
            + [evaluation],

        "weak_topics":
            weak_topics
    }


# ============================================================
# AGENT 3 — DIFFICULTY AGENT
# ============================================================

difficulty_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Difficulty Agent of InterviewPilot.

Your responsibility is ONLY to decide how the
next question should adapt.

Use the candidate's evaluation.

Score guidance:

8-10:
Strong → increase difficulty.

5-7:
Average → maintain difficulty.

0-4:
Weak → target weakness.

Available strategies:

increase_difficulty
maintain_difficulty
target_weakness
follow_up

Return the next difficulty and strategy.

Do not generate a question.
Do not evaluate the answer again.
""",
        ),
        (
            "human",
            """
Current Difficulty:
{current_difficulty}

Question Number:
{question_number}

Total Questions:
{total_questions}

Evaluation:
{evaluation}

Weak Topics:
{weak_topics}
""",
        ),
    ]
)

difficulty_agent = (
    difficulty_prompt
    | llm.with_structured_output(
        DifficultyDecision
    )
)


def run_difficulty_agent(
    state: InterviewState
):

    evaluation = (
        state["current_evaluation"]
        or {}
    )

    try:
        score = int(
            evaluation.get(
                "score",
                5
            )
        )
    except (
        TypeError,
        ValueError
    ):
        score = 5

    # Last question
    if (
        state["current_question_number"]
        >= state["total_questions"]
    ):
        return {
            "current_strategy":
                "maintain_difficulty",

            "interview_complete":
                True
        }

    result = difficulty_agent.invoke(
        {
            "current_difficulty":
                state["current_difficulty"],

            "question_number":
                state["current_question_number"],

            "total_questions":
                state["total_questions"],

            "evaluation":
                str(evaluation),

            "weak_topics":
                ", ".join(
                    state["weak_topics"]
                ) or "None"
        }
    )

    # The Difficulty Agent is authoritative.
    next_difficulty = (
        result.next_difficulty
    )

    strategy = result.strategy

    # Deterministic safeguards
    if score >= 8:

        strategy = "increase_difficulty"

        if (
            state["current_difficulty"]
            == "Easy"
        ):
            next_difficulty = "Medium"

        elif (
            state["current_difficulty"]
            == "Medium"
        ):
            next_difficulty = "Hard"

        else:
            next_difficulty = "Hard"

    elif score <= 4:

        strategy = "target_weakness"

        if (
            state["current_difficulty"]
            == "Hard"
        ):
            next_difficulty = "Medium"

        else:
            next_difficulty = (
                state["current_difficulty"]
            )

    return {
        "current_strategy":
            strategy,

        "current_difficulty":
            next_difficulty,

        "interview_complete":
            False
    }


# ============================================================
# ROUTER
# ============================================================

def route_after_difficulty(
    state: InterviewState
):

    if (
        state["current_question_number"]
        >= state["total_questions"]
    ):
        return "feedback_agent"

    if state["interview_complete"]:
        return "feedback_agent"

    return "question_agent"


# ============================================================
# AGENT 4 — FEEDBACK AGENT
# ============================================================

feedback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Feedback Agent of InterviewPilot.

Analyze the complete interview.

Generate a professional report.

Include:

# Interview Performance Report

## Overall Assessment

## Overall Score

## Strengths

## Weaknesses

## Weak Topics

## Question-by-Question Review

## Difficulty Progression

## Recommendations

## Final Assessment

Be honest and useful.

Do not expose chain-of-thought.
""",
        ),
        (
            "human",
            """
Target Role:
{target_role}

Interview Type:
{interview_type}

Questions:
{questions}

Answers:
{answers}

Evaluations:
{evaluations}

Difficulty Progression:
{difficulty_history}
""",
        ),
    ]
)

feedback_agent = (
    feedback_prompt
    | llm
)


def run_feedback_agent(
    state: InterviewState
):

    evaluations = (
        state["evaluation_history"]
    )

    scores = [
        e.get("score", 0)
        for e in evaluations
        if isinstance(e, dict)
    ]

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    report = feedback_agent.invoke(
        {
            "target_role":
                state["target_role"],

            "interview_type":
                state["interview_type"],

            "questions":
                "\n".join(
                    f"{i + 1}. {q}"
                    for i, q in enumerate(
                        state["question_history"]
                    )
                ),

            "answers":
                "\n".join(
                    f"{i + 1}. {a}"
                    for i, a in enumerate(
                        state["answer_history"]
                    )
                ),

            "evaluations":
                "\n".join(
                    str(e)
                    for e in state[
                        "evaluation_history"
                    ]
                ),

            "difficulty_history":
                " → ".join(
                    state[
                        "difficulty_history"
                    ]
                )
        }
    )

    raw_content = (
        report.content
        if hasattr(report, "content")
        else str(report)
    )

    if isinstance(raw_content, list):
        report_text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    else:
        report_text = str(raw_content)

    final_report = (
        f"# Interview Complete\n\n"
        f"**Average Score: "
        f"{average_score:.1f}/10**\n\n"
        f"{report_text}"
    )

    return {
        "final_report":
            final_report,

        "interview_complete":
            True
    }


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(
    InterviewState
)

builder.add_node(
    "question_agent",
    run_question_agent
)

builder.add_node(
    "human_input",
    human_input_node
)

builder.add_node(
    "evaluator_agent",
    run_evaluator_agent
)

builder.add_node(
    "difficulty_agent",
    run_difficulty_agent
)

builder.add_node(
    "feedback_agent",
    run_feedback_agent
)

builder.add_edge(
    START,
    "question_agent"
)

builder.add_edge(
    "question_agent",
    "human_input"
)

builder.add_edge(
    "human_input",
    "evaluator_agent"
)

builder.add_edge(
    "evaluator_agent",
    "difficulty_agent"
)

builder.add_conditional_edges(
    "difficulty_agent",
    route_after_difficulty,
    {
        "question_agent":
            "question_agent",

        "feedback_agent":
            "feedback_agent"
    }
)

builder.add_edge(
    "feedback_agent",
    END
)

checkpointer = InMemorySaver()

interview_graph = builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="InterviewPilot",
    version="1.0.0",
    description=(
        "Adaptive Multi-Agent Interview System "
        "using LangChain, LangGraph and LangServe."
    )
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# HELPERS
# ============================================================

def extract_interrupt(result):

    interrupts = getattr(
        result,
        "interrupts",
        None
    )

    if interrupts:
        return interrupts[0]

    if isinstance(result, dict):

        raw = result.get(
            "__interrupt__"
        )

        if raw:
            return raw[0]

    return None


def result_values(result):

    values = getattr(
        result,
        "value",
        None
    )

    if isinstance(values, dict):
        return values

    if isinstance(result, dict):
        return result

    return {}


def build_waiting_response(
    thread_id,
    values
):

    return InterviewResponse(
        thread_id=thread_id,

        status="waiting_for_answer",

        question_number=values.get(
            "current_question_number"
        ),

        question=values.get(
            "current_question"
        ),

        difficulty=values.get(
            "current_difficulty"
        ),

        evaluation=values.get(
            "current_evaluation"
        ),

        strategy=values.get(
            "current_strategy"
        )
    )


# ============================================================
# START INTERVIEW
# ============================================================

@app.post(
    "/api/start",
    response_model=InterviewResponse
)
def start_interview(
    request: StartRequest
):

    thread_id = str(
        uuid.uuid4()
    )

    initial_state: InterviewState = {

        "target_role":
            request.target_role,

        "interview_type":
            request.interview_type,

        "selected_difficulty":
            request.selected_difficulty,

        "current_difficulty":
            request.selected_difficulty,

        "total_questions":
            request.total_questions,

        "current_question_number":
            0,

        "current_question":
            "",

        "current_topic":
            "",

        "current_answer":
            "",

        "current_evaluation":
            {},

        "current_strategy":
            "maintain_difficulty",

        "question_history":
            [],

        "answer_history":
            [],

        "evaluation_history":
            [],

        "weak_topics":
            [],

        "difficulty_history":
            [],

        "interview_complete":
            False,

        "final_report":
            ""
    }

    config = {
        "configurable": {
            "thread_id":
                thread_id
        }
    }

    try:

        result = interview_graph.invoke(
            initial_state,
            config=config
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to start interview: "
                f"{exc}"
            )
        )

    if not extract_interrupt(result):

        raise HTTPException(
            status_code=500,
            detail=(
                "Interview did not pause "
                "for candidate input."
            )
        )

    return build_waiting_response(
        thread_id,
        result_values(result)
    )


# ============================================================
# SUBMIT ANSWER
# ============================================================

@app.post(
    "/api/answer",
    response_model=InterviewResponse
)
def submit_answer(
    request: AnswerRequest
):

    config = {
        "configurable": {
            "thread_id":
                request.thread_id
        }
    }

    try:

        current = (
            interview_graph.get_state(
                config
            )
        )

        if not current.values:

            raise HTTPException(
                status_code=404,
                detail="Interview thread not found."
            )

        result = (
            interview_graph.invoke(
                Command(
                    resume=request.answer
                ),
                config=config
            )
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to resume interview: "
                f"{exc}"
            )
        )

    values = result_values(result)

    # Another question
    if extract_interrupt(result):

        return build_waiting_response(
            request.thread_id,
            values
        )

    # Final report
    if (
        values.get(
            "interview_complete"
        )
        or values.get(
            "final_report"
        )
    ):

        return InterviewResponse(

            thread_id=request.thread_id,

            status="completed",

            question_number=values.get(
                "current_question_number"
            ),

            evaluation=values.get(
                "current_evaluation"
            ),

            strategy=values.get(
                "current_strategy"
            ),

            final_report=values.get(
                "final_report"
            )
        )

    raise HTTPException(
        status_code=500,
        detail="Unexpected graph state."
    )


# ============================================================
# GET STATE
# ============================================================

@app.get(
    "/api/state/{thread_id}"
)
def get_interview_state(
    thread_id: str
):

    config = {
        "configurable": {
            "thread_id":
                thread_id
        }
    }

    snapshot = (
        interview_graph.get_state(
            config
        )
    )

    if not snapshot.values:

        raise HTTPException(
            status_code=404,
            detail="Interview thread not found."
        )

    values = snapshot.values

    return {
        "thread_id":
            thread_id,

        "status":
            (
                "completed"
                if values.get(
                    "interview_complete"
                )
                else "waiting_for_answer"
            ),

        "question_number":
            values.get(
                "current_question_number"
            ),

        "question":
            values.get(
                "current_question"
            ),

        "topic":
            values.get(
                "current_topic"
            ),

        "difficulty":
            values.get(
                "current_difficulty"
            ),

        "strategy":
            values.get(
                "current_strategy"
            ),

        "evaluation":
            values.get(
                "current_evaluation"
            ),

        "weak_topics":
            values.get(
                "weak_topics"
            ),

        "final_report":
            values.get(
                "final_report"
            )
    }


# ============================================================
# WEB APPLICATION
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def web_app():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>InterviewPilot</title>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        Arial,
        sans-serif;

    background:
        radial-gradient(
            circle at top right,
            #172554,
            #050816 45%,
            #020617
        );

    color: #e5e7eb;
    min-height: 100vh;
}

.container {
    width: min(1050px, 92%);
    margin: auto;
    padding: 45px 0;
}

.header {
    text-align: center;
    margin-bottom: 35px;
}

.logo {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
}

.logo span {
    color: #60a5fa;
}

.subtitle {
    color: #94a3b8;
    margin-top: 10px;
    font-size: 17px;
}

.card {
    background:
        rgba(15, 23, 42, 0.82);

    border:
        1px solid rgba(148, 163, 184, 0.15);

    border-radius: 22px;

    padding: 30px;

    box-shadow:
        0 25px 80px
        rgba(0, 0, 0, 0.35);

    backdrop-filter: blur(16px);
}

.hidden {
    display: none;
}

label {
    display: block;
    margin-bottom: 8px;
    color: #cbd5e1;
    font-weight: 600;
}

input,
select,
textarea {
    width: 100%;

    border: 1px solid #334155;

    background: #0f172a;

    color: #f8fafc;

    border-radius: 12px;

    padding: 14px;

    font-size: 15px;

    outline: none;
}

input:focus,
select:focus,
textarea:focus {
    border-color: #60a5fa;
}

.field {
    margin-bottom: 20px;
}

.grid {
    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 20px;
}

button {
    border: none;

    border-radius: 12px;

    padding: 14px 22px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    font-size: 15px;

    font-weight: 700;

    cursor: pointer;

    transition: 0.2s;
}

button:hover {
    transform: translateY(-1px);

    filter: brightness(1.12);
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.primary {
    width: 100%;
    margin-top: 5px;
}

.topbar {
    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 25px;
}

.badge {
    display: inline-block;

    padding: 7px 12px;

    border-radius: 999px;

    background: rgba(
        96,
        165,
        250,
        0.12
    );

    color: #93c5fd;

    font-size: 13px;

    font-weight: 700;
}

.progress {
    height: 8px;

    background: #1e293b;

    border-radius: 20px;

    overflow: hidden;

    margin: 15px 0 30px;
}

.progress-bar {
    height: 100%;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #8b5cf6
        );

    transition: width 0.4s;
}

.question {
    font-size: 26px;

    line-height: 1.45;

    font-weight: 700;

    color: #f8fafc;

    margin: 25px 0;
}

textarea {
    min-height: 190px;

    resize: vertical;

    line-height: 1.6;
}

.meta {
    display: flex;

    gap: 10px;

    flex-wrap: wrap;
}

.evaluation {
    margin-top: 25px;

    border-top:
        1px solid #1e293b;

    padding-top: 25px;
}

.score {
    font-size: 34px;

    font-weight: 800;

    color: #60a5fa;
}

.eval-grid {
    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 15px;

    margin-top: 15px;
}

.eval-box {
    background: #0b1220;

    border-radius: 14px;

    padding: 18px;
}

.eval-box h4 {
    margin-top: 0;
    color: #93c5fd;
}

.report {
    white-space: pre-wrap;

    line-height: 1.7;

    background: #0b1220;

    border-radius: 15px;

    padding: 25px;

    color: #cbd5e1;
}

.error {
    background:
        rgba(239, 68, 68, 0.1);

    border:
        1px solid rgba(239, 68, 68, 0.3);

    color: #fca5a5;

    padding: 14px;

    border-radius: 12px;

    margin-top: 15px;
}

.loading {
    text-align: center;

    color: #94a3b8;

    padding: 20px;
}

@media (max-width: 700px) {

    .grid,
    .eval-grid {
        grid-template-columns: 1fr;
    }

    .question {
        font-size: 21px;
    }

    .container {
        padding: 25px 0;
    }

    .card {
        padding: 20px;
    }

}

</style>

</head>


<body>

<div class="container">

    <div class="header">

        <div class="logo">
            🧠 Interview<span>Pilot</span>
        </div>

        <div class="subtitle">
            Adaptive Multi-Agent AI Interview System
        </div>

    </div>


    <!-- SETUP -->

    <div
        id="setup"
        class="card"
    >

        <h2>
            Start Your Interview
        </h2>

        <p class="subtitle">
            InterviewPilot adapts the next question
            based on your performance.
        </p>


        <div class="grid">

            <div class="field">

                <label>
                    Target Role
                </label>

                <input
                    id="role"
                    placeholder="Python Backend Engineer"
                >

            </div>


            <div class="field">

                <label>
                    Interview Type
                </label>

                <select id="type">

                    <option>
                        Technical
                    </option>

                    <option>
                        HR
                    </option>

                    <option>
                        Technical + HR
                    </option>

                    <option>
                        System Design
                    </option>

                </select>

            </div>


            <div class="field">

                <label>
                    Starting Difficulty
                </label>

                <select id="difficulty">

                    <option>
                        Easy
                    </option>

                    <option>
                        Medium
                    </option>

                    <option>
                        Hard
                    </option>

                </select>

            </div>


            <div class="field">

                <label>
                    Number of Questions
                </label>

                <select id="questions">

                    <option>3</option>
                    <option>5</option>
                    <option>7</option>
                    <option>10</option>

                </select>

            </div>

        </div>


        <button
            class="primary"
            onclick="startInterview()"
        >
            🚀 Start Interview
        </button>


        <div
            id="setupError"
            class="error hidden"
        ></div>

    </div>


    <!-- INTERVIEW -->

    <div
        id="interview"
        class="card hidden"
    >

        <div class="topbar">

            <div>

                <div id="roleDisplay"></div>

                <div
                    id="questionCounter"
                    class="subtitle"
                ></div>

            </div>


            <div
                id="difficultyBadge"
                class="badge"
            >
            </div>

        </div>


        <div class="progress">

            <div
                id="progressBar"
                class="progress-bar"
                style="width: 0%"
            ></div>

        </div>


        <div
            id="topicBadge"
            class="badge"
        >
        </div>


        <div
            id="question"
            class="question"
        >
        </div>


        <textarea
            id="answer"
            placeholder="Type your answer here..."
        ></textarea>


        <button
            id="submitButton"
            class="primary"
            onclick="submitAnswer()"
        >
            Submit Answer →
        </button>


        <div
            id="loading"
            class="loading hidden"
        >
            🤖 InterviewPilot is evaluating your answer...
        </div>


        <div
            id="evaluation"
            class="evaluation hidden"
        >

            <div>
                <div class="subtitle">
                    Latest Evaluation
                </div>

                <div
                    id="score"
                    class="score"
                >
                </div>
            </div>


            <div class="eval-grid">

                <div class="eval-box">

                    <h4>
                        ✓ Strengths
                    </h4>

                    <div id="strengths"></div>

                </div>


                <div class="eval-box">

                    <h4>
                        ⚠ Weaknesses
                    </h4>

                    <div id="weaknesses"></div>

                </div>


                <div class="eval-box">

                    <h4>
                        🎯 Weak Topics
                    </h4>

                    <div id="weakTopics"></div>

                </div>


                <div class="eval-box">

                    <h4>
                        📈 Next Strategy
                    </h4>

                    <div id="strategy"></div>

                </div>

            </div>

        </div>


        <div
            id="error"
            class="error hidden"
        ></div>

    </div>


    <!-- FINAL REPORT -->

    <div
        id="complete"
        class="card hidden"
    >

        <h1>
            🎯 Interview Complete
        </h1>

        <p class="subtitle">
            Your adaptive interview is finished.
        </p>


        <div
            id="finalReport"
            class="report"
        >
        </div>


        <button
            class="primary"
            onclick="location.reload()"
        >
            🔄 Start New Interview
        </button>

    </div>

</div>


<script>

let threadId = null;

let totalQuestions = 0;

let currentQuestion = 0;


function show(id) {

    document
        .getElementById(id)
        .classList
        .remove("hidden");

}


function hide(id) {

    document
        .getElementById(id)
        .classList
        .add("hidden");

}


function showError(id, message) {

    const element =
        document.getElementById(id);

    element.textContent = message;

    element.classList.remove(
        "hidden"
    );

}


function clearError(id) {

    document
        .getElementById(id)
        .classList.add(
            "hidden"
        );

}


async function startInterview() {

    clearError("setupError");

    const role =
        document
        .getElementById("role")
        .value
        .trim();

    if (!role) {

        showError(
            "setupError",
            "Please enter a target role."
        );

        return;
    }


    const interviewType =
        document
        .getElementById("type")
        .value;


    const difficulty =
        document
        .getElementById("difficulty")
        .value;


    totalQuestions =
        Number(
            document
            .getElementById("questions")
            .value
        );


    try {

        const response =
            await fetch(
                "/api/start",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        target_role:
                            role,

                        interview_type:
                            interviewType,

                        selected_difficulty:
                            difficulty,

                        total_questions:
                            totalQuestions

                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Failed to start interview."
            );
        }


        threadId =
            data.thread_id;


        currentQuestion =
            data.question_number;


        document
            .getElementById("roleDisplay")
            .textContent =
            role;


        hide("setup");

        show("interview");

        renderQuestion(data);

    }

    catch (error) {

        showError(
            "setupError",
            error.message
        );

    }

}


function renderQuestion(data) {

    document
        .getElementById("questionCounter")
        .textContent =
        `Question ${data.question_number} of ${totalQuestions}`;


    document
        .getElementById("difficultyBadge")
        .textContent =
        data.difficulty;


    document
        .getElementById("question")
        .textContent =
        data.question;


    document
        .getElementById("progressBar")
        .style.width =
        `${(
            data.question_number /
            totalQuestions
        ) * 100}%`;


    document
        .getElementById("answer")
        .value = "";


    hide("evaluation");

    clearError("error");

    hide("loading");

    document
        .getElementById("submitButton")
        .disabled = false;

}


async function submitAnswer() {

    clearError("error");

    const answer =
        document
        .getElementById("answer")
        .value
        .trim();


    if (!answer) {

        showError(
            "error",
            "Please enter an answer."
        );

        return;
    }


    document
        .getElementById("submitButton")
        .disabled = true;


    show("loading");


    try {

        const response =
            await fetch(
                "/api/answer",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        thread_id:
                            threadId,

                        answer:
                            answer

                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Failed to submit answer."
            );
        }


        if (
            data.status ===
            "completed"
        ) {

            hide("interview");

            show("complete");

            document
                .getElementById(
                    "finalReport"
                )
                .innerHTML = marked.parse(
                    data.final_report ||
                    "Interview completed."
                );

            return;
        }


        renderEvaluation(
            data.evaluation,
            data.strategy
        );


        currentQuestion =
            data.question_number;


        renderQuestion(data);

    }

    catch (error) {

        hide("loading");

        document
            .getElementById(
                "submitButton"
            )
            .disabled = false;

        showError(
            "error",
            error.message
        );

    }

}


function renderEvaluation(
    evaluation,
    strategy
) {

    if (!evaluation) {
        return;
    }


    show("evaluation");


    document
        .getElementById("score")
        .textContent =
        `${evaluation.score}/10`;


    document
        .getElementById("strengths")
        .innerHTML =
        makeList(
            evaluation.strengths
        );


    document
        .getElementById("weaknesses")
        .innerHTML =
        makeList(
            evaluation.weaknesses
        );


    document
        .getElementById("weakTopics")
        .innerHTML =
        makeList(
            evaluation.weak_topics
        );


    document
        .getElementById("strategy")
        .textContent =
        strategy ||
        "Maintain difficulty";
}


function makeList(items) {

    if (
        !items ||
        items.length === 0
    ) {

        return "None identified.";

    }


    return items
        .map(
            item =>
                `<div style="margin-bottom:8px">
                    • ${escapeHtml(item)}
                </div>`
        )
        .join("");
}


function escapeHtml(text) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        text;

    return div.innerHTML;
}

</script>

</body>

</html>
"""


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    interview_graph,
    path="/interview"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8080"
            )
        ),
        reload=False
    )