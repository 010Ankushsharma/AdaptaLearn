"""
config.py
Central configuration for the Adaptive Learning System.
Every other module imports from here — never hardcode values elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_RAW_DIR    = Path(os.getenv("DATA_DIR",        ROOT_DIR / "data" / "raw"))
DATA_PROC_DIR   = Path(os.getenv("PROCESSED_DIR",   ROOT_DIR / "data" / "processed"))
CHECKPOINT_DIR  = Path(os.getenv("CHECKPOINT_DIR",  ROOT_DIR / "checkpoints"))
LOG_DIR         = Path(os.getenv("LOG_DIR",         ROOT_DIR / "logs"))

# Create dirs if they don't exist
for d in [DATA_RAW_DIR, DATA_PROC_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
EDNET_URL = (
    "https://drive.google.com/uc?id=1kSuFm0Z__MqKPxVHlHTJnTH5CnfHkLCR"
)
ASSISTMENTS_URL = (
    "https://drive.google.com/uc?id=0B-_gVbXZ_i-vc3NLbUhtSHpFbkE"
)

EDNET_FILE      = DATA_RAW_DIR / "ednet_kt4.parquet"
ASSISTMENTS_FILE= DATA_RAW_DIR / "assistments_2015.parquet"
CONCEPT_GRAPH   = DATA_PROC_DIR / "concept_graph.pkl"
TRAIN_FILE      = DATA_PROC_DIR / "train.parquet"
VAL_FILE        = DATA_PROC_DIR / "val.parquet"
TEST_FILE       = DATA_PROC_DIR / "test.parquet"

# Train / val / test split ratios (by student, not by row)
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# ── Concept / Question Space ──────────────────────────────────────────────────
NUM_CONCEPTS    = 188      # number of distinct concept tags in EdNet
NUM_QUESTIONS   = 13169    # total questions in EdNet KT4
MAX_SEQ_LEN     = 200      # max interactions per student sequence (for KT)
PAD_TOKEN       = -1       # padding value for ragged sequences

# ── DKVMN Knowledge Tracing ───────────────────────────────────────────────────
KT = dict(
    key_dim         = 50,      # dimension of concept key embeddings
    value_dim       = 200,     # dimension of student value memory
    num_concepts    = NUM_CONCEPTS,
    num_questions   = NUM_QUESTIONS,
    dropout         = 0.2,
    learning_rate   = 1e-3,
    batch_size      = 64,
    num_epochs      = 50,
    patience        = 7,       # early stopping patience (epochs)
    checkpoint_path = CHECKPOINT_DIR / "dkvmn_best.pt",
)

# ── Student Simulator (BKT) ───────────────────────────────────────────────────
BKT = dict(
    p_learn_range   = (0.05, 0.40),   # probability of learning a concept per attempt
    p_guess_range   = (0.10, 0.30),   # probability of guessing correctly when unmastered
    p_slip_range    = (0.05, 0.20),   # probability of slipping when mastered
    p_forget_range  = (0.00, 0.05),   # probability of forgetting per timestep
    num_simulators  = 10,             # ensemble size for robustness
)

# ── RL Environment ────────────────────────────────────────────────────────────
ENV = dict(
    max_questions_per_session = 20,
    mastery_threshold         = 0.85,  # episode ends if ALL concepts reach this
    zpd_lower                 = 0.40,  # min P(correct) for ZPD filter
    zpd_upper                 = 0.75,  # max P(correct) for ZPD filter
    state_dim                 = NUM_CONCEPTS + 4,  # mastery vec + [n_q, time, hints, avg_diff]
)

# ── Reward Shaping ─────────────────────────────────────────────────────────────
REWARD = dict(
    alpha   = 1.0,    # weight on Δmastery (primary signal)
    beta    = 0.3,    # weight on correctness bonus
    gamma   = 0.2,    # penalty per hint used
    delta   = 0.1,    # penalty per 10s over time limit
    lam     = 2.0,    # episode-end bonus weight on post-test gain
    time_limit_per_question = 120,  # seconds
)

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
PPO = dict(
    learning_rate   = 3e-4,
    n_steps         = 2048,    # rollout buffer size
    batch_size      = 64,
    n_epochs        = 10,      # update epochs per rollout
    gamma           = 0.99,    # discount factor
    gae_lambda      = 0.95,    # GAE λ
    clip_range      = 0.2,     # PPO ε clipping
    ent_coef        = 0.01,    # entropy bonus coefficient
    vf_coef         = 0.5,     # value function loss coefficient
    max_grad_norm   = 0.5,
    total_timesteps = 500_000,
    policy_kwargs   = dict(
        net_arch = [256, 128, 64],  # shared MLP layers
    ),
    checkpoint_path = CHECKPOINT_DIR / "ppo_best.zip",
    log_path        = LOG_DIR / "ppo",
)

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW = dict(
    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI", str(LOG_DIR / "mlflow")),
    experiment_name = "adaptive-learning-ppo",
)

# ── Django / Backend ──────────────────────────────────────────────────────────
BACKEND = dict(
    database_url = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3"),
    redis_url    = os.getenv("REDIS_URL",    "redis://localhost:6379/0"),
    secret_key   = os.getenv("DJANGO_SECRET_KEY", "dev-secret-change-in-prod"),
    debug        = os.getenv("DJANGO_DEBUG", "True") == "True",
)

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL = dict(
    n_eval_episodes = 100,      # episodes for RL baseline comparison
    alpha           = 0.05,     # significance level for t-test
    random_seed     = 42,
)