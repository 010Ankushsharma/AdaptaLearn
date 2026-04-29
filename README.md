# Personalized Adaptive Learning System
### Final Year Project — PPO / Actor-Critic + Knowledge Tracing

An RL agent that acts as a personal tutor — dynamically selecting which concept
and question to show each student next, based on their evolving knowledge state.

## Architecture

```
Student Interface (React)
        ↕
Django REST API
        ↕
PPO Agent (Stable-Baselines3)  ←→  DKVMN Knowledge Tracing (PyTorch)
        ↕
PostgreSQL + Redis
```

## Project Structure

```
adaptive_learning/
├── data/
│   ├── download_data.py     # File 3 — fetch EdNet + ASSISTments
│   ├── preprocess.py        # File 4 — clean, split, build concept graph
│   └── dataset.py           # File 5 — PyTorch Dataset/DataLoader
│
├── models/
│   ├── dkvmn.py             # File 6 — DKVMN knowledge tracing model
│   └── train_kt.py          # File 7 — KT training loop
│
├── rl/
│   ├── student_simulator.py # File 8  — BKT synthetic student
│   ├── env.py               # File 9  — Custom Gym environment
│   ├── reward.py            # File 10 — Reward shaping functions
│   ├── actor_critic.py      # File 11 — Actor-Critic network
│   ├── train_ppo.py         # File 12 — PPO training loop
│   └── evaluate.py          # File 13 — Baseline comparisons
│
├── backend/
│   ├── models.py            # File 14 — Django DB models
│   ├── agent_service.py     # File 15 — Policy inference service
│   └── views.py             # File 16 — REST API endpoints
│
├── frontend/
│   └── src/
│       └── App.jsx          # File 17 — React student interface
│
├── analysis/
│   ├── learning_gain.py     # File 18 — Statistical evaluation
│   └── ablation.py          # File 19 — Ablation studies
│
├── config.py                # File 2  — All hyperparameters & paths
├── requirements.txt         # File 1  — Dependencies
├── setup.sh                 # File 1  — Bootstrap script
└── .env.example             # File 1  — Environment template
```

## Quickstart

```bash
# 1. Clone and enter the project
git clone <your-repo>
cd adaptive_learning

# 2. Bootstrap environment
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 3. Fill in environment variables
cp .env.example .env
# edit .env with your DB credentials

# 4. Download and preprocess data
python data/download_data.py
python data/preprocess.py

# 5. Train knowledge tracing model
python models/train_kt.py

# 6. Train PPO agent
python rl/train_ppo.py

# 7. Evaluate against baselines
python rl/evaluate.py

# 8. Run web platform
python manage.py migrate
python manage.py runserver
cd frontend && npm start
```

## Datasets

| Dataset | Size | Use |
|---|---|---|
| EdNet KT4 | 131M interactions | Primary RL training data |
| ASSISTments 2015 | 400K interactions | KT model pre-training |

## Key Results (targets)

- Learning gain: >20% over fixed-curriculum baseline
- Knowledge tracing AUC: >0.78 on held-out students
- PPO vs random baseline: >15% higher cumulative reward
- ZPD adherence: >75% of actions within productive difficulty range