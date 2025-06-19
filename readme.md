## 🧠 Report Summary: Imitation Learning with DAgger

This section summarizes the report on training agents using **DAgger (Dataset Aggregation)** — an iterative imitation learning algorithm. DAgger allows the agent to collect data from its own state distribution while continuing to learn from expert feedback, significantly improving over traditional behavior cloning.

---

### 📌 1. Algorithm Overview

We use **DAgger** to iteratively train a policy that mimics expert behavior:

1. The agent collects trajectories using its current policy.
2. At each visited state, the expert is queried for the correct action.
3. These (state, expert-action) pairs are added to the training dataset.
4. The learner minimizes MSE between its prediction and the expert's actions.
5. After each training loop, we evaluate the updated policy.
6. If the average reward improves, the model is checkpointed.

> ✅ This approach improves generalization by letting the learner visit non-expert trajectories and learn to recover.

---

### 🌍 2. Environment Details

#### 🔹 Hopper-v4

| Attribute         | Value                        |
|------------------|------------------------------|
| Observation Space| Shape: (11,) — `qpos` (5) + `qvel` (6) |
| Action Space     | Shape: (3,) — Torque vector for joints |
| Objective        | Learn to hop and balance     |

#### 🔹 Ant-v4

| Attribute         | Value                        |
|------------------|------------------------------|
| Observation Space| Shape: (105,) — `qpos` (13), `qvel` (14), external forces (78) |
| Action Space     | Shape: (8,) — Motor commands for 8 joints |
| Objective        | Learn stable walking using quadruped body |

---

### 🎯 3. Evaluation Strategy

- **Evaluation Metric**: Average **cumulative reward** per episode
- **Checkpoint Criteria**: Model is saved only when its reward surpasses all previous iterations
- **Purpose**: Use return as a direct indicator of task performance
- **Why reward?** Because it best reflects whether the policy is learning to replicate expert success

---

### 📊 4. Results & Visualization

#### 📈 Episode Length Over Time (Hopper-v4)
- **Trend**: Increasing
- **Interpretation**: The agent learns to survive longer → greater stability and coordination.

#### 📈 Average Return Over Time (Hopper-v4)
- **Trend**: Increasing
- **Interpretation**: The agent accumulates more reward → more accurate imitation of expert behavior.

> 📌 These trends confirm that DAgger enables continual policy improvement as more expert-labeled data is added.

---

### 🧾 5. Summary Table

| Metric                | Trend         | Insight                                             |
|-----------------------|---------------|-----------------------------------------------------|
| Episode Length        | 📈 Increasing  | Agent stabilizes, survives longer in the environment |
| Cumulative Reward     | 📈 Increasing  | Agent is achieving higher returns via better control |
| Expert Query Usage    | ✔️ Iterative   | Enables training on diverse state distributions     |

---

### ✅ 6. Conclusion

- DAgger significantly improves agent performance by avoiding distribution drift.
- The agent gradually learns from its own mistakes by querying the expert on its visited states.
- Hopper-v4 results show strong trends toward improved stability and reward accumulation.
- This method balances **exploration** and **imitation**, making it ideal for practical robotics and simulated control tasks.

> 🔍 Future improvements could explore early stopping, action noise, and ensemble experts.

## Commands:
	python train_agent.py --env_name <ENV> [optional tags]

    tensorboard --logdir ../data

### ⚙️ Configuration Tags

| **Tag**            | **Description**                                                                                   | **Possible Values**               |
|--------------------|---------------------------------------------------------------------------------------------------|-----------------------------------|
| `env_name`         | The name of the MuJoCo environment to train on                                                    | `Hopper-v4`, `Ant-v4`             |
| `no_gpu`           | Include this tag if you want to train on **CPU** instead of GPU (GPU not mandatory)              | *(flag only)*                     |
| `scalar_log_freq`  | Frequency at which to log scalar metrics (e.g., loss, reward)                                     | Positive integer, default: `1`    |
| `video_log_freq`   | Frequency at which to log videos showing model performance                                        | Positive integer, default: *None* |
| `n_vids_per_log`   | Number of trajectories to save per video log                                                      | Positive integer, default: `4`    |

