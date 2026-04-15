import pandas as pd


def _reward_by_unique_progress_state(game_df: pd.DataFrame) -> pd.Series:
    """Count reward once per step using the last nonzero reward in that step."""
    if "reward" not in game_df.columns:
        return pd.Series(dtype=float)

    reward_df = game_df.copy()
    reward_df["reward"] = pd.to_numeric(reward_df["reward"], errors="coerce")
    if "step_count" not in reward_df.columns:
        return reward_df["reward"].dropna()

    reward_df["step_count"] = pd.to_numeric(reward_df["step_count"], errors="coerce")
    reward_df = reward_df.loc[reward_df["step_count"].notna()].copy()
    if reward_df.empty:
        return pd.Series(dtype=float)

    def one_reward_per_step(step_rewards: pd.Series) -> float:
        nonzero = step_rewards[step_rewards.ne(0)].dropna()
        if not nonzero.empty:
            return float(nonzero.iloc[-1])
        last = step_rewards.dropna()
        return float(last.iloc[-1]) if not last.empty else 0.0

    return reward_df.groupby("step_count", sort=False)["reward"].apply(
        one_reward_per_step
    )


def extract_game_features(game_data) -> dict:

    deduped_reward = _reward_by_unique_progress_state(game_data)
    cumulative_reward = float(deduped_reward.sum()) if not deduped_reward.empty else 0.0
    features = {
        "n_actions": int(game_data["action"].notna().sum()),
        "n_llm_calls": int(game_data["llm_response"].notna().sum())
        if "llm_response" in game_data.columns
        else 0,
        "saved_victims": int(game_data["saved_victims"].max()),
        "mean_reward": float(reward.mean()) if not reward.empty else 0.0,
        "total_reward": cumulative_reward,
        "cumulative_reward": cumulative_reward,
    }
    max_steps = game_data["step_count"].max()
    features["victims_per_step"] = (
        (features["saved_victims"] / max_steps) if max_steps else 0.0
    )
    return features
