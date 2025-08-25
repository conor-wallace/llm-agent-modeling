# plastic_agent.py
import os
import math
import numpy as np
import jax.numpy as jnp

from .base import BaseAgent  # your file above

TYPES3 = ["POT", "PLATE", "SERVE"]


class PlasticAgent(BaseAgent):
    """
    Minimal PLASTIC-Policy:
      - Prior over teammate types
      - Online Bayesian update using Gaussian likelihoods of short-horizon fingerprints
      - Switch to MAP best-response checkpoint after K probe steps (or keep mixing if desired)
    """

    def __init__(self, config, env, checkpoint_root, br_dirs_by_type, agent_idx):
        """
        Args:
          config: dict with keys:
            - K: int (probe horizon)
            - ACTIVATION: "tanh"/"relu" (used by BaseAgent)
            - FEATURES: list[str] features to score (e.g., ["cumulative_reward","near_plate_pile_steps","near_pot_steps","near_window_steps"])
            - GAUSS_STATS: dict[type][feature] = {"mean": float, "std": float}
            - ALPHA: float, temperature on log-likelihoods (e.g., 1.0)
            - PRIOR: optional dict[type]=float (sums to 1). If None, uniform
            - PROBE_TEAM_DIR: the team_dir to use during probe (e.g., a neutral/default policy for agent_1)
            - HARD_SWITCH: bool (True: use argmax policy after K; False: keep mixing)
          env: Overcooked env
          checkpoint_root: base folder with team subfolders
          br_dirs_by_type: dict[type] -> team_dir (folder containing the BR checkpoint for this agent_idx)
          agent_idx: 0 or 1
        """
        self.checkpoint_root = checkpoint_root
        self.br_dirs_by_type = br_dirs_by_type
        self.types = TYPES3
        self.features = config.get("FEATURES", ["cumulative_reward","near_plate_pile_steps","near_pot_steps","near_window_steps"])
        self.gauss_stats = config["GAUSS_STATS"]
        self.alpha = float(config.get("ALPHA", 1.0))
        self.prior = np.array([config.get("PRIOR", {}).get(t, 1.0/len(self.types)) for t in self.types], dtype=np.float64)
        self.probe_team_dir = config["PROBE_TEAM_DIR"]
        self.hard_switch = bool(config.get("HARD_SWITCH", True))

        # We start with the probe policy; BaseAgent will load it from PROBE_TEAM_DIR
        super().__init__(config, env, checkpoint_dir=checkpoint_root, team_dir=self.probe_team_dir, agent_idx=agent_idx)

        # running belief
        self.belief = self.prior.copy()   # shape (3,)
        self.decided_type = None          # set to a str in TYPES3 once we hard switch

    # --- util: load a different checkpoint on the fly ---
    def _load_br_checkpoint_for(self, teammate_type: str):
        team_dir = self.br_dirs_by_type[teammate_type]
        flat_path = os.path.join(self.checkpoint_root, team_dir) + f"/model_{self.agent_idx}.safetensors"
        # reuse BaseAgent.load_params logic
        from safetensors.flax import load_file
        flat_params = load_file(flat_path)

        self.network_params = {'params': {}}
        for key, value in flat_params.items():
            parts = key.split(',')
            if len(parts) == 3:
                collection, module, param = parts
                if module not in self.network_params[collection]:
                    self.network_params[collection][module] = {}
                self.network_params[collection][module][param] = value.squeeze(0)

    # --- likelihood for one type from current short-horizon fingerprint ---
    def _loglik_type(self, teammate_type: str, fp: dict) -> float:
        # Product of independent Gaussians -> sum of log pdfs
        stats = self.gauss_stats[teammate_type]
        logp = 0.0
        for feat in self.features:
            if feat not in fp: 
                continue
            x = float(fp[feat])
            mu = float(stats[feat]["mean"])
            sd = float(stats[feat]["std"])
            sd = max(sd, 1e-3)  # avoid zero
            # Gaussian log pdf
            logp += -0.5 * math.log(2*math.pi*sd*sd) - 0.5 * ((x - mu)/sd)**2
        return self.alpha * logp

    # --- external hook: get a short-horizon fingerprint from TeammateHistory ---
    def _fingerprint_now(self) -> dict:
        """
        Returns a dict of the same keys used in GAUSS_STATS / FEATURES.
        You likely already compute these in your pipeline; hook it up here.
        Minimal example uses TeammateHistory aggregated stats:
          self.hist.fingerprint() should return those fields over the last K steps.
        Replace this with your actual extractor.
        """
        return self.hist.fingerprint()  # <-- ensure this returns the keys listed in self.features

    def _update_belief(self):
        fp = self._fingerprint_now()
        loglikes = np.array([self._loglik_type(t, fp) for t in self.types], dtype=np.float64)
        # Bayes update in log space: log b' ∝ log b + log L
        log_post = np.log(self.belief + 1e-12) + loglikes
        # softmax normalize
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()
        self.belief = post

    def _maybe_switch_policy(self):
        if self.decided_type is not None:
            return
        # After K steps of probe, optionally hard switch to MAP best-response
        if self.hist.len() >= self.config["K"] and self.hard_switch:
            idx = int(self.belief.argmax())
            t_hat = self.types[idx]
            self._load_br_checkpoint_for(t_hat)
            self.decided_type = t_hat

    # --- main step ---
    def act(self, obs, prev_actions, prev_rewards):
        # push new transition into history (teammate on the "other" agent)
        obs_0 = obs["agent_0"]; obs_1 = obs["agent_1"]
        self.hist.push(
            obs_0, obs_1,
            prev_actions["agent_0"], prev_actions["agent_1"],
            prev_rewards["agent_0"], prev_rewards["agent_1"]
        )

        # Online belief update (always)
        self._update_belief()
        # Optionally hard-switch after K steps
        self._maybe_switch_policy()

        # Act with current network (probe until switch; then BR policy)
        obs_i = obs[f"agent_{self.agent_idx}"].flatten()
        pi, _ = self.network.apply(self.network_params, obs_i)

        # If you want *mixture-of-BR* instead of hard switch, uncomment below:
        # if not self.hard_switch:
        #     # compute action logits under each BR and average by belief
        #     logits_mix = None
        #     saved_params = self.network_params
        #     for t, w in zip(self.types, self.belief):
        #         self._load_br_checkpoint_for(t)
        #         pi_t, _ = self.network.apply(self.network_params, obs_i)
        #         logits_t = pi_t.logits_parameter()
        #         logits_mix = logits_t * w if logits_mix is None else logits_mix + w * logits_t
        #     self.network_params = saved_params  # restore
        #     pi = distrax.Categorical(logits=logits_mix)

        return pi

    # Optional: expose belief for logging
    def get_belief(self):
        return {t: float(p) for t, p in zip(self.types, self.belief)}
