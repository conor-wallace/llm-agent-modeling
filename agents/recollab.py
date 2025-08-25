import json
import os

from langchain.chat_models import init_chat_model
from langchain.schema.messages import SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .base import BaseAgent
from .features import fingerprint_from_window
from .history import TeammateHistory
from .obs import describe_overcooked_obs
from .prompts import SYSTEM_PROMPT, RAG_TASK_PROMPT, QUERY_PROMPT


# class ReCoLLABAgent(BaseAgent):
#     def __init__(
#         self, config, env, checkpoint_dir, team_dirs, agent_idx
#     ):
#         super().__init__(config, env, checkpoint_dir, team_dirs["DEFAULT"], agent_idx)

#         self.t = 0
#         self.team_dirs = team_dirs
#         self.pred_teammate_type = "DEFAULT"
#         self.hist = TeammateHistory(K=self.config["K"])
#         self.db = Chroma(
#             collection_name=self.config["DATABASE_NAME"],
#             embedding_function=OpenAIEmbeddings(model=self.config["EMBEDDING_MODEL"]),
#             persist_directory=self.config["DATABASE_DIR"],
#         )

#     def act(self, obs, prev_actions, prev_rewards):
#         obs_0 = obs["agent_0"]
#         obs_1 = obs["agent_1"]
#         prev_action_0 = prev_actions["agent_0"]
#         prev_action_1 = prev_actions["agent_1"]
#         prev_reward_0 = prev_rewards["agent_0"]
#         prev_reward_1 = prev_rewards["agent_1"]

#         self.hist.push(
#             obs_0,
#             obs_1,
#             prev_action_0,
#             prev_action_1,
#             prev_reward_0,
#             prev_reward_1
#         )

#         if self.hist.loaded and self.t <= self.config["K"]:
#             agent_features_dict_0 = fingerprint_from_window(
#                 self.hist.agent_0_obs,
#                 self.hist.agent_0_actions,
#                 self.hist.agent_0_rewards
#             )
        
#             query = QUERY_PROMPT.format(
#                 cumulative_reward=f"{agent_features_dict_0["cumulative_reward"]:.2f}",
#                 dwell_onion=f"{agent_features_dict_0["dwell_onion"]:.2f}",
#                 dwell_plate=f"{agent_features_dict_0["dwell_plate"]:.2f}",
#                 dwell_pot=f"{agent_features_dict_0["dwell_pot"]:.2f}",
#                 dwell_window=f"{agent_features_dict_0["dwell_window"]:.2f}",
#                 near_onion_pile_steps=f"{agent_features_dict_0["near_onion_pile_steps"]:.2f}",
#                 near_plate_pile_steps=f"{agent_features_dict_0["near_plate_pile_steps"]:.2f}",
#                 near_pot_steps=f"{agent_features_dict_0["near_pot_steps"]:.2f}",
#                 near_window_steps=f"{agent_features_dict_0["near_window_steps"]:.2f}"
#             )

#             # Use the query to retrieve relevant information from the database
#             retrieved_docs = self.db.similarity_search(query, k=1)

#             if retrieved_docs:
#                 doc = retrieved_docs[0]

#                 print(doc)

#                 self.pred_teammate_type = doc.metadata["teammate_type"]
#                 self.team_dir = self.team_dirs.get(self.pred_teammate_type)
#                 self.load_params()

#         self.t += 1

#         obs_i = obs[f"agent_{self.agent_idx}"].flatten()
#         pi, _ = self.network.apply(self.network_params, obs_i)
#         return pi


class ReCoLLABAgent(BaseAgent):
    def __init__(
        self, config, env, checkpoint_dir, team_dirs, agent_idx
    ):
        super().__init__(config, env, checkpoint_dir, team_dirs["DEFAULT"], agent_idx)

        if config["USE_ALL_TEAMMATES"]:
            rubric_path = os.path.join(config['RUBRIC_PATH'], f'ippo_ff_overcooked_{config["LAYOUT_NAME"]}', 'behavior_rubric.txt')
        else:
            rubric_path = os.path.join(config['RUBRIC_PATH'], f'ippo_ff_overcooked_{config["LAYOUT_NAME"]}', 'behavior_rubric_subset.txt')

        with open(rubric_path, 'r') as f:
            self.behavior_rubric = f.read()

        self.t = 0
        self.team_dirs = team_dirs
        self.pred_teammate_type = "DEFAULT"
        self.hist = TeammateHistory(K=self.config["K"])
        self.model = init_chat_model(self.config["MODEL_ID"])

        database_dir = os.path.join(config['DATABASE_PATH'], f'ippo_ff_overcooked_{config["LAYOUT_NAME"]}')
        database_name = f'ippo_ff_overcooked_{config["LAYOUT_NAME"]}'

        self.db = Chroma(
            collection_name=database_name,
            embedding_function=OpenAIEmbeddings(model=self.config["EMBEDDING_MODEL"]),
            persist_directory=database_dir,
        )

    def act(self, obs, prev_actions, prev_rewards):
        obs_0 = obs["agent_0"]
        obs_1 = obs["agent_1"]
        prev_action_0 = prev_actions["agent_0"]
        prev_action_1 = prev_actions["agent_1"]
        prev_reward_0 = prev_rewards["agent_0"]
        prev_reward_1 = prev_rewards["agent_1"]

        self.hist.push(
            obs_0,
            obs_1,
            prev_action_0,
            prev_action_1,
            prev_reward_0,
            prev_reward_1
        )

        if self.hist.loaded and self.t <= self.config["K"]:
            agent_features_dict_0 = fingerprint_from_window(
                self.hist.agent_0_obs,
                self.hist.agent_0_actions,
                self.hist.agent_0_rewards
            )
        
            query = QUERY_PROMPT.format(
                cumulative_reward=f"{agent_features_dict_0["cumulative_reward"]:.2f}",
                dwell_onion=f"{agent_features_dict_0["dwell_onion"]:.2f}",
                dwell_plate=f"{agent_features_dict_0["dwell_plate"]:.2f}",
                dwell_pot=f"{agent_features_dict_0["dwell_pot"]:.2f}",
                dwell_window=f"{agent_features_dict_0["dwell_window"]:.2f}",
                near_onion_pile_steps=f"{agent_features_dict_0["near_onion_pile_steps"]:.2f}",
                near_plate_pile_steps=f"{agent_features_dict_0["near_plate_pile_steps"]:.2f}",
                near_pot_steps=f"{agent_features_dict_0["near_pot_steps"]:.2f}",
                near_window_steps=f"{agent_features_dict_0["near_window_steps"]:.2f}"
            )

            # Use the query to retrieve relevant information from the database
            retrieved_docs = self.db.similarity_search(query, k=3)

            retrieval_results = ""
            for doc in retrieved_docs:
                retrieval_results += (
                    f"Retrieved Teammate Type: {doc.metadata['teammate_type']}\n"
                    f"Retrieved Content:{doc.page_content}\n"
                )

            rag_task_prompt = RAG_TASK_PROMPT.format(
                behavior_rubric=self.behavior_rubric,
                retrieved_examples=retrieval_results,
                cumulative_reward=f"{agent_features_dict_0["cumulative_reward"]:.2f}",
                dwell_onion=f"{agent_features_dict_0["dwell_onion"]:.2f}",
                dwell_plate=f"{agent_features_dict_0["dwell_plate"]:.2f}",
                dwell_pot=f"{agent_features_dict_0["dwell_pot"]:.2f}",
                dwell_window=f"{agent_features_dict_0["dwell_window"]:.2f}",
                near_onion_pile_steps=f"{agent_features_dict_0["near_onion_pile_steps"]:.2f}",
                near_plate_pile_steps=f"{agent_features_dict_0["near_plate_pile_steps"]:.2f}",
                near_pot_steps=f"{agent_features_dict_0["near_pot_steps"]:.2f}",
                near_window_steps=f"{agent_features_dict_0["near_window_steps"]:.2f}"
            )

            print(rag_task_prompt)

            response = self.model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    SystemMessage(content=rag_task_prompt)
                ]
            )

            print(response.content)

            self.pred_teammate_type = json.loads(response.content)["teammate_type"]
            self.team_dir = self.team_dirs.get(self.pred_teammate_type)
            self.load_params()

        self.t += 1

        obs_i = obs[f"agent_{self.agent_idx}"].flatten()
        pi, _ = self.network.apply(self.network_params, obs_i)
        return pi