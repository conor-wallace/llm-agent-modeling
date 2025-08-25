SYSTEM_PROMPT = """
You are an expert Overcooked teammate analyst.
Classify the teammate type from a short probe-phase history.
Use the behavioral rubric below (fingerprint summary statistics).
Do not infer from interactions unless explicitly present.
Return STRICT JSON only (no prose).
"""


TASK_PROMPT = """
# Task
Match the observed fingerprint values to the rubric fingprint statistics to classify the teammate type.
Choose exactly one teammate type: ["DEFAULT","MIXED","POT","PLATE","SERVE"].

Use these fingerprints (mean ± std over 10 episodes for each teammate type):

{behavior_rubric}

# Decision Order & Tie-breaks (apply in order; pick the first that matches)
1) SERVE if rubric holds.
2) PLATE if rubric holds.
3) POT   if rubric holds.
4) MIXED if rubric holds.
5) DEFAULT if rubric holds.

# Observed Fingerprint
Here is the observed fingerprint (raw values):

cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
{cumulative_reward}  {dwell_onion}  {dwell_plate}  {dwell_pot}  {dwell_window}  {near_onion_pile_steps}  {near_plate_pile_steps}  {near_pot_steps}  {near_window_steps}

# Output (STRICT JSON; no extra text)
{{
  "teammate_type": "<Default|Pot|Plate|Serve|Mixed>",
  "confidence": <float 0..1>,
  "scores": {{
    "Default": <int>, "Pot": <int>, "Plate": <int>, "Serve": <int>, "Mixed": <int>
  }},
  "rationales": ["<up to 3 short tags, e.g., 'window_dominant','pot_onion_cycle','balanced_roles'>"],
}}
"""


RAG_TASK_PROMPT = """
# Task
Classify the teammate type using both (a) retrieved trajectory examples and (b) the rubric fingerprint statistics. 
Choose exactly one teammate type: ["DEFAULT","MIXED","POT","PLATE","SERVE"].

# Retrieved Examples
You are given a set of top-k retrieved trajectory fingerprints with known teammate labels. 
Use these as strong evidence. If the observed fingerprint is most similar to the retrieved examples for a certain type, prefer that type.

Retrieved examples:
{retrieved_examples}

# Rubric Fingerprints (mean ± std over 10 episodes for each type)

{behavior_rubric}

# Decision Strategy
1. Compare the observed fingerprint to retrieved examples; assign higher weight to whichever type dominates the retrieved set.
2. Always choose the teammate type that is most supported by the retrieved examples (e.g., 2/3 retrieved samples).
3. If retrieved examples are ambiguous (e.g., 1/3 retrieved samples), combine evidence from the rubric with the first retrieved example.
4. Always output exactly one teammate type.

# Observed Fingerprint (raw values):
cumulative_reward  dwell_onion  dwell_plate  dwell_pot  dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
{cumulative_reward}  {dwell_onion}  {dwell_plate}  {dwell_pot}  {dwell_window}  {near_onion_pile_steps}  {near_plate_pile_steps}  {near_pot_steps}  {near_window_steps}

# Output (STRICT JSON; no extra text)
{{
  "teammate_type": "<Pot|Plate|Serve>",
  "confidence": <float 0..1>,
  "scores": {{
    "Pot": <float>, "Plate": <float>, "Serve": <float>
  }},
  "rationales": ["<up to 3 short tags, e.g., 'high_reward_plate','low_reward_mixed','window_dominant'>"],
  "evidence": {{
    "retrieved_support": "<summary of which retrieved examples were most similar>",
    "rubric_support": "<summary of which rubric features matched>"
  }}
}}
"""


QUERY_PROMPT = """
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
{cumulative_reward}  {dwell_onion}  {dwell_plate}  {dwell_pot}  {dwell_window}  {near_onion_pile_steps}  {near_plate_pile_steps}  {near_pot_steps}  {near_window_steps}
"""