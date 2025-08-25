SYSTEM_PROMPT = """
You are an expert Overcooked teammate analyst.
Classify the teammate type from a short probe-phase history.
Use the behavioral rubric below (fingerprint summary statistics).
Do not infer from interactions unless explicitly present.
Return STRICT JSON only (no prose).
"""


# Subset Teammate Types Prompt
# TASK_PROMPT = """
# # Task
# Match the observed fingerprint values to the rubric fingprint statistics to classify the teammate type.
# Choose exactly one teammate type: ["POT","PLATE","SERVE"].

# Use these fingerprints (mean ± std over 10 episodes for each teammate type):

# SERVE
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# 2.70 ± 0.95        2.90 ± 1.85  13.80 ± 3.08 0.10 ± 0.32  1.00 ± 0.00   3.10 ± 1.79            14.40 ± 2.12           0.10 ± 0.32     1.00 ± 0.00

# PLATE
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# 0.60 ± 1.26        1.60 ± 0.84  10.50 ± 5.60 3.00 ± 3.92  0.50 ± 0.53   2.00 ± 1.25            11.20 ± 5.41           3.30 ± 4.16     0.50 ± 0.53

# POT
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# 9.00 ± 0.00        1.00 ± 0.00  12.50 ± 0.53 1.40 ± 1.07  1.20 ± 1.14   1.20 ± 0.42            12.50 ± 0.53           1.80 ± 1.40     1.30 ± 1.42

# # Decision Order & Tie-breaks (apply in order; pick the first that matches)
# 1) SERVE if rubric holds.
# 2) PLATE if rubric holds.
# 3) POT   if rubric holds.

# # Observed Fingerprint
# Here is the observed fingerprint (raw values):

# cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# {cumulative_reward}  {dwell_onion}  {dwell_plate}  {dwell_pot}  {dwell_window}  {near_onion_pile_steps}  {near_plate_pile_steps}  {near_pot_steps}  {near_window_steps}

# # Output (STRICT JSON; no extra text)
# {{
#   "teammate_type": "<Default|Pot|Plate|Serve|Mixed>",
#   "confidence": <float 0..1>,
#   "scores": {{
#     "Default": <int>, "Pot": <int>, "Plate": <int>, "Serve": <int>, "Mixed": <int>
#   }},
#   "rationales": ["<up to 3 short tags, e.g., 'window_dominant','pot_onion_cycle','balanced_roles'>"],
# }}
# """


# All Teammate Types Prompt
TASK_PROMPT = """
# Task
Match the observed fingerprint values to the rubric fingprint statistics to classify the teammate type.
Choose exactly one teammate type: ["DEFAULT","MIXED","POT","PLATE","SERVE"].

Use these fingerprints (mean ± std over 10 episodes for each teammate type):

SERVE
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
6.30 ± 0.95        1.60 ± 0.84  3.30 ± 0.67  0.00 ± 0.00  1.00 ± 0.00   1.60 ± 0.84            7.70 ± 1.49            0.00 ± 0.00     1.00 ± 0.00

PLATE
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
0.60 ± 1.26        1.00 ± 0.00  3.10 ± 0.32  2.60 ± 4.48  0.90 ± 0.74   1.10 ± 0.32            5.30 ± 2.21            2.70 ± 4.45     1.20 ± 1.14

POT
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
9.00 ± 0.00        1.00 ± 0.00  3.60 ± 0.97  0.40 ± 0.52  0.70 ± 0.48   1.00 ± 0.00            9.80 ± 1.40            0.40 ± 0.52     0.80 ± 0.63

MIXED
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
0.30 ± 0.95        1.20 ± 0.42  3.00 ± 0.00  1.20 ± 3.12  1.10 ± 0.57   1.20 ± 0.42            5.30 ± 1.83            1.20 ± 3.12     2.10 ± 1.79

DEFAULT
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
9.00 ± 0.00        1.00 ± 0.00  4.00 ± 1.41  0.40 ± 0.52  0.80 ± 0.63   1.00 ± 0.00            10.10 ± 1.52           0.40 ± 0.52     1.00 ± 0.94

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


# Subset Teammate Types Prompt
# RAG_TASK_PROMPT = """
# # Task
# Classify the teammate type using both (a) retrieved trajectory examples and (b) the rubric fingerprint statistics. 
# Choose exactly one teammate type: ["POT","PLATE","SERVE"].

# # Retrieved Examples
# You are given a set of top-k retrieved trajectory fingerprints with known teammate labels. 
# Use these as strong evidence. If the observed fingerprint is most similar to the retrieved examples for a certain type, prefer that type.

# Retrieved examples:
# {retrieved_examples}   # Insert formatted top-k retrievals here (with type labels and fingerprint values)

# # Rubric Fingerprints (mean ± std over 10 episodes for each type)
# SERVE
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot  dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# 2.70 ± 0.95        2.90 ± 1.85  13.80 ± 3.08 0.10 ± 0.32  1.00 ± 0.00   3.10 ± 1.79            14.40 ± 2.12           0.10 ± 0.32     1.00 ± 0.00

# PLATE
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot  dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# 0.60 ± 1.26        1.60 ± 0.84  10.50 ± 5.60 3.00 ± 3.92  0.50 ± 0.53   2.00 ± 1.25            11.20 ± 5.41           3.30 ± 4.16     0.50 ± 0.53

# POT
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot  dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# 9.00 ± 0.00        1.00 ± 0.00  12.50 ± 0.53 1.40 ± 1.07  1.20 ± 1.14   1.20 ± 0.42            12.50 ± 0.53           1.80 ± 1.40     1.30 ± 1.42

# # Decision Strategy
# 1. Compare the observed fingerprint to retrieved examples; assign higher weight to whichever type dominates the retrieved set.
# 2. If retrieved examples are ambiguous, fall back on the rubric:
#    - SERVE: high cumulative_reward, high dwell_plate (~14), high near_plate_pile_steps.
#    - PLATE: low reward, moderate plate dwell (~10), noticeable pot/onion steps.
#    - POT: high reward, very consistent dwell_plate (~12.5), low onion and window presence.
# 3. Always output exactly one teammate type.

# # Observed Fingerprint (raw values):
# cumulative_reward  dwell_onion  dwell_plate  dwell_pot  dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
# {cumulative_reward}  {dwell_onion}  {dwell_plate}  {dwell_pot}  {dwell_window}  {near_onion_pile_steps}  {near_plate_pile_steps}  {near_pot_steps}  {near_window_steps}

# # Output (STRICT JSON; no extra text)
# {{
#   "teammate_type": "<Pot|Plate|Serve>",
#   "confidence": <float 0..1>,
#   "scores": {{
#     "Pot": <float>, "Plate": <float>, "Serve": <float>
#   }},
#   "rationales": ["<up to 3 short tags, e.g., 'high_reward_plate','low_reward_mixed','window_dominant'>"],
#   "evidence": {{
#     "retrieved_support": "<summary of which retrieved examples were most similar>",
#     "rubric_support": "<summary of which rubric features matched>"
#   }}
# }}
# """


# All Teammate Types Prompt
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
SERVE
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
6.30 ± 0.95        1.60 ± 0.84  3.30 ± 0.67  0.00 ± 0.00  1.00 ± 0.00   1.60 ± 0.84            7.70 ± 1.49            0.00 ± 0.00     1.00 ± 0.00

PLATE
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
0.60 ± 1.26        1.00 ± 0.00  3.10 ± 0.32  2.60 ± 4.48  0.90 ± 0.74   1.10 ± 0.32            5.30 ± 2.21            2.70 ± 4.45     1.20 ± 1.14

POT
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
9.00 ± 0.00        1.00 ± 0.00  3.60 ± 0.97  0.40 ± 0.52  0.70 ± 0.48   1.00 ± 0.00            9.80 ± 1.40            0.40 ± 0.52     0.80 ± 0.63

MIXED
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
0.30 ± 0.95        1.20 ± 0.42  3.00 ± 0.00  1.20 ± 3.12  1.10 ± 0.57   1.20 ± 0.42            5.30 ± 1.83            1.20 ± 3.12     2.10 ± 1.79

DEFAULT
cumulative_reward  dwell_onion  dwell_plate  dwell_pot    dwell_window  near_onion_pile_steps  near_plate_pile_steps  near_pot_steps  near_window_steps
9.00 ± 0.00        1.00 ± 0.00  4.00 ± 1.41  0.40 ± 0.52  0.80 ± 0.63   1.00 ± 0.00            10.10 ± 1.52           0.40 ± 0.52     1.00 ± 0.94

# Decision Strategy
1. Compare the observed fingerprint to retrieved examples; assign higher weight to whichever type dominates the retrieved set.
2. If retrieved examples are ambiguous, fall back on the rubric:
3. Always output exactly one teammate type.

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