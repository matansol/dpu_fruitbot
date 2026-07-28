# Assessment-Loop Paper Data

## Overview
This folder contains the participant-level and trial-level datasets used for the figures and statistical analyses in the Assessment Policy Update paper submission.
The notebook Assessment-Loop plots.ipynb loads these CSV files, generates the paper plots, and runs the statistical analysist.

## Files
- user_results_compact.csv: One row per participant with demographics, aggregated performance metrics, and post-study questionnaire responses.
- choices_results_compact.csv: One row per choice trial with before/after agent scores, participant choice, correctness labels, and timing fields.
- Assessment-Loop plots.ipynb: Main analysis notebook that reproduces figures and statistical tests from the compact datasets.
- study_materials/: Supplementary study assets including survey PDFs and FruitBot demo videos used during participant tasks.
- study_materials/survey_minigrid_qualtrics.pdf: Survey form used for MiniGrid participants.
- study_materials/survey_fruitbot_qualtrics.pdf: Survey form used for FruitBot participants.
- study_materials/video_fruitbot_feedback_loop.mp4: Video showing the feedback interaction with the FruitBot agent.
- study_materials/video_fruitbot_demonstration.mp4: Video showing a comparison between the before and after agents.

## Plot/Analysis Coverage in Assessment-Loop plots.ipynb
- Correct choice by environment and group (local and generalized criteria).
- Correct choice by policy update direction (negative vs positive) for Same vs Salient-Contrast groups.
- Explanation satisfaction item plots and internal consistency (Cronbach alpha).
- Choice-time comparisons by environment and group.
- Final-agent analyses (count by rank, final agent score, and user evaluation of final agent).
- AI familiarity item analysis by environment/group.

## Data Linkage
The two compact datasets are linked by user_id.
Each user_id appears once in user_results_compact.csv and can appear multiple times in choices_results_compact.csv.

## Column Dictionary: user_results_compact.csv
- user_id: Participant identifier used to link participant-level rows to trial-level rows across files.
- group_label: Experimental condition label assigned to the participant (Control, Random, Same, or Salient-Contrast).
- environment: Environment associated with the participant analysis row (MiniGrid or FruitBot).
- number_of_choices: Number of episodes in which the participant provided feedback and then chose whether to accept or reject the update.
- age: Self-reported participant age.
- gender: Self-reported participant gender.
- education: Self-reported participant education level.
- familiar_with_AI: Response to "How familiar are you with Artificial Intelligence?" on a 1-7 scale.
- final_agent_rank: Rank from 1 to 6 among agents in the same environment, based on mean score across the evaluation suite.
- final_agent_mean_score: Mean score of the participant's final selected agent.
- mean_good_choice_local: Participant-level mean of local correctness across relevant choices.
- mean_good_choice_global: Participant-level mean of generalized correctness across relevant choices.
- mean_good_choice_demo: Participant-level mean of demonstration-based correctness across relevant choices.
- AI_1: Response to the statement "I believe AI will improve my life".
- AI_2: Response to the statement "I believe that AI will improve my work".
- AI_3: Response to the statement "I think I will use AI technology in the future".
- AI_4: Response to the statement "I think AI technology is a threat to humans".
- AI_5: Response to the statement "I think AI technology is positive for humanity".
- evaluate_final_agent: Response to "How do you evaluate the agent performance compared to yours?" on a 1-7 scale.
- ES_understandable: Participant response to whether explanations were clear and understandable.
- ES_overwhelming: Participant response to whether explanations felt too detailed/overwhelming.
- ES_feedback_related: Participant response to whether explanations connected feedback to behavior change.
- ES_helpful: Participant response to whether visual policy-change demonstrations were helpful.
- ES_improvement: Participant response to whether their feedback improved agent performance.
- ES_combined: Composite explanation-satisfaction score computed from selected ES items (including reverse scoring of ES_overwhelming).

## Column Dictionary: choices_results_compact.csv
- user_id: Participant identifier that links each trial to user_results_compact.csv.
- environment: Environment in which the choice trial was performed (MiniGrid or FruitBot).
- group_label: Experimental condition of the participant for this trial.
- prev_agent: Identifier/index of the pre-update (baseline) agent shown in the trial.
- updated_agent: Identifier/index of the post-update agent shown in the trial.
- prev_agent_mean_score: Mean model score for the pre-update agent.
- update_agent_mean_score: Mean model score for the updated agent.
- prev_agent_feedback_score: Score achieved by the previous agent on the episode for which the participant provided feedback.
- updated_agent_feedback_score: Score achieved by the updated agent on the episode for which the participant provided feedback.
- choice: Binary indicator where 1 means the participant accepted the update and 0 means the participant rejected the update.
- good_choice_global: Binary indicator of whether the participant selected the globally better option.
- good_choice_local: Binary indicator of whether the participant selected the feedback-locally better option.
- choice_time: Time taken to make the trial choice.
- good_choice_demo: Binary indicator of whether the agent selected by the participant was better in the demonstration context shown to the participant.
- prev_agent_demo_score: Demonstration-context score for the previous agent on this trial.
- update_agent_demo_score: Demonstration-context score for the updated agent on this trial.
- improved_from_prev_feedback: Binary flag for whether the updated agent scored better than the previous agent in the feedback context.
- improved_from_prev_mean: Binary flag for whether the updated agent scored better than the previous agent across all contexts (mean score).
- feedback_count: Count of correction feedback items provided by the participant in this episode.
