# Verification: what was checked, and how

Every check below is an automated test in `tests/`, run with
`pytest -q`. They are marked so that what ran on generated signals and what
would need recordings can never be confused:

* `@pytest.mark.synthetic` — runs on generated data, no recordings required.
  **Every test in this repository currently carries this mark.**
* `@pytest.mark.realdata` — would require a Sleep-EDF data root. None are
  defined yet; the real-data evidence in this release comes from the pilot
  commands in the README, whose outputs are in `docs/`.
* `@pytest.mark.slow` — takes more than a few seconds (the training round-trip
  and the overfit check).

## Annotation alignment and label mapping

| what is asserted | test |
| --- | --- |
| R&K stage 4 folds into N3; W/1/2/R map as documented | `test_annotations.py::test_r_and_k_stage_four_folds_into_n3` |
| `Movement time` and `Sleep stage ?` are not labels and never become wake | `test_annotations.py::test_movement_and_unknown_are_not_labels` |
| a run annotation expands to exactly the epochs it covers | `test_annotations.py::test_runs_expand_to_the_epochs_they_cover` |
| a misaligned onset is reported, and only wholly covered epochs are labelled | `test_annotations.py::test_misaligned_annotations_are_recorded_not_rounded_away` |
| an epoch scored twice is counted, not silently overwritten | `test_annotations.py::test_overlapping_scoring_is_counted` |
| a generated hypnogram's stages come back exactly, holes included | `test_annotations.py::test_reading_a_generated_hypnogram_recovers_its_stages` |
| labels line up with the generated truth, epoch by epoch | `test_epochs.py::test_labels_line_up_with_the_generated_truth` |

## Units and the EDF header

| what is asserted | test |
| --- | --- |
| the header reports what the file declares (labels, rate, duration, start) | `test_edf_io.py::test_header_reports_what_the_file_declares` |
| the two-digit year uses the EDF convention (89 → 1989, not 2089) | `test_edf_io.py::test_two_digit_year_uses_the_edf_convention` |
| a file declaring millivolts comes back exactly 1,000× larger | `test_edf_io.py::test_microvolt_conversion_is_by_declared_unit` |
| an unrecognised unit raises rather than being scaled by a guess | `test_edf_io.py::test_unknown_unit_raises_rather_than_guessing` |
| a wrong sampling rate is refused, not silently resampled | `test_edf_io.py::test_wrong_sampling_rate_is_refused_not_resampled` |
| channels come back in the order asked for | `test_edf_io.py::test_channel_order_is_the_order_asked_for` |
| amplitudes survive the write/read round trip as scalp-EEG microvolts | `test_edf_io.py::test_amplitudes_survive_the_round_trip` |

## Gap handling and epoching

| what is asserted | test |
| --- | --- |
| epochs are `[2, 3000]` float32 | `test_epochs.py::test_epoch_shape_is_the_documented_one` |
| an excluded epoch leaves a visible hole in the epoch index | `test_epochs.py::test_excluded_epochs_leave_a_visible_hole` |
| adjacency is decidable from the indices, so nothing concatenates across a gap | `test_epochs.py::test_no_epoch_is_concatenated_across_a_gap` |
| the cache round-trips signals, indices, labels, channels and exclusions | `test_epochs.py::test_cache_round_trip_preserves_everything` |
| a flat channel is flagged | `test_epochs.py::test_flatline_is_flagged` |
| the sleep-window crop is opt-in, not the default | `test_epochs.py::test_sleep_window_is_an_option_not_the_default` |

## Participant separation

| what is asserted | test |
| --- | --- |
| no participant appears in two parts of a split | `test_splits.py::test_no_participant_appears_twice` |
| a split that shares a participant **cannot be constructed** | `test_splits.py::test_a_split_that_shares_a_participant_cannot_be_constructed` |
| the same seed gives the same split; a different seed does not | `test_splits.py::test_the_same_seed_gives_the_same_split` |
| split identity is of the participants, not the name | `test_splits.py::test_identity_is_of_the_people_not_the_name` |
| label budgets are nested and never touch validation or test | `test_splits.py::test_label_budgets_are_nested_and_touch_only_training` |
| an edited split file is detected on load | `test_splits.py::test_written_split_detects_editing` |
| train/val/test datasets are participant-disjoint, and normalisation saw neither val nor test | `test_pipeline.py::test_datasets_are_participant_disjoint` |
| class weights come from the training dataset only | `test_pipeline.py::test_class_weights_come_from_training_only` |

## Preprocessing

| what is asserted | test |
| --- | --- |
| statistics are fitted only on the participants given | `test_preprocess.py::test_statistics_are_fitted_only_on_the_participants_given` |
| relative amplitude between epochs survives the default normalisation | `test_preprocess.py::test_channel_normalisation_keeps_relative_amplitude` |
| per-epoch z-scoring demonstrably destroys amplitude (why it is not default) | `test_preprocess.py::test_per_epoch_zscore_removes_amplitude_as_documented` |
| normalisation round-trips through a file with a stable identity | `test_preprocess.py::test_normalisation_round_trips_through_a_file` |
| the band-pass removes drift and keeps the bands | `test_preprocess.py::test_bandpass_removes_drift_and_keeps_the_bands` |

## The model, and that it can be optimised

| what is asserted | test |
| --- | --- |
| shapes: five logits, a 128-d embedding, `[batch, 128, 31]` tokens | `test_models.py::test_shapes_are_the_documented_ones` |
| logits are unnormalised | `test_models.py::test_logits_are_unnormalised` |
| a wrong channel count is refused | `test_models.py::test_wrong_channel_count_is_refused` |
| the head is separable from the backbone (the seam D2/D3 need) | `test_models.py::test_the_head_is_separable_from_the_backbone` |
| the parameter count is exactly the documented 489,477 | `test_models.py::test_parameter_count_is_stable` |
| no BatchNorm anywhere | `test_models.py::test_batchnorm_is_not_used` |
| **D1 overfits a tiny synthetic batch**: loss → <0.05, accuracy 1.000 | `test_models.py::test_d1_can_overfit_a_tiny_batch` |

## D2: windows, masking and the temporal stack

| what is asserted | test |
| --- | --- |
| an excluded epoch is masked in every window that would contain it, never bridged | `test_d2.py::test_a_gap_is_masked_not_bridged` |
| a recording boundary is masked the same way as a gap | `test_d2.py::test_a_boundary_is_masked_the_same_way_as_a_gap` |
| every present position sits at exactly the expected distance from the centre | `test_d2.py::test_every_present_position_is_at_exactly_the_expected_distance` |
| the centre of a window is always present | `test_d2.py::test_the_centre_is_always_present` |
| an even context is refused, in the planner and in the model | `test_d2.py::test_even_context_is_refused`, `::test_even_context_model_is_refused` |
| with every neighbour masked, changing them does not move the logits at all | `test_d2.py::test_only_the_central_epoch_decides_when_context_is_absent` |
| with neighbours present, changing them does | `test_d2.py::test_a_present_neighbour_does_change_the_answer` |
| a window with a masked centre is refused | `test_d2.py::test_a_masked_centre_is_refused` |
| D2 reuses D1's encoder object and produces identical embeddings from it | `test_d2.py::test_d2_reuses_the_d1_encoder_unchanged` |
| the pre-computed-embedding path equals the full forward path | `test_d2.py::test_the_two_inference_paths_agree` |
| windows never cross a recording or a participant | `test_d2.py::test_windows_never_cross_a_recording_or_participant` |
| every dataset mask agrees with the stored epoch indices, position by position | `test_d2.py::test_dataset_masks_match_the_stored_epoch_indices` |
| an absent position is zero in the tensor and false in the mask | `test_d2.py::test_absent_positions_are_zero_and_flagged` |
| genuine-neighbour coverage is computed and in range | `test_d2.py::test_context_coverage_is_reported_and_sane` |
| **D1 and D2 get identical class counts and class weights** | `test_d2.py::test_class_weights_match_the_epoch_dataset` |
| the shuffled-neighbour control keeps the centre and the amount of context | `test_d2.py::test_shuffling_context_keeps_the_centre_and_the_amount_of_context` |
| the masking control leaves only the centre present | `test_d2.py::test_masking_context_leaves_only_the_centre` |
| masking reduces the model to its encoder on the central epoch | `test_d2.py::test_masking_context_reduces_the_model_to_its_encoder` |
| the two controls ask different questions of the same windows | `test_d2.py::test_the_two_controls_ask_different_questions` |
| the segment path gives the same logits as the window path | `test_d2.py::test_segment_forward_equals_window_forward` |
| the segment path gives the same probabilities end to end | `test_d2.py::test_segment_and_window_predictions_agree` |
| segments cover every centre exactly once per pass | `test_d2.py::test_segments_cover_every_centre_exactly_once` |
| `forward` accepts both dataset layouts | `test_d2.py::test_forward_accepts_both_dataset_layouts` |
| D2 trains through the segment path and reloads | `test_d2.py::test_d2_trains_through_the_segment_path` |
| **D2 overfits a tiny batch**: accuracy 1.000, loss < 0.05 | `test_d2.py::test_d2_can_overfit_a_tiny_batch` |

## Self-supervised pretraining, and the leak it must not have

| what is asserted | test |
| --- | --- |
| the mask hides whole patches, and the right number of them | `test_pretrain.py::test_the_mask_hides_whole_patches_and_the_right_number_of_them` |
| the mask differs between epochs in a batch | `test_pretrain.py::test_the_mask_differs_between_epochs_in_a_batch` |
| an impossible mask is refused | `test_pretrain.py::test_an_impossible_mask_is_refused` |
| **changing hidden values leaves the encoder's input identical** | `test_pretrain.py::test_hidden_values_never_reach_the_encoder` |
| **and leaves its embedding unchanged** | `test_pretrain.py::test_hidden_values_do_not_move_the_embedding` |
| hidden samples are replaced by the mask value, not attenuated | `test_pretrain.py::test_hidden_samples_are_replaced_not_attenuated` |
| visible samples are untouched | `test_pretrain.py::test_visible_samples_are_untouched` |
| the loss ignores visible samples entirely | `test_pretrain.py::test_the_loss_ignores_visible_samples` |
| the loss is exactly the mean squared error over hidden samples | `test_pretrain.py::test_the_loss_counts_masked_samples` |
| the loss can exclude samples marked invalid | `test_pretrain.py::test_the_loss_can_exclude_invalid_samples` |
| the patch is aligned to the encoder's tokens | `test_pretrain.py::test_the_patch_is_aligned_to_the_encoders_tokens` |
| the uncovered tail is never masked and never scored | `test_pretrain.py::test_the_uncovered_tail_is_never_masked_and_never_scored` |
| **the decoder can represent a 13 Hz rhythm** (spindles are 12–16 Hz) | `test_pretrain.py::test_the_reconstruction_can_represent_a_fast_rhythm` |
| pretraining does not change the encoder's architecture | `test_pretrain.py::test_pretraining_does_not_change_the_encoder_architecture` |
| the decoder is under a tenth of the encoder | `test_pretrain.py::test_the_decoder_is_small_relative_to_the_encoder` |
| reconstruction loss at least halves on a structured signal | `test_pretrain.py::test_reconstruction_learns_something_on_a_structured_signal` |
| pretraining settings do not change the epoch cache's identity | `test_pretrain.py::test_config_carries_the_pretraining_settings` |

The smoke run additionally asserts, on generated recordings, that the encoder saw
only training participants and that its checkpoint round-trips into a usable
backbone with the held-out participants refused.

## The command line exists and dispatches

| what is asserted | test |
| --- | --- |
| the module is the command line, with `main` and `build_parser` | `test_cli.py::test_the_module_is_the_command_line` |
| every subcommand is present | `test_cli.py::test_every_subcommand_is_present` |
| each dispatches to a function this module defines | `test_cli.py::test_every_subcommand_dispatches_to_a_function` |
| each has working `--help` | `test_cli.py::test_every_subcommand_has_help` |
| the argument names the README uses still parse | `test_cli.py::test_the_training_commands_take_the_arguments_the_readme_uses` |

These exist because of a real failure: an editing script wrote the trainer's
source over `cli.py` and the whole suite still passed, because nothing imported
the command line. Against the broken file all twenty-six of these fail.

## The repository contains what it claims to

| what is asserted | test |
| --- | --- |
| no source file is git-ignored | `test_packaging.py::test_no_source_file_is_git_ignored` |
| every subpackage imports | `test_packaging.py::test_every_subpackage_is_importable` |

These exist because of a real failure: a `.gitignore` line reading `data/` also
matched `src/sleepstatelab/data/`, so ten modules were pushed missing and the
linter — which honours `.gitignore` — was skipping the same directory, hiding
nine errors in them.

## Checkpoints

| what is asserted | test |
| --- | --- |
| a checkpoint records split identity, channels, training participants | `test_pipeline.py::test_train_checkpoint_reload_predict_and_score` |
| a reloaded model predicts identically to the one that was saved | same test |
| loading refuses a different channel order | `test_pipeline.py::test_checkpoint_refuses_a_different_channel_order` |
| loading refuses a different preprocessing identity | same test |
| a checkpoint records its architecture, so D1 and D2 rebuild correctly | `test_pipeline.py::test_train_checkpoint_reload_predict_and_score`, `test_d2.py` (via the smoke run) |
| prediction uses the checkpoint's own normalisation, not a re-fitted one | `test_pipeline.py::test_prediction_uses_the_checkpoints_own_normalisation` |

## Metrics

| what is asserted | test |
| --- | --- |
| a perfect prediction scores 1.0 and κ 1.0 | `test_metrics.py::test_perfect_prediction_scores_one` |
| a class absent from both truth and predictions does not count | `test_metrics.py::test_absent_class_with_no_predictions_does_not_count` |
| predicting a stage that never occurred **is** penalised | `test_metrics.py::test_predicting_a_stage_that_never_happened_is_penalised` |
| participant-averaged and pooled macro-F1 differ when sizes differ | `test_metrics.py::test_participant_average_differs_from_pooled_when_sizes_differ` |
| a constant predictor scores κ = 0 with high accuracy | `test_metrics.py::test_kappa_of_a_constant_predictor_is_zero` |
| κ matches a hand-worked example (0.400) | `test_metrics.py::test_kappa_matches_a_worked_example` |
| confusion rows are truth, columns are prediction | `test_metrics.py::test_confusion_rows_are_truth` |

## Baselines and features

| what is asserted | test |
| --- | --- |
| the class prior predicts the training distribution and is smoothed | `test_baselines.py::test_class_prior_predicts_the_training_distribution` |
| probability columns are remapped when a class is absent from training | `test_baselines.py::test_probability_columns_are_remapped_when_a_class_is_absent` |
| feature names and matrix widths agree, and all values are finite | `test_baselines.py::test_features_are_named_and_counted_consistently` |
| delta relative power separates a 1 Hz epoch from a 20 Hz one | `test_baselines.py::test_features_separate_a_slow_epoch_from_a_fast_one` |
| baselines are never fitted on the evaluation set | `test_baselines.py::test_run_baselines_never_fits_on_the_evaluation_set` |

## Configuration, devices, provenance

| what is asserted | test |
| --- | --- |
| an unknown configuration key is an error, not a silent default | `test_config_and_provenance.py::test_unknown_keys_are_rejected` |
| the configuration identity changes with the settings | `..::test_identity_changes_with_the_settings` |
| the preprocessing identity ignores training settings and reacts to filters | `..::test_preprocessing_identity_ignores_training_settings` |
| every shipped config file parses and implies 3,000 samples per epoch | `..::test_shipped_configs_parse` |
| CPU is always available and selectable | `..::test_cpu_is_always_available` |
| an absent accelerator raises rather than falling back | `..::test_an_absent_device_raises_rather_than_falling_back` |
| the run record carries config hash, split, channels, revision | `..::test_provenance_records_the_contract` |

## The synthetic end-to-end run

`sleepstatelab smoke` generates six participants as real EDF files and runs the
entire pipeline over them: discovery, pairing, header validation, epoching with
deliberate movement and unscored gaps, participant-disjoint splitting,
training-only normalisation (asserted not to have seen the test participant),
D1 training, checkpointing, reloading, prediction, baseline fitting, saved
predictions, and generated report tables — followed by the overfit check. It
exits non-zero if D1 fails to memorise a tiny batch.

Every artefact it writes is named and labelled synthetic.
