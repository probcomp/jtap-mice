"""
Combined metrics computation across multiple trials.

This module contains functions for computing metrics across multiple trials
by concatenating data from all trials and treating them as one long trial.
"""

import numpy as np
from typing import Dict, NamedTuple, Optional, List
from .results import JTAP_Results
from .metrics import CorrelationMetrics, extract_decision_probabilities, extract_conditional_green_probabilities, calculate_decision_weights, AnalysisData
from jtap.utils.common_math import check_valid
from jtap.utils.stimuli import HumanData


class CombinedDecisionMetrics(NamedTuple):
    """Metrics comparing model and human decision patterns across all trials."""
    # RMSE loss over all timesteps
    rmse_loss: float
    
    # RMSE loss over occlusion periods only
    occ_rmse_loss: float
    
    # Analysis data for targeted analysis (occlusion frames only)
    targeted: AnalysisData
    
    # Analysis data for non-targeted analysis (all frames)
    non_targeted: AnalysisData


class CombinedJTAP_Metrics(NamedTuple):
    """Combined metrics across all trials for all model types."""
    model_metrics: CombinedDecisionMetrics
    frozen_metrics: CombinedDecisionMetrics
    decaying_metrics: CombinedDecisionMetrics


def compute_combined_metrics(trial_results: Dict[str, JTAP_Results],
                           jtap_stimuli: Dict[str, object],
                           ignore_uncertain_line: bool = True,
                           include_partial_occlusion: bool = True) -> CombinedJTAP_Metrics:
    """
    Compute combined metrics across all trials by concatenating data.
    
    Args:
        trial_results: Dictionary mapping trial names to JTAP_Results objects
        jtap_stimuli: Dictionary mapping trial names to jtap_stimulus objects
        ignore_uncertain_line: If True, ignore uncertain periods in RMSE calculation
        include_partial_occlusion: If True, include partial occlusion in analysis
        
    Returns:
        CombinedJTAP_Metrics object with combined metrics across all trials
    """
    if not trial_results:
        raise ValueError("trial_results dictionary cannot be empty")
    
    if not jtap_stimuli:
        raise ValueError("jtap_stimuli dictionary cannot be empty")
    
    # Collect data from all trials
    model_distributions_list = []
    frozen_distributions_list = []
    decaying_distributions_list = []
    human_distributions_list = []
    human_keypresses_list = []
    occlusion_frames_list = []
    
    current_offset = 0
    
    for trial_name, jtap_results in trial_results.items():
        if trial_name not in jtap_stimuli:
            raise ValueError(f"Trial {trial_name} not found in jtap_stimuli")
        
        # Get distributions from decisions
        decisions = jtap_results.jtap_decisions
        model_distributions_list.append(decisions.model_output)
        frozen_distributions_list.append(decisions.frozen_output)
        decaying_distributions_list.append(decisions.decaying_output)
        
        # Get human data from stimulus
        stimulus = jtap_stimuli[trial_name]
        if stimulus.human_data is None:
            raise ValueError(f"Human data is not available for trial {trial_name}")
        
        human_data = stimulus.human_data
        human_distributions_list.append(human_data.human_output)
        human_keypresses_list.append(human_data.human_keypresses)
        
        # Get occlusion information from stimulus
        if include_partial_occlusion:
            occlusion_mask = stimulus.fully_occluded_bool | stimulus.partially_occluded_bool
        else:
            occlusion_mask = stimulus.fully_occluded_bool
        
        # Convert to frame indices and adjust for concatenation
        trial_occlusion_frames = np.where(occlusion_mask)[0] + current_offset
        occlusion_frames_list.extend(trial_occlusion_frames.tolist())
        
        # Update offset for next trial
        current_offset += len(decisions.model_output)
    
    # Concatenate all data
    all_model_distributions = np.concatenate(model_distributions_list)
    all_frozen_distributions = np.concatenate(frozen_distributions_list)
    all_decaying_distributions = np.concatenate(decaying_distributions_list)
    all_human_distributions = np.concatenate(human_distributions_list)
    all_human_keypresses = np.concatenate(human_keypresses_list, axis = 1)
    all_occlusion_frames = occlusion_frames_list
    
    # Create combined human data object
    combined_human_data = HumanData(
        human_output=all_human_distributions,
        human_keypresses=all_human_keypresses
    )
    
    # Compute metrics for each model type
    model_metrics = _compute_single_model_combined_metrics(
        all_model_distributions, combined_human_data, all_occlusion_frames, ignore_uncertain_line
    )
    
    frozen_metrics = _compute_single_model_combined_metrics(
        all_frozen_distributions, combined_human_data, all_occlusion_frames, ignore_uncertain_line
    )
    
    decaying_metrics = _compute_single_model_combined_metrics(
        all_decaying_distributions, combined_human_data, all_occlusion_frames, ignore_uncertain_line
    )
    
    return CombinedJTAP_Metrics(
        model_metrics=model_metrics,
        frozen_metrics=frozen_metrics,
        decaying_metrics=decaying_metrics
    )


def _compute_single_model_combined_metrics(model_distributions: np.ndarray,
                                         human_data: HumanData,
                                         occlusion_frames: List[int],
                                         ignore_uncertain_line: bool = True) -> CombinedDecisionMetrics:
    """
    Compute combined metrics for a single model type across all concatenated trials.
    """
    human_distributions = human_data.human_output
    
    # Calculate RMSE loss over all timesteps
    rmse_loss = _calculate_combined_rmse_loss(
        model_distributions, human_distributions, ignore_uncertain_line
    )
    
    # Calculate RMSE loss over occlusion periods only
    occ_rmse_loss = _calculate_combined_occlusion_rmse_loss(
        model_distributions, human_distributions, occlusion_frames, ignore_uncertain_line
    )
    
    # Compute targeted analysis (occlusion frames only)
    targeted_analysis = _compute_combined_analysis_data(
        model_distributions, human_distributions, human_data.human_keypresses, occlusion_frames
    )
    
    # Compute non-targeted analysis (all frames)
    non_targeted_analysis = _compute_combined_analysis_data(
        model_distributions, human_distributions, human_data.human_keypresses, None
    )
    
    return CombinedDecisionMetrics(
        rmse_loss=rmse_loss,
        occ_rmse_loss=occ_rmse_loss,
        targeted=targeted_analysis,
        non_targeted=non_targeted_analysis
    )


def _compute_combined_analysis_data(model_distributions: np.ndarray,
                                  human_distributions: np.ndarray,
                                  human_keypresses: np.ndarray,
                                  occlusion_frames: Optional[List[int]] = None) -> AnalysisData:
    """
    Compute analysis data for combined metrics (either targeted or non-targeted analysis).
    
    Args:
        model_distributions: Model probability distributions (T, 3)
        human_distributions: Human probability distributions (T, 3)
        human_keypresses: Human keypress data (num_participants, T)
        occlusion_frames: Optional list of frame indices to analyze (None for all frames)
        
    Returns:
        AnalysisData object with computed metrics
    """
    # Extract decision probabilities
    model_decision_probs, human_decision_probs = extract_decision_probabilities(
        model_distributions, human_distributions, occlusion_frames
    )
    
    # Extract conditional green probabilities
    model_green_given_decision, human_green_given_decision = extract_conditional_green_probabilities(
        model_distributions, human_distributions, occlusion_frames
    )
    
    # Calculate weights for correlation analysis
    decision_weights = calculate_decision_weights(human_keypresses, occlusion_frames)
    
    # Find valid data points for analysis
    valid_decision_mask = np.logical_and(
        check_valid(model_decision_probs),
        check_valid(human_decision_probs)
    )
    
    valid_conditional_mask = np.logical_and(
        np.logical_and(
            check_valid(model_green_given_decision),
            check_valid(human_green_given_decision)
        ),
        valid_decision_mask  # Also require valid decision probabilities
    )
    
    # Calculate correlations for decision probabilities
    decision_corr_calc = CorrelationMetrics(
        human_decision_probs[valid_decision_mask],
        model_decision_probs[valid_decision_mask]
    )
    decision_prob_corr = decision_corr_calc.simple_corr()
    
    # Calculate correlations for conditional green choices
    conditional_corr_calc = CorrelationMetrics(
        human_green_given_decision[valid_conditional_mask],
        model_green_given_decision[valid_conditional_mask],
        decision_weights[valid_conditional_mask]
    )
    conditional_green_corr = conditional_corr_calc.simple_corr()
    weighted_conditional_green_corr = conditional_corr_calc.weighted_corr()
    
    return AnalysisData(
        decision_prob_corr=decision_prob_corr,
        conditional_green_corr=conditional_green_corr,
        weighted_conditional_green_corr=weighted_conditional_green_corr,
        model_decision_probs=model_decision_probs,
        human_decision_probs=human_decision_probs,
        model_green_given_decision=model_green_given_decision,
        human_green_given_decision=human_green_given_decision,
        valid_decision_mask=valid_decision_mask,
        valid_conditional_mask=valid_conditional_mask,
        decision_weights=decision_weights
    )


def _calculate_combined_rmse_loss(model_distributions: np.ndarray,
                                human_distributions: np.ndarray,
                                ignore_uncertain_line: bool = True) -> float:
    """Calculate RMSE loss over combined data from all trials."""
    squared_diff = (model_distributions - human_distributions) ** 2
    
    if ignore_uncertain_line:
        squared_diff = squared_diff[:, :2]
    
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def _calculate_combined_occlusion_rmse_loss(model_distributions: np.ndarray,
                                          human_distributions: np.ndarray,
                                          occlusion_frames: List[int],
                                          ignore_uncertain_line: bool = True) -> Optional[float]:
    """Calculate RMSE loss over occlusion periods from combined data."""
    if not occlusion_frames:
        return None
    
    # Extract occlusion periods
    occlusion_indices = np.array(occlusion_frames)
    model_occ = model_distributions[occlusion_indices]
    human_occ = human_distributions[occlusion_indices]
    
    if len(model_occ) == 0:
        return None
    
    squared_diff = (model_occ - human_occ) ** 2
    
    if ignore_uncertain_line:
        squared_diff = squared_diff[:, :2]
    
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    
    return float(rmse)


