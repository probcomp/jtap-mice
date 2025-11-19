"""
Decision analysis functions for comparing model and human decision patterns.

This module contains functions for analyzing decision patterns, computing
conditional probabilities, and comparing model predictions with human behavior.
"""

import numpy as np
from typing import NamedTuple, Optional, List

from .decisions import JTAP_Decisions
from jtap.utils.stimuli import HumanData
from jtap.utils.common_math import check_valid


class CorrelationMetrics:
    """Class for computing correlations between model and human data."""
    
    def __init__(self, human_data: np.ndarray, model_data: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Initialize correlation metrics.
        
        Args:
            human_data: Human behavioral data
            model_data: Model prediction data
            weights: Optional weights for weighted correlation
        """
        self.human_data = human_data
        self.model_data = model_data
        self.weights = weights
    
    def simple_corr(self) -> float:
        """Compute simple Pearson correlation coefficient."""
        return float(np.corrcoef(self.human_data, self.model_data)[0, 1])
    
    def weighted_corr(self) -> float:
        """Compute weighted Pearson correlation coefficient."""
        if self.weights is None:
            return self.simple_corr()
        
        # Handle case where all weights are zero
        if np.sum(self.weights) == 0:
            return 0.0
        
        # Normalize weights
        w = self.weights / np.sum(self.weights)
        
        # Weighted means
        mean_human = np.sum(w * self.human_data)
        mean_model = np.sum(w * self.model_data)
        
        # Weighted covariance and variances
        cov = np.sum(w * (self.human_data - mean_human) * (self.model_data - mean_model))
        var_human = np.sum(w * (self.human_data - mean_human) ** 2)
        var_model = np.sum(w * (self.model_data - mean_model) ** 2)
        
        # Avoid division by zero
        denominator = np.sqrt(var_human * var_model)
        if denominator == 0:
            return 0.0
        
        return float(cov / denominator)


class AnalysisData(NamedTuple):
    """Data for a specific analysis type (targeted or non-targeted)."""
    # Decision probability correlations
    decision_prob_corr: float
    
    # Conditional green choice correlations
    conditional_green_corr: float
    weighted_conditional_green_corr: float
    
    # Raw probability arrays for further analysis
    model_decision_probs: np.ndarray
    human_decision_probs: np.ndarray
    model_green_given_decision: np.ndarray
    human_green_given_decision: np.ndarray
    
    # Valid data masks and weights
    valid_decision_mask: np.ndarray
    valid_conditional_mask: np.ndarray
    decision_weights: np.ndarray


class DecisionMetrics(NamedTuple):
    """Metrics comparing model and human decision patterns."""
    # RMSE loss over all timesteps
    rmse_loss: float
    
    # RMSE loss over occlusion periods only
    occ_rmse_loss: float
    
    # Analysis data for targeted analysis (occlusion frames only)
    targeted: AnalysisData
    
    # Analysis data for non-targeted analysis (all frames)
    non_targeted: AnalysisData

class JTAP_Metrics(NamedTuple):
    model_metrics: DecisionMetrics
    frozen_metrics: DecisionMetrics
    decaying_metrics: DecisionMetrics

def calculate_rmse_loss(model_distributions: np.ndarray,
                       human_distributions: np.ndarray, ignore_uncertain_line: bool = False) -> float:
    """
    Calculate global RMSE loss over all timesteps and channels.
    
    Args:
        model_distributions: Model probability distributions (T, 3)
        human_distributions: Human probability distributions (T, 3)
        ignore_uncertain_line: If True, ignore uncertain periods
    Returns:
        Global RMSE loss across all timesteps and channels
    """

    # Calculate squared differences for all elements
    squared_diff = (model_distributions - human_distributions) ** 2

    if ignore_uncertain_line:
        squared_diff = squared_diff[:, :2]
    
    # Calculate global MSE across all timesteps and channels
    mse = np.mean(squared_diff)
    
    # Calculate global RMSE
    rmse = np.sqrt(mse)
    
    return float(rmse)


def calculate_occlusion_rmse_loss(model_distributions: np.ndarray,
                                 human_distributions: np.ndarray,
                                 occlusion_mask: np.ndarray,
                                 ignore_uncertain_line: bool = False) -> float:
    """
    Calculate RMSE loss over occlusion periods only.
    
    Args:
        model_distributions: Model probability distributions (T, 3)
        human_distributions: Human probability distributions (T, 3)
        occlusion_mask: Boolean mask indicating occlusion periods (T,)
        ignore_uncertain_line: If True, ignore uncertain periods
    Returns:
        RMSE loss over occlusion periods only
    """
    # Extract occlusion periods
    model_occ = model_distributions[occlusion_mask]
    human_occ = human_distributions[occlusion_mask]
    
    # If no occlusion periods, return None
    if len(model_occ) == 0:
        return None
    
    # Calculate squared differences for occlusion periods
    squared_diff = (model_occ - human_occ) ** 2
    
    if ignore_uncertain_line:
        squared_diff = squared_diff[:, :2]
    
    # Calculate MSE over occlusion periods
    mse = np.mean(squared_diff)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    return float(rmse)


def extract_decision_probabilities(model_distributions: np.ndarray,
                                 human_distributions: np.ndarray,
                                 occlusion_frames: Optional[List[int]] = None) -> tuple:
    """
    Extract decision probabilities from model and human distributions.
    
    Args:
        model_distributions: Model probability distributions (T, 3)
        human_distributions: Human probability distributions (T, 3)  
        occlusion_frames: Optional list of frame indices to analyze (if None, use all)
        
    Returns:
        Tuple of (model_decision_probs, human_decision_probs) where decision_probs
        is P(Green or Red) = P(decision made)
    """
    if occlusion_frames is not None and len(occlusion_frames) > 0:
        frame_indices = np.array(occlusion_frames)
        model_dist = model_distributions[frame_indices]
        human_dist = human_distributions[frame_indices]
    else:
        model_dist = model_distributions
        human_dist = human_distributions
    
    # P(Green or Red) = P(decision made) = 1 - P(uncertain)
    model_decision_probs = np.sum(model_dist[:, :2], axis=1)
    human_decision_probs = np.sum(human_dist[:, :2], axis=1)
    
    return model_decision_probs, human_decision_probs


def extract_conditional_green_probabilities(model_distributions: np.ndarray,
                                          human_distributions: np.ndarray,
                                          occlusion_frames: Optional[List[int]] = None) -> tuple:
    """
    Extract conditional green choice probabilities: P(Green | Decision made).
    
    Args:
        model_distributions: Model probability distributions (T, 3)
        human_distributions: Human probability distributions (T, 3)
        occlusion_frames: Optional list of frame indices to analyze
        
    Returns:
        Tuple of (model_green_given_decision, human_green_given_decision)
    """
    model_decision_probs, human_decision_probs = extract_decision_probabilities(
        model_distributions, human_distributions, occlusion_frames
    )
    
    if occlusion_frames is not None and len(occlusion_frames) > 0:
        frame_indices = np.array(occlusion_frames)
        model_green_probs = model_distributions[frame_indices, 0]
        human_green_probs = human_distributions[frame_indices, 0]
    else:
        model_green_probs = model_distributions[:, 0]
        human_green_probs = human_distributions[:, 0]
    
    # P(Green | Decision) = P(Green) / P(Decision)
    model_green_given_decision = model_green_probs / model_decision_probs
    human_green_given_decision = human_green_probs / human_decision_probs
    
    return model_green_given_decision, human_green_given_decision


def calculate_decision_weights(human_keypresses: np.ndarray,
                              occlusion_frames: Optional[List[int]] = None) -> np.ndarray:
    """
    Calculate weights based on number of participants making decisions.
    
    Args:
        human_keypresses: Human keypress data (num_participants, T)
        occlusion_frames: Optional list of frame indices to analyze
        
    Returns:
        Array of weights for each time point
    """
    if occlusion_frames is not None and len(occlusion_frames) > 0:
        frame_indices = np.array(occlusion_frames)
        keypresses = human_keypresses[:, frame_indices]
    else:
        keypresses = human_keypresses
    
    # Count participants who made a decision (not uncertain) at each time point
    decision_counts = np.sum(keypresses != 2, axis=0)
    return decision_counts.astype(float)


def _compute_analysis_data(model_distributions: np.ndarray,
                          human_distributions: np.ndarray,
                          human_keypresses: np.ndarray,
                          occlusion_frames: Optional[List[int]] = None) -> AnalysisData:
    """
    Compute analysis data for either targeted or non-targeted analysis.
    
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


def analyze_single_trial_decisions(jtap_decisions: JTAP_Decisions,
                                 human_trial_data,
                                 jtap_stimulus,
                                 model_type: str = "model",
                                 occlusion_frames: Optional[List[int]] = None,
                                 ignore_uncertain_line: bool = False,
                                 rmse_only: bool = False) -> DecisionMetrics:
    """
    Analyze decision patterns for a single trial comparing model and human data.
    
    Args:
        jtap_decisions: JTAP model decisions and outputs
        human_trial_data: Human behavioral data for the trial
        jtap_stimulus: JTAP stimulus object containing occlusion information
        model_type: Which model to analyze ("model", "frozen", or "decaying")
        occlusion_frames: Optional list of frame indices to focus analysis on
        ignore_uncertain_line: If True, ignore uncertain periods
        rmse_only: If True, only compute RMSE losses and skip other computations
    Returns:
        DecisionMetrics containing correlation analysis results
    """

    assert isinstance(human_trial_data, HumanData), "human_trial_data must be a HumanData object"
    assert isinstance(jtap_decisions, JTAP_Decisions), "jtap_decisions must be a JTAP_Decisions object"

    # Get the appropriate model output based on model_type
    if model_type == "model":
        model_distributions = jtap_decisions.model_output
    elif model_type == "frozen":
        model_distributions = jtap_decisions.frozen_output
    elif model_type == "decaying":
        model_distributions = jtap_decisions.decaying_output
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    human_distributions = human_trial_data.human_output
    
    # Calculate RMSE loss over all timesteps
    rmse_loss = calculate_rmse_loss(model_distributions, human_distributions, ignore_uncertain_line)
    
    # Calculate RMSE loss over occlusion periods only
    occlusion_mask = jtap_stimulus.fully_occluded_bool | jtap_stimulus.partially_occluded_bool
    occ_rmse_loss = calculate_occlusion_rmse_loss(model_distributions, human_distributions, occlusion_mask, ignore_uncertain_line)
    
    if rmse_only:
        # Return DecisionMetrics with only RMSE values filled and dummy analysis data
        dummy_analysis = AnalysisData(
            decision_prob_corr=0.0,
            conditional_green_corr=0.0,
            weighted_conditional_green_corr=0.0,
            model_decision_probs=np.array([]),
            human_decision_probs=np.array([]),
            model_green_given_decision=np.array([]),
            human_green_given_decision=np.array([]),
            valid_decision_mask=np.array([]),
            valid_conditional_mask=np.array([]),
            decision_weights=np.array([])
        )
        return DecisionMetrics(
            rmse_loss=rmse_loss,
            occ_rmse_loss=occ_rmse_loss,
            targeted=dummy_analysis,
            non_targeted=dummy_analysis
        )
    
    # Compute targeted analysis (occlusion frames only)
    targeted_analysis = _compute_analysis_data(
        model_distributions, human_distributions, human_trial_data.human_keypresses, occlusion_frames
    )
    
    # Compute non-targeted analysis (all frames)
    non_targeted_analysis = _compute_analysis_data(
        model_distributions, human_distributions, human_trial_data.human_keypresses, None
    )
    
    return DecisionMetrics(
        rmse_loss=rmse_loss,
        occ_rmse_loss=occ_rmse_loss,
        targeted=targeted_analysis,
        non_targeted=non_targeted_analysis
    )


def jtap_compute_decision_metrics(jtap_decisions: JTAP_Decisions,
                      jtap_stimulus,
                      partial_occlusion_in_targeted_analysis: bool = True,
                      ignore_uncertain_line: bool = True,
                      rmse_only: bool = False) -> JTAP_Metrics:
    """
    Compare all three model types (model, frozen, decaying) against human data.
    
    Args:
        jtap_decisions: JTAP model decisions and outputs
        jtap_stimulus: JTAP stimulus object containing human data and occlusion info
        partial_occlusion_in_targeted_analysis: If True, include partial occlusion frames 
                                               in targeted analysis along with full occlusion
        ignore_uncertain_line: If True, ignore uncertain periods
    Returns:
        Dictionary mapping model_type -> DecisionMetrics
        
    Raises:
        ValueError: If human data is not available in the stimulus
    """
    assert isinstance(jtap_decisions, JTAP_Decisions), "jtap_decisions must be a JTAP_Decisions object"
    
    # Check if human data exists
    if jtap_stimulus.human_data is None:
        raise ValueError("Human data is not available in the stimulus. Cannot perform comparison.")
    
    human_trial_data = jtap_stimulus.human_data
    assert isinstance(human_trial_data, HumanData), "human_trial_data must be a HumanData object"
    
    # Handle occlusion frames based on boolean masks
    if partial_occlusion_in_targeted_analysis:
        # Include both partial and full occlusion frames
        occlusion_mask = jtap_stimulus.fully_occluded_bool | jtap_stimulus.partially_occluded_bool
    else:
        # Only include full occlusion frames
        occlusion_mask = jtap_stimulus.fully_occluded_bool
    
    # Convert boolean mask to frame indices
    occlusion_frames = np.where(occlusion_mask)[0].tolist()

    # Compute metrics for each model type
    model_metrics = analyze_single_trial_decisions(
        jtap_decisions, human_trial_data, jtap_stimulus, "model", occlusion_frames, ignore_uncertain_line, rmse_only
    )
    
    frozen_metrics = analyze_single_trial_decisions(
        jtap_decisions, human_trial_data, jtap_stimulus, "frozen", occlusion_frames, ignore_uncertain_line, rmse_only
    )
    
    decaying_metrics = analyze_single_trial_decisions(
        jtap_decisions, human_trial_data, jtap_stimulus, "decaying", occlusion_frames, ignore_uncertain_line, rmse_only
    )
    
    results = JTAP_Metrics(
        model_metrics=model_metrics,
        frozen_metrics=frozen_metrics,
        decaying_metrics=decaying_metrics
    )
    return results