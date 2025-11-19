from typing import NamedTuple, Tuple
import numpy as np

from jtap_mice.evaluation import JTAP_Beliefs
from jtap_mice.distributions import truncated_normal_sample, discrete_normal_sample
from jtap_mice.utils.stimuli import jtap_compute_outputs

class JTAP_Decision_Model_Hyperparams(NamedTuple):
    key_seed: int
    pseudo_participant_multiplier: int
    press_thresh_hyperparams: Tuple[float, float, float, float]
    tau_press_hyperparams: Tuple[float, float, np.ndarray]
    hysteresis_delay_hyperparams: Tuple[float, float, np.ndarray]
    regular_delay_hyperparams: Tuple[float, float, np.ndarray]
    starting_delay_hyperparams: Tuple[float, float, np.ndarray]

class JTAP_Decision_Model_Params(NamedTuple):
    pseudo_participant_multiplier: int
    press_threshes: np.ndarray
    tau_presses: np.ndarray
    hysteresis_delays: np.ndarray
    regular_delays: np.ndarray
    starting_delays: np.ndarray

class JTAP_Decisions(NamedTuple):
    num_jtap_runs: int
    num_pseudo_participants: int
    model_keypresses: np.ndarray
    frozen_keypresses: np.ndarray
    decaying_keypresses: np.ndarray
    model_output: np.ndarray # type: ignore
    frozen_output: np.ndarray # type: ignore
    decaying_output: np.ndarray # type: ignore

def jtap_compute_decisions(
    jtap_beliefs: JTAP_Beliefs,
    jtap_decision_model_hyperparams: JTAP_Decision_Model_Hyperparams,
    decision_model_version: str = "v3",
    remove_keypress_to_save_memory: bool = False
):
    """
    Function to sample and calculate keypress and scores based on beliefs.
    """

    # assert type checks
    assert isinstance(jtap_beliefs, JTAP_Beliefs), "jtap_beliefs must be a JTAP_Beliefs"
    assert isinstance(jtap_decision_model_hyperparams, JTAP_Decision_Model_Hyperparams), "jtap_decision_model_params must be a JTAP_Decision_Model_Hyperparams"

    num_jtap_runs = jtap_beliefs.num_jtap_runs
    pseudo_participant_multiplier = jtap_decision_model_hyperparams.pseudo_participant_multiplier
    #alias
    jdmhp = jtap_decision_model_hyperparams

    # Generate 6 different integer keys using the key_seed from the decision model params
    # We'll use numpy's SeedSequence to split the seed deterministically
    seed_seq = np.random.SeedSequence(jdmhp.key_seed)
    child_seeds = seed_seq.spawn(6)
    keys = [int(cs.generate_state(1)[0]) for cs in child_seeds]

    # Sample from truncated normal and discrete normal distributions using the generated keys
    # sample a total of pseudo_participant_multiplier times for each parameter
    press_threshes = truncated_normal_sample(keys[0], jdmhp.press_thresh_hyperparams, pseudo_participant_multiplier)
    tau_presses = discrete_normal_sample(keys[1], jdmhp.tau_press_hyperparams, pseudo_participant_multiplier)
    hysteresis_delays = discrete_normal_sample(keys[3], jdmhp.hysteresis_delay_hyperparams, pseudo_participant_multiplier)
    regular_delays = discrete_normal_sample(keys[4], jdmhp.regular_delay_hyperparams, pseudo_participant_multiplier)
    starting_delays = discrete_normal_sample(keys[5], jdmhp.starting_delay_hyperparams, pseudo_participant_multiplier)

    jtap_decision_model_params = JTAP_Decision_Model_Params(
        pseudo_participant_multiplier = pseudo_participant_multiplier,
        press_threshes = press_threshes,
        tau_presses = tau_presses,
        hysteresis_delays = hysteresis_delays,
        regular_delays = regular_delays,
        starting_delays = starting_delays
    )

    # print(f"Using {decision_model_version} decision model version")

    model_keypresses, frozen_keypresses, decaying_keypresses, num_pseudo_participants = jtap_compute_keypresses(
        jtap_beliefs=jtap_beliefs,
        jtap_decision_model_params=jtap_decision_model_params,
        decision_model_version=decision_model_version
    )

    jtap_decisions = JTAP_Decisions(
        num_jtap_runs = num_jtap_runs,
        num_pseudo_participants = num_pseudo_participants,
        model_keypresses = model_keypresses if not remove_keypress_to_save_memory else None,
        frozen_keypresses = frozen_keypresses if not remove_keypress_to_save_memory else None,
        decaying_keypresses = decaying_keypresses if not remove_keypress_to_save_memory else None,
        model_output = jtap_compute_outputs(model_keypresses), # type: ignore
        frozen_output = jtap_compute_outputs(frozen_keypresses), # type: ignore
        decaying_output = jtap_compute_outputs(decaying_keypresses), # type: ignore
    )

    return jtap_decisions, jtap_decision_model_params


def jtap_compute_keypresses(jtap_beliefs, jtap_decision_model_params, decision_model_version: str = "v3"):
    """
    v1 is old logic from CogSci 2025
    v2 is updated logic where starting delay does not process the beliefs internally
    v3 is updated logic where starting delay processes the beliefs internally, it acts more liek a reaction time delay

    Convert JTAP beliefs to keypresses for multiple pseudo-participants using vectorized operations.
    
    Args:
        jtap_beliefs: JTAP_Beliefs - contains model_beliefs, frozen_beliefs, decaying_beliefs
        jtap_decision_model_params: JTAP_Decision_Model_Params - decision model parameters
        decision_model_version: str - version of the decision model to use
    
    Returns:
        dict with keys 'raw', 'frozen', 'decaying', each containing:
            np.ndarray, shape (num_jtap_runs * pseudo_participant_multiplier, T) - keypresses
    """

    #assert type checks
    assert isinstance(jtap_beliefs, JTAP_Beliefs), "jtap_beliefs must be a JTAP_Beliefs"
    assert isinstance(jtap_decision_model_params, JTAP_Decision_Model_Params), "jtap_decision_model_params must be a JTAP_Decision_Model_Params"

    # Get parameters
    press_threshes = jtap_decision_model_params.press_threshes
    tau_presses = jtap_decision_model_params.tau_presses
    hysteresis_delays = jtap_decision_model_params.hysteresis_delays
    regular_delays = jtap_decision_model_params.regular_delays
    starting_delays = jtap_decision_model_params.starting_delays
    
    num_jtap_runs = jtap_beliefs.num_jtap_runs
    pseudo_participant_multiplier = len(press_threshes)
    
    # Prepare beliefs arrays with robust normalization to (N, T, 3)
    def normalize_beliefs(arr):
        a = np.asarray(arr)
        # First squeeze singleton dims
        a_sq = np.squeeze(a)
        if a_sq.ndim == 2:
            # (T,3) -> (1,T,3)
            assert a_sq.shape[1] == 3
            return a_sq[None, ...]
        if a_sq.ndim == 3 and a_sq.shape[-1] == 3:
            return a_sq
        # Try specific common patterns
        if a_sq.ndim == 4:
            if a_sq.shape[-2] == 1 and a_sq.shape[-1] == 3:
                return a_sq[:, :, 0, :]
            if a_sq.shape[-2] == 3 and a_sq.shape[-1] == 1:
                return a_sq[:, :, :, 0]
        raise ValueError(f"Unexpected beliefs shape {a.shape}; expected (N,T,3) up to singleton dims")

    model_beliefs = normalize_beliefs(jtap_beliefs.model_beliefs)
    frozen_beliefs = normalize_beliefs(jtap_beliefs.frozen_beliefs)
    decaying_beliefs = normalize_beliefs(jtap_beliefs.decaying_beliefs)

    T = model_beliefs.shape[1]

    assert model_beliefs.shape == (num_jtap_runs, T, 3)
    assert frozen_beliefs.shape == (num_jtap_runs, T, 3)
    assert decaying_beliefs.shape == (num_jtap_runs, T, 3)

    # Total number of independent runs
    num_pseudo_participants = num_jtap_runs * pseudo_participant_multiplier
    
    # Expand beliefs to match all parameter combinations
    # Each JTAP run gets replicated for each pseudo-participant
    expanded_model_beliefs = np.repeat(model_beliefs, pseudo_participant_multiplier, axis=0)  # (num_pseudo_participants, T, 3)
    expanded_frozen_beliefs = np.repeat(frozen_beliefs, pseudo_participant_multiplier, axis=0)
    expanded_decaying_beliefs = np.repeat(decaying_beliefs, pseudo_participant_multiplier, axis=0)
    
    # Expand parameters to match all JTAP runs
    # Each parameter set gets replicated for each JTAP run
    expanded_press_threshes = np.tile(press_threshes, num_jtap_runs)  # (num_pseudo_participants,)
    expanded_tau_presses = np.tile(tau_presses, num_jtap_runs)
    expanded_hysteresis_delays = np.tile(hysteresis_delays, num_jtap_runs)
    expanded_regular_delays = np.tile(regular_delays, num_jtap_runs)
    expanded_starting_delays = np.tile(starting_delays, num_jtap_runs)


    if decision_model_version == "v1":
        decision_model_fn = decision_model_v1
    elif decision_model_version == "v2":
        decision_model_fn = decision_model_v2
    elif decision_model_version == "v3":
        decision_model_fn = decision_model_v3
    elif decision_model_version == "v4":
        decision_model_fn = decision_model_v4
    elif decision_model_version == "v5":
        decision_model_fn = decision_model_v5
    else:
        raise ValueError(f"Invalid decision model version: {decision_model_version}")
    
    # Run vectorized decision model
    model_keypresses = decision_model_fn(
        beliefs_batch=expanded_model_beliefs,
        press_thresh_batch=expanded_press_threshes,
        tau_press_batch=expanded_tau_presses,
        hysteresis_delay_batch=expanded_hysteresis_delays,
        regular_delay_batch=expanded_regular_delays,
        starting_delay_batch=expanded_starting_delays
    )
        
    frozen_keypresses = decision_model_fn(
        beliefs_batch=expanded_frozen_beliefs,
        press_thresh_batch=expanded_press_threshes,
        tau_press_batch=expanded_tau_presses,
        hysteresis_delay_batch=expanded_hysteresis_delays,
        regular_delay_batch=expanded_regular_delays,
        starting_delay_batch=expanded_starting_delays
    )
        
    decaying_keypresses = decision_model_fn(
        beliefs_batch=expanded_decaying_beliefs,
        press_thresh_batch=expanded_press_threshes,
        tau_press_batch=expanded_tau_presses,
        hysteresis_delay_batch=expanded_hysteresis_delays,
        regular_delay_batch=expanded_regular_delays,
        starting_delay_batch=expanded_starting_delays
    )


    assert model_keypresses.shape == (num_pseudo_participants, T)
    assert frozen_keypresses.shape == (num_pseudo_participants, T)
    assert decaying_keypresses.shape == (num_pseudo_participants, T)

    return model_keypresses, frozen_keypresses, decaying_keypresses, num_pseudo_participants

def decision_model_v5(
    beliefs_batch,
    press_thresh_batch,
    tau_press_batch,
    hysteresis_delay_batch,
    regular_delay_batch,
    starting_delay_batch,
):
    """
    Vectorized decision model for keypresses based on beliefs.
    Rule change: Once a button is pressed, you will never release it to "no button" (2).
    When a release condition is met, switch to the other button ONLY if its belief is higher,
    else maintain the current button.
    """
    beliefs_batch = np.asarray(beliefs_batch)
    squeezed = np.squeeze(beliefs_batch)
    if squeezed.ndim == 2:
        assert squeezed.shape[1] == 3
        beliefs_batch = squeezed[None, ...]
    elif squeezed.ndim == 3 and squeezed.shape[-1] == 3:
        beliefs_batch = squeezed
    else:
        if beliefs_batch.ndim == 4:
            if beliefs_batch.shape[-2] == 1 and beliefs_batch.shape[-1] == 3:
                beliefs_batch = beliefs_batch[:, :, 0, :]
            elif beliefs_batch.shape[-2] == 3 and beliefs_batch.shape[-1] == 1:
                beliefs_batch = beliefs_batch[:, :, :, 0]
            else:
                alt = np.squeeze(beliefs_batch)
                if alt.ndim == 3 and alt.shape[-1] == 3:
                    beliefs_batch = alt
                else:
                    raise ValueError(
                        f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims"
                    )
        else:
            raise ValueError(
                f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims"
            )
    batch_size, T, _ = beliefs_batch.shape
    button_pressed = np.full(batch_size, 2, dtype=int)  # 2: No button, 1: Red, 0: Green
    red_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    green_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    release_accum_counter = np.zeros(batch_size, dtype=int)  # for release decisions
    red_wins_counter = np.zeros(batch_size, dtype=int)  # Track red wins when both should press
    green_wins_counter = np.zeros(batch_size, dtype=int)  # Track green wins when both should press

    decisions = np.full((batch_size, T), 2, dtype=int)
    t_indices = np.arange(T)[None, :] - regular_delay_batch[:, None]  # (batch_size, T)

    for t in range(T):
        t_adjusted = t_indices[:, t]
        valid_time_mask = (t_adjusted >= 0) & (t_adjusted < T)
        past_starting_delay = t >= starting_delay_batch
        can_act_mask = valid_time_mask & past_starting_delay

        safe_t_adjusted = np.asarray(np.clip(t_adjusted, 0, T - 1), dtype=int).reshape(batch_size)
        batch_indices = np.arange(batch_size, dtype=int)
        current_red_belief = np.where(valid_time_mask, beliefs_batch[batch_indices, safe_t_adjusted, 1], 0.0)
        current_green_belief = np.where(valid_time_mask, beliefs_batch[batch_indices, safe_t_adjusted, 0], 0.0)
        # Evidence accumulation
        red_crosses_threshold = (current_red_belief >= press_thresh_batch) & valid_time_mask
        green_crosses_threshold = (current_green_belief >= press_thresh_batch) & valid_time_mask
        red_evidence_accum_counter = np.where(red_crosses_threshold, red_evidence_accum_counter + 1, 0)
        green_evidence_accum_counter = np.where(green_crosses_threshold, green_evidence_accum_counter + 1, 0)
        should_press_red = red_evidence_accum_counter >= tau_press_batch
        should_press_green = green_evidence_accum_counter >= tau_press_batch

        no_button_mask = (button_pressed == 2)
        both_should_press = should_press_red & should_press_green & no_button_mask & can_act_mask

        # Win counters for both-should-press
        red_higher = current_red_belief > current_green_belief
        red_wins_counter = np.where(
            both_should_press & red_higher,
            red_wins_counter + 1,
            np.where(~both_should_press, 0, red_wins_counter),
        )
        green_wins_counter = np.where(
            both_should_press & ~red_higher,
            green_wins_counter + 1,
            np.where(~both_should_press, 0, green_wins_counter),
        )
        choose_red_by_wins = red_wins_counter > green_wins_counter
        choose_green_by_wins = green_wins_counter > red_wins_counter
        wins_tied = red_wins_counter == green_wins_counter

        # If no button, decide what to press (or remain no press)
        new_button_no_press = np.where(
            both_should_press,
            np.where(
                choose_red_by_wins,
                1,
                np.where(
                    choose_green_by_wins,
                    0,
                    np.where(wins_tied & red_higher, 1, 0),
                ),
            ),
            np.where(
                should_press_red & no_button_mask & can_act_mask,
                1,
                np.where(should_press_green & no_button_mask & can_act_mask, 0, 2),
            ),
        )

        button_pressed_mask = (button_pressed != 2)
        current_pressed_belief = np.where(
            button_pressed == 1, current_red_belief,
            np.where(button_pressed == 0, current_green_belief, 0.0)
        )
        # Who is the other button, and its belief?
        other_button = np.where(button_pressed == 1, 0, 1)  # if pressing red, other is green, and vice versa (and junk if button==2)
        current_other_belief = np.where(
            button_pressed == 1, current_green_belief,
            np.where(button_pressed == 0, current_red_belief, 0.0)
        )

        belief_above_thresh = current_pressed_belief >= press_thresh_batch
        hold_continue = belief_above_thresh & button_pressed_mask & can_act_mask

        belief_below_thresh = (current_pressed_belief < press_thresh_batch) & button_pressed_mask
        release_accum_counter = np.where(belief_below_thresh & can_act_mask, release_accum_counter + 1, 0)

        should_release_basic = button_pressed_mask & ~hold_continue & can_act_mask
        should_release = should_release_basic & (release_accum_counter >= tau_press_batch)

        # New rule: When it's time to release, never go to "no press".
        # Instead, switch to the other button ONLY if its belief is strictly higher; else, maintain.
        # If switched, reset evidence counter and release counter.

        switch_to_other = should_release & (current_other_belief > current_pressed_belief)
        new_button_release = np.where(
            switch_to_other,
            other_button,
            button_pressed
        )

        # Update button state:  
        button_pressed = np.where(no_button_mask, new_button_no_press, new_button_release)

        # Reset release accumulator if switching buttons or belief now above threshold
        new_press_mask = (no_button_mask & (new_button_no_press != 2)) | (switch_to_other)
        # In this version, we only ever reset release_accum_counter if we've actually changed state or if pressed-button now above threshold
        release_accum_counter = np.where(new_press_mask | belief_above_thresh, 0, release_accum_counter)

        # Set decisions for this timestep (default to no press if can't act)
        decisions[:, t] = np.where(can_act_mask, button_pressed, 2)
    return decisions
    
    return decisions

def decision_model_v4(beliefs_batch, 
                                      press_thresh_batch, 
                                      tau_press_batch, 
                                      hysteresis_delay_batch,
                                      regular_delay_batch,
                                      starting_delay_batch):
    """
    Vectorized decision model for keypresses based on beliefs.
    
    Args:
        beliefs_batch: np.ndarray, shape (batch_size, T, 3) - beliefs over time for all runs
        press_thresh_batch: np.ndarray, shape (batch_size,) - threshold for pressing
        tau_press_batch: np.ndarray, shape (batch_size,) - time steps before pressing
        hysteresis_delay_batch: None (not used)
        regular_delay_batch: np.ndarray, shape (batch_size,) - regular delay for tau_delay
        starting_delay_batch: None (not used)
    
    Returns:
        np.ndarray, shape (batch_size, T) - button presses (0=green, 1=red, 2=no press)
    """
    # Normalize beliefs_batch to shape (batch_size, T, 3). Single JTAP run or
    # single particle configurations can leave stray singleton dimensions.
    beliefs_batch = np.asarray(beliefs_batch)
    # Aggressively squeeze singleton dimensions and then normalize to (B,T,3)
    squeezed = np.squeeze(beliefs_batch)
    if squeezed.ndim == 2:
        # (T,3) -> (1,T,3)
        assert squeezed.shape[1] == 3
        beliefs_batch = squeezed[None, ...]
    elif squeezed.ndim == 3 and squeezed.shape[-1] == 3:
        beliefs_batch = squeezed
    else:
        # Handle specific common 4D cases before erroring
        if beliefs_batch.ndim == 4:
            if beliefs_batch.shape[-2] == 1 and beliefs_batch.shape[-1] == 3:
                beliefs_batch = beliefs_batch[:, :, 0, :]
            elif beliefs_batch.shape[-2] == 3 and beliefs_batch.shape[-1] == 1:
                beliefs_batch = beliefs_batch[:, :, :, 0]
            else:
                alt = np.squeeze(beliefs_batch)
                if alt.ndim == 3 and alt.shape[-1] == 3:
                    beliefs_batch = alt
                else:
                    raise ValueError(f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims")
        else:
            raise ValueError(f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims")

    batch_size, T, _ = beliefs_batch.shape
    
    # Initialize state arrays
    button_pressed = np.full(batch_size, 2, dtype=int)  # 2: No button, 1: Red, 0: Green
    red_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    green_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    release_accum_counter = np.zeros(batch_size, dtype=int)  # New accumulator for release decisions
    red_wins_counter = np.zeros(batch_size, dtype=int)  # Track red wins when both should press
    green_wins_counter = np.zeros(batch_size, dtype=int)  # Track green wins when both should press
    
    decisions = np.full((batch_size, T), 2, dtype=int)
    
    # Precompute time adjustments for all timesteps and batches
    t_indices = np.arange(T)[None, :] - regular_delay_batch[:, None]  # (batch_size, T)
    
    for t in range(T):
        # Get adjusted time indices for this timestep
        t_adjusted = t_indices[:, t]  # (batch_size,)
        
        # Create masks for valid time indices
        valid_time_mask = (t_adjusted >= 0) & (t_adjusted < T)
        past_starting_delay = t >= starting_delay_batch
        can_act_mask = valid_time_mask & past_starting_delay
        
        # Get current beliefs (handle invalid indices with 0s)
        # Ensure 1D integer indices with matching shape
        safe_t_adjusted = np.asarray(np.clip(t_adjusted, 0, T - 1), dtype=int).reshape(batch_size)
        batch_indices = np.arange(batch_size, dtype=int)
        
        # Extract beliefs for evidence accumulation (can happen during starting delay)
        current_red_belief = np.where(valid_time_mask, 
                                    beliefs_batch[batch_indices, safe_t_adjusted, 1], 
                                    0.0)
        current_green_belief = np.where(valid_time_mask,
                                      beliefs_batch[batch_indices, safe_t_adjusted, 0],
                                      0.0)
        
        # Check if beliefs cross threshold (can happen during starting delay)
        red_crosses_threshold = (current_red_belief >= press_thresh_batch) & valid_time_mask
        green_crosses_threshold = (current_green_belief >= press_thresh_batch) & valid_time_mask
        
        # Update evidence accumulation counters (can happen during starting delay)
        red_evidence_accum_counter = np.where(red_crosses_threshold, 
                                            red_evidence_accum_counter + 1, 
                                            0)
        green_evidence_accum_counter = np.where(green_crosses_threshold,
                                               green_evidence_accum_counter + 1,
                                               0)
        
        # Check if we should press (but only act if past starting delay)
        should_press_red = red_evidence_accum_counter >= tau_press_batch
        should_press_green = green_evidence_accum_counter >= tau_press_batch
        
        # Handle cases where no button is currently pressed
        no_button_mask = (button_pressed == 2)
        
        # Case 1: No button pressed - decide what to press (only if can act)
        both_should_press = should_press_red & should_press_green & no_button_mask & can_act_mask
        
        # Update win counters when both should press, reset when both should NOT press
        red_higher = current_red_belief > current_green_belief
        red_wins_counter = np.where(both_should_press & red_higher, red_wins_counter + 1,
                                  np.where(~both_should_press, 0, red_wins_counter))
        green_wins_counter = np.where(both_should_press & ~red_higher, green_wins_counter + 1,
                                    np.where(~both_should_press, 0, green_wins_counter))
        
        # Choose button based on accumulated wins, fall back to current belief if tied
        choose_red_by_wins = red_wins_counter > green_wins_counter
        choose_green_by_wins = green_wins_counter > red_wins_counter
        wins_tied = red_wins_counter == green_wins_counter
        
        new_button_no_press = np.where(both_should_press,
                                     np.where(choose_red_by_wins, 1,
                                            np.where(choose_green_by_wins, 0,
                                                   np.where(wins_tied & red_higher, 1, 0))),
                                     np.where(should_press_red & no_button_mask & can_act_mask, 1,
                                            np.where(should_press_green & no_button_mask & can_act_mask, 0, 2)))
        
        # Case 2: Button is pressed - check if we should hold or release (only if can act)
        button_pressed_mask = (button_pressed != 2)
        
        # Get current belief for the pressed button
        current_pressed_belief = np.where(button_pressed == 1, current_red_belief,
                                        np.where(button_pressed == 0, current_green_belief, 0.0))
        
        # Check if we should continue holding based on belief
        belief_above_thresh = current_pressed_belief >= press_thresh_batch
        hold_continue = belief_above_thresh & button_pressed_mask & can_act_mask
        
        # Update release accumulator - increment when button is held AND belief is below threshold
        belief_below_thresh = (current_pressed_belief < press_thresh_batch) & button_pressed_mask
        release_accum_counter = np.where(belief_below_thresh & can_act_mask,
                                       release_accum_counter + 1,
                                       0)
        
        # Only actually release after accumulating for tau_press timesteps
        should_release_basic = button_pressed_mask & ~hold_continue & can_act_mask
        should_release = should_release_basic & (release_accum_counter >= tau_press_batch)
        
        # When releasing, also consider accumulated wins if both should press
        new_button_release = np.where(should_release,
                                    np.where(both_should_press & should_release,
                                           np.where(choose_red_by_wins, 1,
                                                  np.where(choose_green_by_wins, 0,
                                                         np.where(wins_tied & red_higher, 1, 0))),
                                           np.where(should_press_red & should_release, 1,
                                                  np.where(should_press_green & should_release, 0, 2))),
                                    button_pressed)
        
        # Update button state
        button_pressed = np.where(no_button_mask, new_button_no_press, new_button_release)
        
        # Reset hysteresis counter for newly pressed buttons or released buttons
        new_press_mask = (no_button_mask & (new_button_no_press != 2)) | (should_release & (new_button_release != 2))
        release_mask = should_release & (new_button_release == 2)

        
        # Reset release accumulator when button changes or when belief is above threshold
        release_accum_counter = np.where(new_press_mask | release_mask | belief_above_thresh, 0, release_accum_counter)
        
        # Set decisions for this timestep (default to no press if can't act)
        decisions[:, t] = np.where(can_act_mask, button_pressed, 2)
    
    return decisions

def decision_model_v2(beliefs_batch, 
                                      press_thresh_batch, 
                                      tau_press_batch, 
                                      hysteresis_delay_batch,
                                      regular_delay_batch,
                                      starting_delay_batch):
    """
    Vectorized decision model for keypresses based on beliefs.
    
    Args:
        beliefs_batch: np.ndarray, shape (batch_size, T, 3) - beliefs over time for all runs
        press_thresh_batch: np.ndarray, shape (batch_size,) - threshold for pressing
        tau_press_batch: np.ndarray, shape (batch_size,) - time steps before pressing
        hysteresis_delay_batch: np.ndarray, shape (batch_size,) - hysteresis counter for holding
        regular_delay_batch: np.ndarray, shape (batch_size,) - regular delay for tau_delay
        starting_delay_batch: np.ndarray, shape (batch_size,) - initial delay before any action
    
    Returns:
        np.ndarray, shape (batch_size, T) - button presses (0=green, 1=red, 2=no press)
    """
    # Normalize beliefs_batch to shape (batch_size, T, 3). Single JTAP run or
    # single particle configurations can leave stray singleton dimensions.
    beliefs_batch = np.asarray(beliefs_batch)
    # Aggressively squeeze singleton dimensions and then normalize to (B,T,3)
    squeezed = np.squeeze(beliefs_batch)
    if squeezed.ndim == 2:
        # (T,3) -> (1,T,3)
        assert squeezed.shape[1] == 3
        beliefs_batch = squeezed[None, ...]
    elif squeezed.ndim == 3 and squeezed.shape[-1] == 3:
        beliefs_batch = squeezed
    else:
        # Handle specific common 4D cases before erroring
        if beliefs_batch.ndim == 4:
            if beliefs_batch.shape[-2] == 1 and beliefs_batch.shape[-1] == 3:
                beliefs_batch = beliefs_batch[:, :, 0, :]
            elif beliefs_batch.shape[-2] == 3 and beliefs_batch.shape[-1] == 1:
                beliefs_batch = beliefs_batch[:, :, :, 0]
            else:
                alt = np.squeeze(beliefs_batch)
                if alt.ndim == 3 and alt.shape[-1] == 3:
                    beliefs_batch = alt
                else:
                    raise ValueError(f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims")
        else:
            raise ValueError(f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims")

    batch_size, T, _ = beliefs_batch.shape
    
    # Initialize state arrays
    button_pressed = np.full(batch_size, 2, dtype=int)  # 2: No button, 1: Red, 0: Green
    hysteresis_counter = np.zeros(batch_size, dtype=int)
    red_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    green_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    release_accum_counter = np.zeros(batch_size, dtype=int)  # New accumulator for release decisions
    
    decisions = np.full((batch_size, T), 2, dtype=int)
    
    # Precompute time adjustments for all timesteps and batches
    t_indices = np.arange(T)[None, :] - regular_delay_batch[:, None]  # (batch_size, T)
    
    for t in range(T):
        # Get adjusted time indices for this timestep
        t_adjusted = t_indices[:, t]  # (batch_size,)
        
        # Create masks for valid time indices
        valid_time_mask = (t_adjusted >= 0) & (t_adjusted < T)
        past_starting_delay = t >= starting_delay_batch
        can_act_mask = valid_time_mask & past_starting_delay
        
        # Get current beliefs (handle invalid indices with 0s)
        # Ensure 1D integer indices with matching shape
        safe_t_adjusted = np.asarray(np.clip(t_adjusted, 0, T - 1), dtype=int).reshape(batch_size)
        batch_indices = np.arange(batch_size, dtype=int)
        
        current_red_belief = np.where(can_act_mask, 
                                    beliefs_batch[batch_indices, safe_t_adjusted, 1], 
                                    0.0)
        current_green_belief = np.where(can_act_mask,
                                      beliefs_batch[batch_indices, safe_t_adjusted, 0],
                                      0.0)
        
        # Check if beliefs cross threshold
        red_crosses_threshold = (current_red_belief >= press_thresh_batch) & can_act_mask
        green_crosses_threshold = (current_green_belief >= press_thresh_batch) & can_act_mask
        
        # Update evidence accumulation counters
        red_evidence_accum_counter = np.where(red_crosses_threshold, 
                                            red_evidence_accum_counter + 1, 
                                            0)
        green_evidence_accum_counter = np.where(green_crosses_threshold,
                                               green_evidence_accum_counter + 1,
                                               0)
        
        # Check if we should press
        should_press_red = red_evidence_accum_counter >= tau_press_batch
        should_press_green = green_evidence_accum_counter >= tau_press_batch
        
        # Handle cases where no button is currently pressed
        no_button_mask = (button_pressed == 2)
        
        # Case 1: No button pressed - decide what to press
        both_should_press = should_press_red & should_press_green & no_button_mask
        red_higher = current_red_belief > current_green_belief
        
        new_button_no_press = np.where(both_should_press,
                                     np.where(red_higher, 1, 0),
                                     np.where(should_press_red & no_button_mask, 1,
                                            np.where(should_press_green & no_button_mask, 0, 2)))
        
        # Case 2: Button is pressed - check if we should hold or release
        button_pressed_mask = (button_pressed != 2)
        
        # Get current belief for the pressed button
        current_pressed_belief = np.where(button_pressed == 1, current_red_belief,
                                        np.where(button_pressed == 0, current_green_belief, 0.0))
        
        # Update hysteresis counter for pressed buttons
        hysteresis_counter = np.where(button_pressed_mask, hysteresis_counter + 1, 0)
        
        # Check if we should continue holding based on belief and hysteresis
        belief_above_thresh = current_pressed_belief >= press_thresh_batch
        within_hysteresis = hysteresis_counter <= hysteresis_delay_batch
        hold_continue = (belief_above_thresh | within_hysteresis) & button_pressed_mask
        
        # Update release accumulator - increment when button is held AND belief is below threshold
        belief_below_thresh = (current_pressed_belief < press_thresh_batch) & button_pressed_mask
        release_accum_counter = np.where(belief_below_thresh,
                                       release_accum_counter + 1,
                                       0)
        
        # Only actually release after accumulating for tau_press timesteps AND passing hysteresis check
        should_release_basic = button_pressed_mask & ~hold_continue
        should_release = should_release_basic & (release_accum_counter >= tau_press_batch)
        
        new_button_release = np.where(should_release,
                                    np.where(both_should_press & should_release,
                                           np.where(red_higher, 1, 0),
                                           np.where(should_press_red & should_release, 1,
                                                  np.where(should_press_green & should_release, 0, 2))),
                                    button_pressed)
        
        # Update button state
        button_pressed = np.where(no_button_mask, new_button_no_press, new_button_release)
        
        # Reset hysteresis counter for newly pressed buttons or released buttons
        new_press_mask = (no_button_mask & (new_button_no_press != 2)) | (should_release & (new_button_release != 2))
        release_mask = should_release & (new_button_release == 2)
        hysteresis_counter = np.where(new_press_mask, 1, 
                                    np.where(release_mask, 0, hysteresis_counter))
        
        # Reset release accumulator when button changes or when belief is above threshold
        release_accum_counter = np.where(new_press_mask | release_mask | belief_above_thresh, 0, release_accum_counter)
        
        # Set decisions for this timestep (default to no press if can't act)
        decisions[:, t] = np.where(can_act_mask, button_pressed, 2)
    
    return decisions

def decision_model_v3(beliefs_batch, 
                                      press_thresh_batch, 
                                      tau_press_batch, 
                                      hysteresis_delay_batch,
                                      regular_delay_batch,
                                      starting_delay_batch):
    """
    Vectorized decision model for keypresses based on beliefs.
    
    Args:
        beliefs_batch: np.ndarray, shape (batch_size, T, 3) - beliefs over time for all runs
        press_thresh_batch: np.ndarray, shape (batch_size,) - threshold for pressing
        tau_press_batch: np.ndarray, shape (batch_size,) - time steps before pressing
        hysteresis_delay_batch: np.ndarray, shape (batch_size,) - hysteresis counter for holding
        regular_delay_batch: np.ndarray, shape (batch_size,) - regular delay for tau_delay
        starting_delay_batch: np.ndarray, shape (batch_size,) - initial delay before any action
    
    Returns:
        np.ndarray, shape (batch_size, T) - button presses (0=green, 1=red, 2=no press)
    """
    # Normalize beliefs_batch to shape (batch_size, T, 3). Single JTAP run or
    # single particle configurations can leave stray singleton dimensions.
    beliefs_batch = np.asarray(beliefs_batch)
    # Aggressively squeeze singleton dimensions and then normalize to (B,T,3)
    squeezed = np.squeeze(beliefs_batch)
    if squeezed.ndim == 2:
        # (T,3) -> (1,T,3)
        assert squeezed.shape[1] == 3
        beliefs_batch = squeezed[None, ...]
    elif squeezed.ndim == 3 and squeezed.shape[-1] == 3:
        beliefs_batch = squeezed
    else:
        # Handle specific common 4D cases before erroring
        if beliefs_batch.ndim == 4:
            if beliefs_batch.shape[-2] == 1 and beliefs_batch.shape[-1] == 3:
                beliefs_batch = beliefs_batch[:, :, 0, :]
            elif beliefs_batch.shape[-2] == 3 and beliefs_batch.shape[-1] == 1:
                beliefs_batch = beliefs_batch[:, :, :, 0]
            else:
                alt = np.squeeze(beliefs_batch)
                if alt.ndim == 3 and alt.shape[-1] == 3:
                    beliefs_batch = alt
                else:
                    raise ValueError(f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims")
        else:
            raise ValueError(f"Unexpected beliefs_batch shape {beliefs_batch.shape}; expected (B,T,3) up to singleton dims")

    batch_size, T, _ = beliefs_batch.shape
    
    # Initialize state arrays
    button_pressed = np.full(batch_size, 2, dtype=int)  # 2: No button, 1: Red, 0: Green
    hysteresis_counter = np.zeros(batch_size, dtype=int)
    red_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    green_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    release_accum_counter = np.zeros(batch_size, dtype=int)  # New accumulator for release decisions
    red_wins_counter = np.zeros(batch_size, dtype=int)  # Track red wins when both should press
    green_wins_counter = np.zeros(batch_size, dtype=int)  # Track green wins when both should press
    
    decisions = np.full((batch_size, T), 2, dtype=int)
    
    # Precompute time adjustments for all timesteps and batches
    t_indices = np.arange(T)[None, :] - regular_delay_batch[:, None]  # (batch_size, T)
    
    for t in range(T):
        # Get adjusted time indices for this timestep
        t_adjusted = t_indices[:, t]  # (batch_size,)
        
        # Create masks for valid time indices
        valid_time_mask = (t_adjusted >= 0) & (t_adjusted < T)
        past_starting_delay = t >= starting_delay_batch
        can_act_mask = valid_time_mask & past_starting_delay
        
        # Get current beliefs (handle invalid indices with 0s)
        # Ensure 1D integer indices with matching shape
        safe_t_adjusted = np.asarray(np.clip(t_adjusted, 0, T - 1), dtype=int).reshape(batch_size)
        batch_indices = np.arange(batch_size, dtype=int)
        
        # Extract beliefs for evidence accumulation (can happen during starting delay)
        current_red_belief = np.where(valid_time_mask, 
                                    beliefs_batch[batch_indices, safe_t_adjusted, 1], 
                                    0.0)
        current_green_belief = np.where(valid_time_mask,
                                      beliefs_batch[batch_indices, safe_t_adjusted, 0],
                                      0.0)
        
        # Check if beliefs cross threshold (can happen during starting delay)
        red_crosses_threshold = (current_red_belief >= press_thresh_batch) & valid_time_mask
        green_crosses_threshold = (current_green_belief >= press_thresh_batch) & valid_time_mask
        
        # Update evidence accumulation counters (can happen during starting delay)
        red_evidence_accum_counter = np.where(red_crosses_threshold, 
                                            red_evidence_accum_counter + 1, 
                                            0)
        green_evidence_accum_counter = np.where(green_crosses_threshold,
                                               green_evidence_accum_counter + 1,
                                               0)
        
        # Check if we should press (but only act if past starting delay)
        should_press_red = red_evidence_accum_counter >= tau_press_batch
        should_press_green = green_evidence_accum_counter >= tau_press_batch
        
        # Handle cases where no button is currently pressed
        no_button_mask = (button_pressed == 2)
        
        # Case 1: No button pressed - decide what to press (only if can act)
        both_should_press = should_press_red & should_press_green & no_button_mask & can_act_mask
        
        # Update win counters when both should press, reset when both should NOT press
        red_higher = current_red_belief > current_green_belief
        red_wins_counter = np.where(both_should_press & red_higher, red_wins_counter + 1,
                                  np.where(~both_should_press, 0, red_wins_counter))
        green_wins_counter = np.where(both_should_press & ~red_higher, green_wins_counter + 1,
                                    np.where(~both_should_press, 0, green_wins_counter))
        
        # Choose button based on accumulated wins, fall back to current belief if tied
        choose_red_by_wins = red_wins_counter > green_wins_counter
        choose_green_by_wins = green_wins_counter > red_wins_counter
        wins_tied = red_wins_counter == green_wins_counter
        
        new_button_no_press = np.where(both_should_press,
                                     np.where(choose_red_by_wins, 1,
                                            np.where(choose_green_by_wins, 0,
                                                   np.where(wins_tied & red_higher, 1, 0))),
                                     np.where(should_press_red & no_button_mask & can_act_mask, 1,
                                            np.where(should_press_green & no_button_mask & can_act_mask, 0, 2)))
        
        # Case 2: Button is pressed - check if we should hold or release (only if can act)
        button_pressed_mask = (button_pressed != 2)
        
        # Get current belief for the pressed button
        current_pressed_belief = np.where(button_pressed == 1, current_red_belief,
                                        np.where(button_pressed == 0, current_green_belief, 0.0))
        
        # Update hysteresis counter for pressed buttons
        hysteresis_counter = np.where(button_pressed_mask, hysteresis_counter + 1, 0)
        
        # Check if we should continue holding based on belief and hysteresis
        belief_above_thresh = current_pressed_belief >= press_thresh_batch
        within_hysteresis = hysteresis_counter <= hysteresis_delay_batch
        hold_continue = (belief_above_thresh | within_hysteresis) & button_pressed_mask & can_act_mask
        
        # Update release accumulator - increment when button is held AND belief is below threshold
        belief_below_thresh = (current_pressed_belief < press_thresh_batch) & button_pressed_mask
        release_accum_counter = np.where(belief_below_thresh & can_act_mask,
                                       release_accum_counter + 1,
                                       0)
        
        # Only actually release after accumulating for tau_press timesteps AND passing hysteresis check
        should_release_basic = button_pressed_mask & ~hold_continue & can_act_mask
        should_release = should_release_basic & (release_accum_counter >= tau_press_batch)
        
        # When releasing, also consider accumulated wins if both should press
        new_button_release = np.where(should_release,
                                    np.where(both_should_press & should_release,
                                           np.where(choose_red_by_wins, 1,
                                                  np.where(choose_green_by_wins, 0,
                                                         np.where(wins_tied & red_higher, 1, 0))),
                                           np.where(should_press_red & should_release, 1,
                                                  np.where(should_press_green & should_release, 0, 2))),
                                    button_pressed)
        
        # Update button state
        button_pressed = np.where(no_button_mask, new_button_no_press, new_button_release)
        
        # Reset hysteresis counter for newly pressed buttons or released buttons
        new_press_mask = (no_button_mask & (new_button_no_press != 2)) | (should_release & (new_button_release != 2))
        release_mask = should_release & (new_button_release == 2)
        hysteresis_counter = np.where(new_press_mask, 1, 
                                    np.where(release_mask, 0, hysteresis_counter))
        
        # Reset release accumulator when button changes or when belief is above threshold
        release_accum_counter = np.where(new_press_mask | release_mask | belief_above_thresh, 0, release_accum_counter)
        
        # Set decisions for this timestep (default to no press if can't act)
        decisions[:, t] = np.where(can_act_mask, button_pressed, 2)
    
    return decisions

def decision_model_v1(beliefs_batch, 
                                      press_thresh_batch, 
                                      tau_press_batch, 
                                      hysteresis_delay_batch,
                                      regular_delay_batch,
                                      starting_delay_batch):
    """
    Vectorized decision model for keypresses based on beliefs.
    
    Args:
        beliefs_batch: np.ndarray, shape (batch_size, T, 3) - beliefs over time for all runs
        press_thresh_batch: np.ndarray, shape (batch_size,) - threshold for pressing
        tau_press_batch: np.ndarray, shape (batch_size,) - time steps before pressing
        hysteresis_delay_batch: np.ndarray, shape (batch_size,) - hysteresis counter for holding
        regular_delay_batch: np.ndarray, shape (batch_size,) - regular delay for tau_delay
        starting_delay_batch: np.ndarray, shape (batch_size,) - initial delay before any action
    
    Returns:
        np.ndarray, shape (batch_size, T) - button presses (0=green, 1=red, 2=no press)
    """
    batch_size, T, _ = beliefs_batch.shape
    
    # Initialize state arrays
    button_pressed = np.full(batch_size, 2, dtype=int)  # 2: No button, 1: Red, 0: Green
    hysteresis_counter = np.zeros(batch_size, dtype=int)
    red_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    green_evidence_accum_counter = np.zeros(batch_size, dtype=int)
    
    decisions = np.full((batch_size, T), 2, dtype=int)
    
    # Precompute time adjustments for all timesteps and batches
    t_indices = np.arange(T)[None, :] - regular_delay_batch[:, None]  # (batch_size, T)
    
    for t in range(T):
        # Get adjusted time indices for this timestep
        t_adjusted = t_indices[:, t]  # (batch_size,)
        
        # Create masks for valid time indices
        valid_time_mask = (t_adjusted >= 0) & (t_adjusted < T)
        past_starting_delay = t >= starting_delay_batch
        can_act_mask = valid_time_mask & past_starting_delay
        
        # Get current beliefs (handle invalid indices with 0s)
        safe_t_adjusted = np.clip(t_adjusted, 0, T-1)
        batch_indices = np.arange(batch_size)
        
        current_red_belief = np.where(can_act_mask, 
                                    beliefs_batch[batch_indices, safe_t_adjusted, 1], 
                                    0.0)
        current_green_belief = np.where(can_act_mask,
                                      beliefs_batch[batch_indices, safe_t_adjusted, 0],
                                      0.0)
        
        # Check if beliefs cross threshold
        red_crosses_threshold = (current_red_belief >= press_thresh_batch) & can_act_mask
        green_crosses_threshold = (current_green_belief >= press_thresh_batch) & can_act_mask
        
        # Update evidence accumulation counters
        red_evidence_accum_counter = np.where(red_crosses_threshold, 
                                            red_evidence_accum_counter + 1, 
                                            0)
        green_evidence_accum_counter = np.where(green_crosses_threshold,
                                               green_evidence_accum_counter + 1,
                                               0)
        
        # Check if we should press
        should_press_red = red_evidence_accum_counter >= tau_press_batch
        should_press_green = green_evidence_accum_counter >= tau_press_batch
        
        # Handle cases where no button is currently pressed
        no_button_mask = (button_pressed == 2)
        
        # Case 1: No button pressed - decide what to press
        both_should_press = should_press_red & should_press_green & no_button_mask
        red_higher = current_red_belief > current_green_belief
        
        new_button_no_press = np.where(both_should_press,
                                     np.where(red_higher, 1, 0),
                                     np.where(should_press_red & no_button_mask, 1,
                                            np.where(should_press_green & no_button_mask, 0, 2)))
        
        # Case 2: Button is pressed - check if we should hold or release
        button_pressed_mask = (button_pressed != 2)
        
        # Get current belief for the pressed button
        current_pressed_belief = np.where(button_pressed == 1, current_red_belief,
                                        np.where(button_pressed == 0, current_green_belief, 0.0))
        
        # Update hysteresis counter for pressed buttons
        hysteresis_counter = np.where(button_pressed_mask, hysteresis_counter + 1, 0)
        
        # Check if we should continue holding
        belief_above_thresh = current_pressed_belief >= press_thresh_batch
        within_hysteresis = hysteresis_counter <= hysteresis_delay_batch
        hold_continue = (belief_above_thresh | within_hysteresis) & button_pressed_mask
        
        # For buttons that should be released, check for new presses
        should_release = button_pressed_mask & ~hold_continue
        
        new_button_release = np.where(should_release,
                                    np.where(both_should_press & should_release,
                                           np.where(red_higher, 1, 0),
                                           np.where(should_press_red & should_release, 1,
                                                  np.where(should_press_green & should_release, 0, 2))),
                                    button_pressed)
        
        # Update button state
        button_pressed = np.where(no_button_mask, new_button_no_press, new_button_release)
        
        # Reset hysteresis counter for newly pressed buttons or released buttons
        new_press_mask = (no_button_mask & (new_button_no_press != 2)) | (should_release & (new_button_release != 2))
        release_mask = should_release & (new_button_release == 2)
        hysteresis_counter = np.where(new_press_mask, 1, 
                                    np.where(release_mask, 0, hysteresis_counter))
        
        # Set decisions for this timestep (default to no press if can't act)
        decisions[:, t] = np.where(can_act_mask, button_pressed, 2)
    
    return decisions