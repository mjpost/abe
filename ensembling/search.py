import os, sys
import logging
import pathlib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")


if (__package__ is None or __package__ == "")  and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'ensembling'

import heapq
import torch

from ensembling.utils import compatibility

class TokenExtension():
    def __init__(self, score, idx, token, hyp_len):
        self.score = score
        self.idx = idx
        self.token = token
        self.hyp_len = hyp_len

class BeamState():
    def __init__(self, outputs=None, beam_index=0, weights=None):
        self.outputs = outputs # list of tuples (id, TokenExtension). one for each model
        self.beam_index = beam_index
        if weights is None:
            self.weights = [(1.0 / len(outputs)) for _ in range(len(outputs))]
        else:
            self.weights = weights
        self.weighted_score = sum([self.weights[i] * (output[1].score/output[1].hyp_len) for i, output in enumerate(self.outputs)])
        self.unnormalized = sum([self.weights[i] * output[1].score for i, output in enumerate(self.outputs)])

    def score(self):
        return self.weighted_score
    
    def raw_score(self):
        return self.unnormalized

    def __str__(self):
        out_str = f"STATE({self.beam_index})\t"
        out_str += " ".join([str(output[1].idx.item()) for output in self.outputs])
        return out_str 
    
    def __hash__(self):
        return hash(self.__str__())

    def __lt__(self, other):
        # Things are inverted for heapq reasons
        return self.weighted_score > other.weighted_score
    
    def __gt__(self, other):
        # Things are inverted for heapq reasons
        return self.score() < other.score()
    
    # def __eq__(self, other):
    #     return self.score() == other.score()

class Hypothesis():
    def __init__(self, output_ids, scores, weights, token_scores):
        self.output_ids = output_ids
        self.scores = scores
        self.token_scores = token_scores
        self.weighted_score = sum([weights[i] * score for i, score in enumerate(scores)])
        self.unnormalized = sum([weights[i] * score for i, score in enumerate(scores)])

    def score(self):
        return self.weighted_score

    def raw_score(self):
        return self.unnormalized
    
    def __lt__(self, other):
        return self.score() < other.score()

    def __gt__(self, other):
        return self.score() > other.score()
    
    def __eq__(self, other):
        return self.score() == other.score()


def expand_frontier(models, state, paired_outputs):
    beam_i = state.beam_index
    neighbors = []
    for model_i, model in enumerate(models):
        add = True
        outputs = []
        for output_i, state_output in enumerate(state.outputs):
            if model_i == output_i:
                next_id = state_output[0] + 1
                if next_id < len(paired_outputs[beam_i][model_i][0]):
                    outputs.append(
                        (next_id, TokenExtension(score=paired_outputs[beam_i][model_i][0][next_id],
                                                idx=paired_outputs[beam_i][model_i][1][next_id],
                                                token=model.id_to_token(paired_outputs[beam_i][model_i][1][next_id]),
                                                hyp_len=len(model.generated_tokens[beam_i]) + 1))
                    )
                else:
                    add = False
            else:
                outputs.append(state_output)
        if add:
            neighbors.append(
                BeamState(outputs=outputs, beam_index=beam_i, weights=state.weights)
            )
    return neighbors

def initialize_heap(
    batch_offset,
    paired_outputs,
    models,
    weights,
    num_beams,
    stalled_states
):
    # A heap that will store all beam candidates, used to form the next beam.
    # This will include single-model advances (where one model is behind and is trying to catch up)
    # and paired-model advances (where both models advanced a compatible token).
    # candidates = [[] for _ in range(num_beams)]
    candidates = []
    visited = set() # keeping track of which states get pushed so we don't push the same one many times

    # Initialize the heap with the first token from each model
    for beam_i in range(num_beams):

        # if all the models are stalled, this was a "padded beam" which was only added to fill the beam_size
        if all(stalled_states[beam_i]):
            continue
        next_outputs = [
            (
                0, # 0 signifies that this is the 0th index of the beam item -- we start with the highest score/first index
                TokenExtension( # TokenExtension object which contains all the information about this token as an extension
                    score = paired_outputs[batch_offset + beam_i][model_i][0][0],
                    idx = paired_outputs[batch_offset + beam_i][model_i][1][0],
                    token = model.id_to_token(paired_outputs[batch_offset + beam_i][model_i][1][0]),
                    hyp_len = len(model.generated_tokens[batch_offset + beam_i]) + 1
                )
            )
        for model_i, model in enumerate(models)]
        
        next_state = BeamState(
            outputs = next_outputs,
            beam_index = batch_offset + beam_i,
            weights = weights
        )
        heapq.heappush(candidates, next_state)
        visited.add(hash(next_state))

    return candidates, visited

def beam_search(
        batch_offset,
        num_beams,
        paired_outputs,
        models,
        weights,
        completed_beams,
        max_length,
        max_score,
        stalled_states,
        min_beams = 1,
        max_depth = 10000,
        ):

    num_models = len(models)

    candidates, visited = initialize_heap(
        batch_offset,
        paired_outputs,
        models,
        weights,
        num_beams,
        stalled_states
    )

    next_beam = []
    beam_completed = []
    while (len(next_beam) < num_beams) and ((len(beam_completed) + completed_beams) < num_beams):
        if len(visited) > max_depth and len(next_beam) > min_beams:
            logger.debug(f"Search is stopping early because the depth is too high: {len(visited)}")
            return next_beam, beam_completed
        next_state = heapq.heappop(candidates)

        if next_state.raw_score() < max_score:
            logger.debug(f"Search is stopping early because the score is too low: {next_state.raw_score()} compared to best hypothesis {max_score}")
            return next_beam, beam_completed
        
        if len(beam_completed) > 0 and next_state.raw_score() < beam_completed[0].raw_score():
            logger.debug(f"Search is stopping early because the score is too low: {next_state.raw_score()} compared to best hypothesis {beam_completed[0].raw_score()}")
            return next_beam, beam_completed

        # add neighbors regardless of compatibility
        for neighbor in expand_frontier(models, next_state, paired_outputs):
            if hash(neighbor) not in visited:
                visited.add(hash(neighbor))
                heapq.heappush(candidates, neighbor)

        
        compat_code, next_stall_states = compatibility(models, next_state)

        # we also want to add the "models are at max length bit here"
        if compat_code == 0 or (compat_code == 1 and (max([output[1].hyp_len for output in next_state.outputs]) >= max_length)):
            # all models have terminated with eos
            beam_completed.append(Hypothesis(
                output_ids = [
                    model.decoder_tokens[next_state.beam_index] + [next_state.outputs[model_i][1].idx.item()]
                    if next_state.outputs[model_i][1].idx != model.target_tokenizer.pad_token_id else \
                        model.decoder_tokens[next_state.beam_index]
                    for model_i, model in enumerate(models)
                ],
                scores = [next_state.outputs[model_i][1].score for model_i, model in enumerate(models)],
                weights = weights,
                token_scores = [
                    [_.item() for _ in model.beam_token_scores[next_state.beam_index]] + \
                        [(next_state.outputs[model_i][1].score - models[model_i].beam_scores[next_state.beam_index]).item()]
                    if next_state.outputs[model_i][1].idx != model.target_tokenizer.pad_token_id else \
                        [_.item() for _ in model.beam_token_scores[next_state.beam_index]]
                    for model_i, model in enumerate(models)
                ]
            ))

        elif compat_code == 1:
            next_beam.append((next_state, next_stall_states))
            logger.debug(
                f"SELECTED {len(next_beam)-1} {' ||| '.join([next_beam[-1][0].outputs[_][1].token for _ in range(num_models)])}"
            )

    logger.debug(f"VISITED_STATES: {len(visited)}")

    return next_beam, beam_completed

def get_pad_beams(next_batch_beam, models, batch_i, num_beams, weights):
    beams = next_batch_beam
    stall_state = [True for _ in models]
    batch_offset = batch_i * num_beams
    for beam_i in range(len(next_batch_beam), num_beams):
        outputs = []
        for model in models:
            outputs.append(
                (
                    0,
                    TokenExtension(
                        score = model.beam_scores[batch_offset + beam_i],
                        idx = torch.tensor([-1], dtype=torch.long, device=model.device),
                        token = "<pad>",
                        hyp_len = len(model.generated_tokens[batch_offset + beam_i]) + 1
                    )
                )
            )
        beams.append((BeamState(outputs=outputs, beam_index=batch_offset + beam_i, weights=weights), stall_state))
    return beams

