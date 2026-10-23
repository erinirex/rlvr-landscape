# Copyright (c) ModelScope Contributors. All rights reserved.
# Outcome Reward Model (ORM) implementations for GRPO training.

import json
from swift.grading.grader import grade_answer
import os
import re
import sys
from typing import Any, List
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import sympy as sp

from swift.infer_engine import InferRequest

if TYPE_CHECKING:
    from swift.megatron.arguments import MegatronArguments
    from swift.rlhf_trainers import GRPOConfig


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError



def extract_latex_command_argument(
    text: str,
    command: str,
) -> list[str]:
    """Extract arguments from commands such as \\boxed{...}.

    Supports nested braces, unlike a regular expression.
    """
    results: list[str] = []
    pattern = re.compile(
        rf"\\{re.escape(command)}\s*\{{"
    )

    for match in pattern.finditer(text):
        start = match.end()
        depth = 1
        index = start

        while index < len(text) and depth > 0:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1

            index += 1

        if depth == 0:
            results.append(
                text[start : index - 1].strip()
            )

    return results

def extract_math500_answer(text: str) -> str:
    """Extract a mathematical answer from a model completion."""
    text = str(text).strip()

    boxed_answers = extract_latex_command_argument(
        text,
        "boxed",
    )

    if boxed_answers:
        return boxed_answers[-1].strip()

    final_answer_matches = list(
        re.finditer(
            r"(?:Final\s+Answer|Answer)\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if final_answer_matches:
        answer_start = final_answer_matches[-1].end()
        answer_text = text[answer_start:].strip()

        # Prefer the first non-empty line after "Final Answer".
        for line in answer_text.splitlines():
            cleaned_line = line.strip()

            if cleaned_line:
                return cleaned_line

    return ""



class Math500Accuracy(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

        self.debug_count = 0
        self.debug_limit = int(
            os.environ.get("MATH500_DEBUG_LIMIT", "5")
        )

    @staticmethod
    def _completion_to_text(completion: Any) -> str:
        """Normalize completion formats into plain text."""
        if isinstance(completion, str):
            return completion

        # Example:
        # [{"role": "assistant", "content": "..."}]
        if isinstance(completion, list):
            text_parts: list[str] = []

            for item in completion:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue

                if not isinstance(item, dict):
                    continue

                content = item.get("content", "")

                if isinstance(content, str):
                    text_parts.append(content)

                elif isinstance(content, list):
                    for content_item in content:
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "text"
                        ):
                            text_parts.append(
                                str(content_item.get("text", ""))
                            )

            return "\n".join(text_parts).strip()

        # Example:
        # {"role": "assistant", "content": "..."}
        if isinstance(completion, dict):
            content = completion.get("content", "")

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return "\n".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                ).strip()

        return str(completion)

    @staticmethod
    def _extract_latex_command_argument(
        text: str,
        command: str,
    ) -> list[str]:
        """Extract nested LaTeX command arguments such as \\boxed{...}."""
        results: list[str] = []

        pattern = re.compile(
            rf"\\{re.escape(command)}\s*\{{"
        )

        for match in pattern.finditer(str(text)):
            start = match.end()
            depth = 1
            index = start

            while index < len(text) and depth > 0:
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1

                index += 1

            if depth == 0:
                results.append(
                    text[start:index - 1].strip()
                )

        return results

    @staticmethod
    def _extract_answer_tag(text: str) -> tuple[str, bool]:
        """Extract the last <answer>...</answer> block."""
        matches = re.findall(
            r"<answer>(.*?)</answer>",
            str(text),
            flags=re.IGNORECASE | re.DOTALL,
        )

        if matches:
            return matches[-1].strip(), True

        return str(text).strip(), False

    @classmethod
    def _extract_prediction(
        cls,
        text: str,
    ) -> tuple[str, str]:
        """Extract the model's final answer."""
        text = str(text).strip()

        tagged_answer, has_answer_tag = (
            cls._extract_answer_tag(text)
        )

        if has_answer_tag:
            return tagged_answer, "answer_tag"

        boxed_answers = cls._extract_latex_command_argument(
            text,
            "boxed",
        )

        if boxed_answers:
            return boxed_answers[-1].strip(), "boxed"

        final_answer_matches = list(
            re.finditer(
                r"(?:Final\s+Answer|Answer)\s*:?\s*",
                text,
                flags=re.IGNORECASE,
            )
        )

        if final_answer_matches:
            answer_start = final_answer_matches[-1].end()
            answer_text = text[answer_start:].strip()

            for line in answer_text.splitlines():
                cleaned_line = line.strip()

                if cleaned_line:
                    return cleaned_line, "final_answer"

        # Avoid sending the full chain of thought to the grader.
        return "", "no_answer_marker"

    @classmethod
    def _extract_gold(cls, gold: Any) -> str:
        """Extract the Math500 reference answer."""
        if gold is None:
            return ""

        if isinstance(gold, dict):
            if "answer" in gold:
                gold = gold["answer"]
            elif "solution" in gold:
                gold = gold["solution"]

        text = str(gold).strip()

        tagged_answer, has_answer_tag = (
            cls._extract_answer_tag(text)
        )

        if has_answer_tag:
            return tagged_answer

        # If a full worked solution is passed, extract its final boxed answer.
        boxed_answers = cls._extract_latex_command_argument(
            text,
            "boxed",
        )

        if boxed_answers:
            return boxed_answers[-1].strip()

        # Math500's answer field is normally already a plain answer.
        return text

    def __call__(
        self,
        completions,
        solution=None,
        answer=None,
        **kwargs,
    ) -> List[float]:
        """Return one PRM800K/MATH-style correctness reward per completion."""
        gold_answers = (
            answer if answer is not None else solution
        )

        if gold_answers is None:
            raise ValueError(
                "Math500Accuracy requires either "
                "`solution` or `answer`."
            )

        if isinstance(completions, str):
            completions = [completions]

        if isinstance(gold_answers, (str, dict)):
            gold_answers = [gold_answers]

        # Useful when one prompt has multiple sampled generations.
        if len(gold_answers) == 1 and len(completions) > 1:
            gold_answers = gold_answers * len(completions)

        if len(completions) != len(gold_answers):
            raise ValueError(
                "completions and gold answers must have the "
                f"same length, got {len(completions)} and "
                f"{len(gold_answers)}"
            )

        rewards: List[float] = []

        for completion, gold_answer in zip(
            completions,
            gold_answers,
        ):
            completion_text = self._completion_to_text(
                completion
            )

            prediction_text, extraction_method = (
                self._extract_prediction(completion_text)
            )

            target_text = self._extract_gold(
                gold_answer
            )

            try:
                if not prediction_text or not target_text:
                    is_correct = False
                else:
                    is_correct = grade_answer(
                        given_answer=prediction_text,
                        ground_truth=target_text,
                    )

                reward = float(is_correct)

            except Exception as error:
                print(
                    "[Math500Accuracy] Failed to score answer: "
                    f"prediction={prediction_text!r}, "
                    f"target={target_text!r}, "
                    f"error={type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

                reward = 0.0

            if self.debug_count < self.debug_limit:
                self.debug_count += 1

                print(
                    "\n"
                    + "=" * 100
                    + "\n"
                    + f"[Math500Accuracy Debug "
                    f"#{self.debug_count}]\n"
                    + f"RAW GENERATION:\n"
                    f"{completion_text}\n\n"
                    + f"RAW GOLD:\n"
                    f"{gold_answer!r}\n\n"
                    + f"EXTRACTION METHOD:\n"
                    f"{extraction_method}\n\n"
                    + f"EXTRACTED PREDICTION:\n"
                    f"{prediction_text!r}\n\n"
                    + f"EXTRACTED TARGET:\n"
                    f"{target_text!r}\n\n"
                    + f"REWARD: {reward}\n"
                    + "=" * 100,
                    file=sys.stderr,
                    flush=True,
                )

            rewards.append(reward)

        return rewards
    

class MathAccuracy(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")
        # raise RuntimeError("MY MathAccuracy loaded")


    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')


            # print("=" * 80, file=sys.stderr, flush=True)
            # print("RAW completion:", repr(content), file=sys.stderr, flush=True)
            # print("RAW solution:", repr(sol), file=sys.stderr, flush=True)
            # print("content_to_parse:", repr(content_to_parse), file=sys.stderr, flush=True)
            # print("sol_to_parse:", repr(sol_to_parse), file=sys.stderr, flush=True)
            # print("gold_parsed:", gold_parsed, file=sys.stderr, flush=True)
            # print("=" * 80, file=sys.stderr, flush=True)

            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                # reward = 0.0


                # pred = content_to_parse.strip().replace(" ", "").lower()
                # gold = sol_to_parse.strip().replace(" ", "").lower()

                # reward = float(pred == gold)


                def extract_uci(x):
                    x = str(x).strip().lower()

                    m = re.search(r'<answer>(.*?)</answer>', x, re.DOTALL)
                    if m:
                        x = m.group(1).strip().lower()

                    moves = re.findall(r'\b[a-h][1-8][a-h][1-8][qrbn]?\b', x)
                    if moves:
                        return moves[-1]

                    return x.replace(" ", "").strip()

                pred = extract_uci(content_to_parse)
                gold = extract_uci(sol_to_parse)

                reward = float(pred == gold)

                # if len(rewards) < 8:
                    

            rewards.append(reward)
        return rewards


class ChessAccuracy(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

        import sys
        sys.path.append('/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/miles')
        sys.path.append('/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/chess-rl-miles')

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        import re
        import sys

        sys.path.append('/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/miles')
        sys.path.append('/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/chess-rl-miles')

        from chess_rl_miles.moves import (
            extract_move_after_thinking,
            extract_first_move,
            safe_move_to_uci,
            parse_ground_truth,
        )

        def get_item(obj, idx, default=None):
            if isinstance(obj, list) and idx < len(obj):
                return obj[idx]
            return default

        def get_fen(idx):
            metadata = get_item(kwargs.get("metadata"), idx, {}) or {}
            extra_info = get_item(kwargs.get("extra_info"), idx, {}) or {}

            if isinstance(metadata, dict):
                fen = metadata.get("FEN") or metadata.get("fen")
                if fen:
                    return fen

            if isinstance(extra_info, dict):
                fen = (
                    extra_info.get("original_FEN")
                    or extra_info.get("original_fen")
                    or extra_info.get("FEN")
                    or extra_info.get("fen")
                )
                if fen:
                    return fen

            return ""

        def extract_raw_move(text):
            move, follows_format = extract_move_after_thinking(
                text,
                strict_single_close=True,
            )

            if move is None and not follows_format:
                move = extract_first_move(text)

            # 如果原 parser 没找到，再手动取 </T> 后第一个 token
            if move is None and "</T>" in text:
                after = text.split("</T>", 1)[1].strip()
                if after:
                    move = after.split()[0].strip()

            # 清理 markdown / 标点
            if move:
                move = move.strip()
                move = move.strip("`*_")
                move = move.rstrip(".,;:")

            return move, follows_format

        def move_to_uci(move_text, fen=""):
            if not move_text:
                return ""

            move_text = str(move_text).strip()
            move_text = move_text.strip("`*_")
            move_text = move_text.rstrip(".,;:")

            # 1. UCI: b5c7, e7e8q
            m = re.fullmatch(
                r"[a-h][1-8][a-h][1-8][qrbn]?",
                move_text.lower(),
            )

            if m:
                uci = m.group(0).lower()

                if fen:
                    try:
                        import chess

                        board = chess.Board(fen)
                        move = chess.Move.from_uci(uci)

                        if move in board.legal_moves:
                            return move.uci()

                        print(
                            "Illegal UCI move:",
                            repr(uci),
                            repr(fen),
                            file=sys.stderr,
                            flush=True,
                        )
                        return ""

                    except Exception as e:
                        print(
                            "UCI validation failed:",
                            repr(uci),
                            repr(fen),
                            repr(e),
                            file=sys.stderr,
                            flush=True,
                        )
                        return ""

                return uci

            # 2. LAN parser: Nb5c7, Rh4xg4, Pe7e8=Q
            try:
                uci = safe_move_to_uci(move_text)

                if uci:
                    uci = str(uci).strip().lower()

                    if fen:
                        try:
                            import chess

                            board = chess.Board(fen)
                            move = chess.Move.from_uci(uci)

                            if move in board.legal_moves:
                                return move.uci()

                            return ""
                        except Exception:
                            return ""

                    return uci

            except Exception:
                pass

            # 3. SAN parser: Nxc7+, Qh4, hxg5
            if fen:
                try:
                    import chess

                    board = chess.Board(fen)
                    return board.parse_san(move_text).uci()

                except Exception as e:
                    print(
                        "SAN parse failed:",
                        repr(move_text),
                        repr(fen),
                        repr(e),
                        file=sys.stderr,
                        flush=True,
                    )

            return ""

        rewards = []

        metadata_list = kwargs.get("metadata")
        extra_info_list = kwargs.get("extra_info")

        for i, (content, sol) in enumerate(zip(completions, solution)):
            metadata = get_item(metadata_list, i, {}) or {}
            extra_info = get_item(extra_info_list, i, {}) or {}
            fen = get_fen(i)

            target_moves = parse_ground_truth(sol, metadata)
            gold = str(target_moves[0]).strip().lower() if target_moves else ""

            extracted_move, follows_format = extract_raw_move(content)
            pred_uci = move_to_uci(extracted_move, fen)

            pred_uci = pred_uci.strip().lower()

            reward = float(
                bool(pred_uci)
                and pred_uci == gold
            )

            # print("=" * 80, file=sys.stderr, flush=True)
            # print("RAW completion:", repr(content), file=sys.stderr, flush=True)
            # print("RAW solution:", repr(sol), file=sys.stderr, flush=True)
            # print("kwargs keys:", list(kwargs.keys()), file=sys.stderr, flush=True)
            # print("metadata:", repr(metadata), file=sys.stderr, flush=True)
            # print("extra_info:", repr(extra_info), file=sys.stderr, flush=True)
            # print("fen:", repr(fen), file=sys.stderr, flush=True)
            # print("target_moves:", repr(target_moves), file=sys.stderr, flush=True)
            # print("gold:", repr(gold), file=sys.stderr, flush=True)
            # print("follows_format:", follows_format, file=sys.stderr, flush=True)
            # print("extracted_move:", repr(extracted_move), file=sys.stderr, flush=True)
            # print("pred_uci:", repr(pred_uci), file=sys.stderr, flush=True)
            # print("reward:", reward, file=sys.stderr, flush=True)
            # print("=" * 80, file=sys.stderr, flush=True)

            # with open("/mnt/sj/home/yichen/chess_reward_debug.log", "a") as f:
            #     f.write("=" * 80 + "\n")
            #     f.write(f"RAW completion: {repr(content)}\n")
            #     f.write(f"RAW solution: {repr(sol)}\n")
            #     f.write(f"kwargs keys: {list(kwargs.keys())}\n")
            #     f.write(f"metadata: {repr(metadata)}\n")
            #     f.write(f"extra_info: {repr(extra_info)}\n")
            #     f.write(f"fen: {repr(fen)}\n")
            #     f.write(f"target_moves: {repr(target_moves)}\n")
            #     f.write(f"gold: {repr(gold)}\n")
            #     f.write(f"follows_format: {follows_format}\n")
            #     f.write(f"extracted_move: {repr(extracted_move)}\n")
            #     f.write(f"pred_uci: {repr(pred_uci)}\n")
            #     f.write(f"reward: {reward}\n")

            rewards.append(reward)

        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, accuracy_orm=None):
        super().__init__(args)
        self.min_len_value_wrong = args.cosine_min_len_value_wrong
        self.max_len_value_wrong = args.cosine_max_len_value_wrong
        self.min_len_value_correct = args.cosine_min_len_value_correct
        self.max_len_value_correct = args.cosine_max_len_value_correct
        self.max_len = args.cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        self.ngram_size = args.repetition_n_grams
        self.max_penalty = args.repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        assert args.soft_cache_length < args.soft_max_length
        self.soft_max_length = args.soft_max_length
        self.soft_cache_length = args.soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'math500_acc': Math500Accuracy,
    'chess_accuracy': ChessAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
