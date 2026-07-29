"""
Auto-generated self-contained HF tokenizer.
Do NOT edit manually -- regenerate via training.hf_tokenizer_utils.save_hf_tokenizer().
"""
from __future__ import annotations

# --- BaseTokenizer (inlined) ---
# base_tokenizer.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseTokenizer(ABC):
    """Minimal interface for tokenizers used in pretraining."""

    # ---- required ----
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text/PGN to token IDs."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text/PGN."""
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Return token -> id mapping (if available)."""
        raise NotImplementedError

    def bos_id(self) -> Optional[int]: return None
    def eos_id(self) -> Optional[int]: return None
    def pad_id(self) -> Optional[int]: return None
    def get_vocab_size(self) -> int: return len(self.get_vocab())

    def __call__(self, text: str) -> List[int]:
        """Alias for encode()."""
        return self.encode(text)

# --- Concrete tokenizer (inlined) ---
# lan_tokenizer_sft.py
"""
LAN Tokenizer with SFT support (CoT format with <T> and <sep> tokens).

This extends the base LAN tokenizer with SFT-specific functionality:
- <T> token for marking thinking/CoT content
- <sep> token for separating prompt from response
"""
from typing import List, Dict, Optional, Tuple
import io
import chess, chess.pgn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
_RESULT = {"1-0", "0-1", "1/2-1/2", "*"}
FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f+r for f in FILES for r in RANKS]
PROMOS = "QRBN"
DIGITS = set("0123456789")

# SFT special tokens for CoT format
T_TOKEN = "<T>"
T_END_TOKEN = "</T>"
SEP_TOKEN = "<sep>"

# Environment interaction / reward special tokens
CALL_ENV_TOKEN = "<call_env>"
VERIFY_TOKEN = "<verify>"
REWARD_POS_TOKEN = "<+1>"
REWARD_NEG_TOKEN = "<-1>"
REWARD_ZERO_TOKEN = "<0>"
ENV_TOKENS = [CALL_ENV_TOKEN]
REWARD_TOKENS = [VERIFY_TOKEN, REWARD_POS_TOKEN, REWARD_NEG_TOKEN, REWARD_ZERO_TOKEN]

def _vocab_with_sft(
    include_move_numbers: bool,
    keep_result: bool,
    bos: str,
    eos: str,
    unk: str,
    include_env_tokens: bool = False,
    include_reward_tokens: bool = False,
) -> Dict[str, int]:
    """Create vocabulary including SFT special tokens."""
    base = [bos, eos, unk]
    ops = ["x", "=", "+", "#", "O-O", "O-O-O", ".", "..."]
    toks = base + list("KQRBNP") + SQUARES + list(PROMOS) + ops
    if include_move_numbers:
        toks += list("0123456789")
    if keep_result:
        toks += sorted(_RESULT)

    # Add SFT special tokens for CoT format
    sft_tokens = [T_TOKEN, T_END_TOKEN, SEP_TOKEN]
    toks += sft_tokens

    # Add environment / reward tokens when requested
    if include_env_tokens:
        toks += ENV_TOKENS
    if include_reward_tokens:
        toks += REWARD_TOKENS

    return {t: i for i, t in enumerate(dict.fromkeys(toks))}


class LanTokenizerSFT(BaseTokenizer):
    """
    LAN Tokenizer with SFT capabilities.

    This tokenizer extends the base LAN tokenizer with:
    - <T> token for marking thinking/CoT boundaries
    - <sep> token for separating candidate trajectories

    CoT Format: {prompt} <T> <sep> {traj1} <sep> {traj2} <sep> ... <sep> {trajN} <sep> <T> {answer}

    Where:
    - {prompt}: The game history/board state (PGN moves)
    - {trajN}: Candidate reasoning trajectories
    - {answer}: The final best move
    """

    # Special tokens for CoT format
    T = T_TOKEN
    T_END = T_END_TOKEN
    SEP = SEP_TOKEN

    # Environment / reward tokens (class-level constants for easy access)
    CALL_ENV = CALL_ENV_TOKEN   # "<call_env>"
    VERIFY = VERIFY_TOKEN       # "<verify>"
    REWARD_POS = REWARD_POS_TOKEN  # "<+1>"
    REWARD_NEG = REWARD_NEG_TOKEN  # "<-1>"
    REWARD_ZERO = REWARD_ZERO_TOKEN  # "<0>"
    ENV_TOKENS = ENV_TOKENS     # full list

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Configuration dict with tokenizer settings.
                include_env_tokens (bool): add <call_env>, <verify>, <+1>, <-1>, <0>
                    to the vocabulary.  Default: False.
        """
        config = config or {}

        include_move_numbers = config.get("include_move_numbers", False)
        include_black_tripledots = config.get("include_black_tripledots", False)
        bos = config.get("bos", "<bos>")
        eos = config.get("eos", "<eos>")
        unk = config.get("unk", "<unk>")
        keep_result = config.get("keep_result", False)
        include_env_tokens = config.get("include_env_tokens", False)
        include_reward_tokens = config.get("include_reward_tokens", False)

        self._bos = bos
        self._eos = eos
        self._unk = unk
        self._keep_res = keep_result
        self._include_nums = include_move_numbers
        self._include_black_ellipses = include_black_tripledots
        self._include_env_tokens = include_env_tokens
        self._include_reward_tokens = include_reward_tokens

        # Create vocabulary with SFT tokens
        tok2id = _vocab_with_sft(
            include_move_numbers, keep_result, bos, eos, unk,
            include_env_tokens=include_env_tokens,
            include_reward_tokens=include_reward_tokens,
        )
        self._tok2id = tok2id

        # Initialize tokenizer
        self.tk = Tokenizer(WordLevel(vocab=tok2id, unk_token=self._unk))
        self.tk.pre_tokenizer = WhitespaceSplit()
    
    def _pgn_to_tokens(self, text: str) -> Optional[List[str]]:
        """Convert PGN text to tokens."""
        import os, contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            g = chess.pgn.read_game(io.StringIO(text))
        if g is None:
            return None
        
        b, out, n = g.board(), [], 1
        for mv in g.mainline_moves():
            if b.turn == chess.WHITE and self._include_nums:
                out += list(str(n)) + (
                    ["..."] if self._include_black_ellipses and b.fullmove_number < n else ["."]
                )
            
            if b.is_castling(mv):
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                b.pop()
                out.append("O-O" if chess.square_file(mv.to_square) == 6 else "O-O-O")
                if suf:
                    out.append(suf)
                b.push(mv)
            else:
                piece = b.piece_at(mv.from_square).symbol().upper()
                frm = chess.square_name(mv.from_square)
                to = chess.square_name(mv.to_square)
                is_cap = b.is_capture(mv)
                promo = mv.promotion
                
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                
                # Emit LAN tokens
                out.append(piece)
                out.append(frm)
                if is_cap:
                    out.append("x")
                out.append(to)
                if promo:
                    out += ["=", chess.piece_symbol(promo).upper()]
                if suf:
                    out.append(suf)
            
            if b.turn == chess.WHITE:
                n += 1
        
        res = g.headers.get("Result")
        if self._keep_res and res in _RESULT:
            out.append(res)
        
        return out
    
    def _lan_move_to_tokens(self, move: str) -> List[str]:
        """
        Convert a single LAN move to tokens.
        
        LAN format: [Piece][from_square][x]?[to_square][=Promo]?[+#]?
        
        Examples:
            "Ng1f3" -> ["N", "g1", "f3"]
            "Nd4xe6" -> ["N", "d4", "x", "e6"]
            "Pe2e4" -> ["P", "e2", "e4"]
            "Pe4xd5" -> ["P", "e4", "x", "d5"]
            "O-O" -> ["O-O"]
            "O-O-O" -> ["O-O-O"]
            "Pe7e8=Q" -> ["P", "e7", "e8", "=", "Q"]
            "Ng1f3+" -> ["N", "g1", "f3", "+"]
        """
        # Handle castling
        if move in {"O-O", "O-O-O"}:
            return [move]
        if move.rstrip("+#") in {"O-O", "O-O-O"}:
            base = move.rstrip("+#")
            suffix = move[len(base):]
            return [base] + ([suffix] if suffix else [])
        
        out = []
        i = 0
        n = len(move)
        
        # Get piece letter (required in LAN format)
        if i < n and move[i] in "KQRBNP":
            out.append(move[i])
            i += 1
        else:
            # No piece letter - might be malformed, return as-is
            return [move]
        
        # Get from square (required in LAN format)
        if i + 1 < n and move[i] in FILES and move[i + 1] in RANKS:
            out.append(move[i:i+2])
            i += 2
        
        # Handle capture
        if i < n and move[i] == "x":
            out.append("x")
            i += 1
        
        # Get to square (required in LAN format)
        if i + 1 < n and move[i] in FILES and move[i + 1] in RANKS:
            out.append(move[i:i+2])
            i += 2
        
        # Handle promotion
        if i < n and move[i] == "=":
            out.append("=")
            i += 1
            if i < n and move[i] in PROMOS:
                out.append(move[i])
                i += 1
        
        # Handle check/checkmate
        if i < n and move[i] in "+#":
            out.append(move[i])
            i += 1
        
        return out
    
    def _active_env_tokens(self) -> set:
        """Return the set of env tokens that are active for this instance."""
        return set(ENV_TOKENS) if self._include_env_tokens else set()

    def _cot_to_tokens(self, text: str) -> List[str]:
        """
        Convert CoT formatted text to tokens.
        Handles special tokens and LAN moves.
        """
        env_toks = self._active_env_tokens()
        out = []
        for token in text.split():
            if token in {self.T, self.T_END, self.SEP} or token in env_toks:
                # Keep special tokens as-is
                out.append(token)
            elif token in _RESULT:
                # Game result
                out.append(token)
            elif token and token[0].isdigit() and "." in token:
                # Move number like "1." or "15..."
                # Split into digits and dots
                num_part = token.rstrip(".")
                dot_part = token[len(num_part):]
                out.extend(list(num_part))
                if dot_part:
                    out.append("..." if len(dot_part) > 1 else ".")
            elif token and all(c.isdigit() for c in token):
                # Pure number - tokenize each digit
                out.extend(list(token))
            else:
                # LAN move - tokenize it
                out.extend(self._lan_move_to_tokens(token))
        return out
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode (can be PGN or CoT formatted)
        
        Returns:
            List of token IDs
        """
        # Check if this is CoT-formatted text (contains special tokens)
        sft_special = (
            [self.T, self.T_END, self.SEP]
            + (ENV_TOKENS if self._include_env_tokens else [])
        )
        is_cot_format = any(token in text for token in sft_special)
        
        if is_cot_format:
            t_idx = text.index(self.T)
            prompt_part = text[:t_idx].strip()
            rest_part = text[t_idx:]  # starts with <T>

            pgn_tokens = self._pgn_to_tokens(prompt_part) if prompt_part else None
            if pgn_tokens is None:
                pgn_tokens = self._cot_to_tokens(prompt_part) if prompt_part else []
            rest_tokens = self._cot_to_tokens(rest_part)
            tokens = [self._bos] + pgn_tokens + rest_tokens + [self._eos]
        else:
            pgn_tokens = self._pgn_to_tokens(text)
            if pgn_tokens is not None and len(pgn_tokens) > 0:
                tokens = [self._bos] + pgn_tokens + [self._eos]
            else:
                # Not valid PGN — treat each word as a LAN move
                lan_tokens = []
                for word in text.split():
                    lan_tokens.extend(self._lan_move_to_tokens(word))
                tokens = [self._bos] + lan_tokens + [self._eos]
        
        return self.tk.encode(" ".join(tokens)).ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        toks = [t for t in self.tk.decode(ids).split() if t not in {self._bos, self._eos}]
        
        # Otherwise, use LAN decoding logic
        out: List[str] = []
        i, n = 0, len(toks)
        
        while i < n:
            t = toks[i]
            
            if t in {self.T, self.T_END, self.SEP} or t in _RESULT or t in self._active_env_tokens():
                out.append(t)
                i += 1
                continue
            
            if t and all(ch in DIGITS for ch in t):
                j = i
                num = []
                while j < n and all(ch in DIGITS for ch in toks[j]):
                    num.append(toks[j])
                    j += 1
                dots = ""
                if j < n and toks[j] in {".", "..."}:
                    dots = toks[j]
                    j += 1
                out.append("".join(num) + dots)
                i = j
                continue
            
            if t in {"O-O", "O-O-O"}:
                j = i + 1
                suf = toks[j] if j < n and toks[j] in {"+", "#"} else ""
                if suf:
                    j += 1
                out.append(t + suf)
                i = j
                continue
            
            if t in set("KQRBNP"):
                piece = t
                j = i + 1
                frm = toks[j] if j < n else ""
                j += 1
                cap = ""
                if j < n and toks[j] == "x":
                    cap = "x"
                    j += 1
                to = toks[j] if j < n else ""
                j += 1
                promo = ""
                if j + 1 <= n - 1 and toks[j] == "=" and toks[j + 1] in set(PROMOS):
                    promo = "=" + toks[j + 1]
                    j += 2
                suf = ""
                if j < n and toks[j] in {"+", "#"}:
                    suf = toks[j]
                    j += 1
                lan = f"{piece}{frm}{cap}{to}{promo}{suf}"
                out.append(lan)
                i = j
                continue
            
            out.append(t)
            i += 1
        
        return " ".join(out)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get token-to-id vocabulary mapping."""
        return self._tok2id
    
    def bos_id(self) -> Optional[int]:
        """Get BOS token ID."""
        return self._tok2id[self._bos]
    
    def eos_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self._tok2id[self._eos]
    
    def pad_id(self) -> Optional[int]:
        """Get PAD token ID (uses BOS as pad by default)."""
        return self._tok2id.get("<pad>", self.bos_id())
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._tok2id)
    
    def t_id(self) -> int:
        """Get <T> token ID."""
        return self._tok2id[self.T]
    
    def sep_id(self) -> int:
        """Get <sep> token ID."""
        return self._tok2id[self.SEP]
    
    def t_end_id(self) -> int:
        """Get </T> token ID."""
        return self._tok2id[self.T_END]
    
    # ------------------------------------------------------------------
    # Environment / reward token accessors
    # ------------------------------------------------------------------

    def _require_env_tokens(self) -> None:
        if not self._include_env_tokens:
            raise ValueError(
                "Environment tokens are not enabled. "
                "Pass include_env_tokens=True in the config."
            )

    def call_env_id(self) -> int:
        """Get <call_env> token ID."""
        self._require_env_tokens()
        return self._tok2id[CALL_ENV_TOKEN]

    def verify_id(self) -> int:
        """Get <verify> token ID."""
        self._require_env_tokens()
        return self._tok2id[VERIFY_TOKEN]

    def reward_pos_id(self) -> int:
        """Get <+1> (positive reward) token ID."""
        self._require_env_tokens()
        return self._tok2id[REWARD_POS_TOKEN]

    def reward_neg_id(self) -> int:
        """Get <-1> (negative reward) token ID."""
        self._require_env_tokens()
        return self._tok2id[REWARD_NEG_TOKEN]

    def reward_zero_id(self) -> int:
        """Get <0> (zero reward) token ID."""
        self._require_env_tokens()
        return self._tok2id[REWARD_ZERO_TOKEN]

    def reward_id(self, value) -> int:
        """
        Get reward token ID by numeric value.

        Args:
            value: 1, -1, or 0  (or the strings "+1", "-1", "0")

        Returns:
            Token ID for the corresponding reward token.
        """
        self._require_env_tokens()
        mapping = {1: REWARD_POS_TOKEN, -1: REWARD_NEG_TOKEN, 0: REWARD_ZERO_TOKEN,
                   "+1": REWARD_POS_TOKEN, "-1": REWARD_NEG_TOKEN, "0": REWARD_ZERO_TOKEN}
        if value not in mapping:
            raise ValueError(f"reward value must be one of 1, -1, 0 (or '+1', '-1', '0'), got {value!r}")
        return self._tok2id[mapping[value]]

    def env_token_ids(self) -> Dict[str, int]:
        """Get mapping of all env/reward special tokens to their IDs."""
        self._require_env_tokens()
        return {tok: self._tok2id[tok] for tok in ENV_TOKENS}
    
    def extract_parts(self, text: str) -> Tuple[Optional[str], Optional[List[str]], str]:
        """
        Extract prompt, trajectories and answer from BoN CoT formatted text.
        
        Args:
            text: Text in format: {prompt} <T> <sep> {traj1} <sep> ... <sep> <T> {answer}
        
        Returns:
            prompt: The prompt/context (or None if not present)
            trajectories: List of trajectory strings (or None if not present)
            answer: The final answer
        """
        if self.T not in text:
            return None, None, text
        
        try:
            # Split by <T> to get prompt, thinking section, and answer
            t_parts = text.split(self.T)
            if len(t_parts) < 3:
                return None, None, text
            
            # t_parts[0] is prompt (before first <T>)
            # t_parts[1] is the thinking section with trajectories
            # t_parts[2] is the answer
            prompt = t_parts[0].strip() if t_parts[0].strip() else None
            thinking_section = t_parts[1].strip()
            answer = t_parts[2].strip()
            
            # Split thinking section by <sep> to get trajectories
            trajectories = [t.strip() for t in thinking_section.split(self.SEP) if t.strip()]
            
            return prompt, trajectories, answer
        except (ValueError, IndexError):
            return None, None, text
    
    def extract_thinking_and_answer(self, text: str) -> Tuple[Optional[List[str]], str]:
        """
        Extract trajectories and answer from BoN CoT formatted text (ignores prompt).
        
        Args:
            text: Text in format: {prompt} <T> <sep> {traj1} <sep> ... <sep> <T> {answer}
        
        Returns:
            trajectories: List of trajectory strings (or None if not present)
            answer: The final answer
        """
        _, trajectories, answer = self.extract_parts(text)
        return trajectories, answer
    
    def get_sft_special_tokens(self) -> List[str]:
        """Get list of SFT special tokens (including env/reward tokens if enabled)."""
        toks = [self.T, self.T_END, self.SEP]
        if self._include_env_tokens:
            toks += ENV_TOKENS
        return toks

    def get_sft_token_ids(self) -> Dict[str, int]:
        """Get mapping of SFT special tokens to their IDs."""
        result = {
            self.T: self._tok2id[self.T],
            self.T_END: self._tok2id[self.T_END],
            self.SEP: self._tok2id[self.SEP],
        }
        if self._include_env_tokens:
            for tok in ENV_TOKENS:
                result[tok] = self._tok2id[tok]
        return result
    
    def parse_cot_line(self, line: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Parse a CoT data line in format: <T> <sep> ... <sep> <T> {answer}
        
        Args:
            line: A line from the CoT data file
        
        Returns:
            trajectories: List of trajectory strings
            answer: The final answer/move
        """
        line = line.strip()
        if not line or not line.startswith(self.T):
            return None, None
        
        return self.extract_thinking_and_answer(line)

# ============================================================
# HuggingFace-compatible wrapper (auto-generated)
# ============================================================
import json as _json
from pathlib import Path as _Path
from transformers import PreTrainedTokenizer
import torch
from transformers.tokenization_utils_base import BatchEncoding

from huggingface_hub import hf_hub_download

class HFTokenizerWrapper(PreTrainedTokenizer):
    def __init__(self, model_max_length=2048, **kwargs):
        # These are usually provided by from_pretrained
        repo_id = kwargs.get("name_or_path") or kwargs.get("_name_or_path")
        revision = kwargs.get("revision", None)

        if not repo_id or "/" not in str(repo_id):
            # Fallback: user may pass repo_id explicitly
            repo_id = kwargs.get("repo_id", None)
        if not repo_id:
            raise ValueError("Cannot infer repo_id; pass repo_id=... or ensure name_or_path is set.")

        import os
        if os.path.isdir(repo_id):
            vocab_path = os.path.join(repo_id, "vocab.json")
            cfg_path   = os.path.join(repo_id, "tokenizer_config.json")
        else:
            vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", revision=revision)
            cfg_path   = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", revision=revision)

        with open(vocab_path, "r", encoding="utf-8") as _f:
            saved_vocab = _json.load(_f)
        with open(cfg_path, "r", encoding="utf-8") as _f:
            _tok_cfg = _json.load(_f)

        lan_config = _tok_cfg.get("lan_config", {})
        lan_class_name = _tok_cfg.get("lan_tokenizer_class", "LanTokenizerSFT")

        _cls = globals()[lan_class_name]
        custom_tokenizer = _cls(config=lan_config)

        # Override vocab with the saved vocab
        custom_tokenizer._tok2id = saved_vocab
        from tokenizers import Tokenizer as _TkTokenizer
        from tokenizers.models import WordLevel as _WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit as _WhitespaceSplit
        custom_tokenizer.tk = _TkTokenizer(_WordLevel(vocab=saved_vocab, unk_token=custom_tokenizer._unk))
        custom_tokenizer.tk.pre_tokenizer = _WhitespaceSplit()

        self.custom_tokenizer = custom_tokenizer
        self._vocab = dict(saved_vocab)
        self._id_to_token = {i: t for t, i in self._vocab.items()}

        bos_token = _tok_cfg.get("bos_token")
        eos_token = _tok_cfg.get("eos_token")
        pad_token = _tok_cfg.get("pad_token")
        unk_token = _tok_cfg.get("unk_token")
        env_token = _tok_cfg.get("env_token")
        if "env_id" in _tok_cfg:
            env_token = self._id_to_token[_tok_cfg.get("env_id")]
        else:
            env_token = _tok_cfg.get("env_token")
        self.env_token = env_token

        for _key in ("bos_token","eos_token","pad_token","unk_token","env_token",
                     "model_max_length","name_or_path","lan_config",
                     "lan_tokenizer_class","tokenizer_class","auto_map","use_fast",
                     "revision","repo_id"):
            kwargs.pop(_key, None)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    # ---- PreTrainedTokenizer interface ----

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        return []  # we override encode/decode directly

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, self.unk_token or "")

    def convert_tokens_to_string(self, tokens):
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self.custom_tokenizer.decode(ids)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def encode(self, text, add_special_tokens=True, **kwargs):
        ids = self.custom_tokenizer.encode(text)
        if add_special_tokens:
            return ids[:-1]  # strip trailing EOS; vLLM adds its own
        if (len(ids) >= 2
                and self.bos_token_id is not None
                and self.eos_token_id is not None
                and ids[0] == self.bos_token_id
                and ids[-1] == self.eos_token_id):
            return ids[1:-1]
        return ids

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        import numpy as np
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return self.custom_tokenizer.decode(token_ids)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        save_directory = _Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        vocab_file = save_directory / (
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            _json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        return (str(vocab_file),)

    def __call__(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if text_pair is not None:
            raise ValueError("text_pair not supported for this tokenizer.")

        # Normalize to batch
        is_batched = isinstance(text, (list, tuple))
        texts = list(text) if is_batched else [text]

        input_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        # Truncation
        if truncation and max_length is not None:
            if self.truncation_side == "left":
                input_ids = [ids[-max_length:] for ids in input_ids]
            else:
                input_ids = [ids[:max_length] for ids in input_ids]

        # Attention masks (pre-padding)
        attention_mask = [[1] * len(ids) for ids in input_ids]

        # Padding
        if padding:
            if padding == "max_length":
                if max_length is None:
                    raise ValueError("padding='max_length' requires max_length.")
                pad_to = max_length
            else:
                pad_to = max(len(ids) for ids in input_ids) if input_ids else 0

            pad_id = self.pad_token_id
            if pad_id is None:
                pad_id = self.bos_token_id if self.bos_token_id is not None else 0

            for i, ids in enumerate(input_ids):
                pad_len = pad_to - len(ids)
                if pad_len > 0:
                    input_ids[i] = ids + [pad_id] * pad_len
                    attention_mask[i] = attention_mask[i] + [0] * pad_len

        data = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Unbatch if single example and no tensor return
        if not is_batched and return_tensors is None:
            data = {"input_ids": data["input_ids"][0], "attention_mask": data["attention_mask"][0]}

        # Tensors
        if return_tensors == "pt":
            data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

        return BatchEncoding(data, tensor_type=None)


__all__ = ["HFTokenizerWrapper"]
