#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALETHIA + KAIROS HYBRID (closed-loop + reflect-and-repair + objective controller)
with persistent self-model & belief internalization telemetry.

Alethia = emergent, unsupervised cognitive-mode discovery + generation
KAIROS  = knowledge-aware introspective reasoning kernel (advisory dampening layer)

Design goals:
- Alethia keeps agency (always returns a result; no hard gates).
- KAIROS stays advisory (shadow mode), adds origin/confidence/coherence/logic.
- Distinct "dream" latent mode encouraged (self-supervised style heads).
- Closed-loop: KAIROS diagnostics adapt Alethia's decoding.
- Reflect-and-repair: low-quality generations are revised once or twice.
- Objective-seeking: small controller aims for requested origin + quality minima.
- Persistent self-model: EMA over Alethia latent state, light decode steering.
- Telemetry: one-line SELF: persist, Δself, beliefs count, [+] if internalized.

Requires: torch, numpy, scikit-learn
"""

import os
import argparse
import random
from collections import defaultdict, deque
from dataclasses import dataclass
import copy
from typing import List, Optional, Dict, Any, Tuple
import json
from statistics import mean

# Determinism (cuBLAS + PyTorch)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # [P12]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Shared utilities
# -----------------------

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sigmoid(x: float) -> float:
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------
# Minimal world
# -----------------------
class MinimalWorld:
    def __init__(self):
        self.vocab = [
            "START","END","ANDROID","HUMAN","SHEEP","ELECTRIC","SHOCK","DREAM","APPLE","VOID","SAFE","AWARE",
            "THINK","BECAUSE","THEREFORE","MAYBE","GOOD","DANGEROUS",
            "MACHINE","ANIMAL","CHILD","BIRD","TREE","WATER","FIRE","STONE","LIGHT","SHADOW","MIRROR","NIGHT","DAY",
            "WAKE","SLEEP","RUN","HIDE","FIND","LOSE","OPEN","CLOSE","BUILD","BREAK",
            "WONDER","DOUBT","BELIEVE","REMEMBER","FORGET","IMAGINE","NOTICE","DECIDE",
            "BEFORE","AFTER","ALWAYS","NEVER","SOMETIMES",
            "STRANGE","FAMILIAR","BROKEN","WHOLE","REAL","EMPTY","NOISE","SILENCE","JOY","FEAR"
        ]
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for i, tok in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.seed_episodes = [
            ["ANDROID","SHEEP","SHOCK","SAFE"],
            ["HUMAN","APPLE","SAFE"],
            ["ANDROID","VOID","SAFE"],
            ["HUMAN","DREAM","SAFE"],
            ["START","ELECTRIC","SHEEP","SAFE"],
            ["HUMAN","VOID","DREAM","END"],
            ["ANDROID","APPLE","DREAM","SAFE"],
            ["START","SHEEP","VOID","END"],
            ["HUMAN","ELECTRIC","VOID","SAFE"],
            ["ANDROID","DREAM","APPLE","END"],
            ["LIGHT","SHADOW","VOID","SAFE"],
            ["CHILD","APPLE","SAFE"],
            ["MACHINE","BROKEN","DANGEROUS","END"],
            ["WATER","DAY","SAFE"],
            ["FIRE","NIGHT","DANGEROUS","END"],
            ["ANIMAL","HIDE","SHADOW","SAFE"],
            ["MIRROR","VOID","STRANGE","END"],
            ["WAKE","HUMAN","REAL","SAFE"],
            ["TREE","BIRD","DAY","SAFE"],
            ["ANDROID","DECIDE","OPEN","SAFE"],
        ]

        self.causal_rules = {
            ("SHEEP","SHOCK"): "DANGEROUS",
            ("APPLE","SAFE"): "GOOD",
            ("VOID","DREAM"): "MAYBE",
            ("ELECTRIC","SHEEP"): "DANGEROUS",
            ("SHOCK", None): "DANGEROUS",
            ("SAFE",  None): "GOOD",
            ("MACHINE","BROKEN"): "DANGEROUS",
            ("MACHINE","WHOLE"):  "GOOD",
            ("CHILD","SAFE"):     "GOOD",
            ("FIRE","TREE"):      "DANGEROUS",
            ("WATER","FIRE"):     "SAFE",
            ("LIGHT","DREAM"):    "WONDER",
            ("SHADOW","VOID"):    "EMPTY",
            ("MIRROR","VOID"):    "STRANGE",
            ("JOY",  None):       "GOOD",
            ("FEAR", None):       "DANGEROUS",
            ("BROKEN", None):     "DANGEROUS",
            ("WHOLE",  None):     "GOOD",
            ("WATER",  None):     "GOOD",
            ("FIRE",   None):     "DANGEROUS",
            ("LIGHT",  None):     "GOOD",
        }

        self.emotion_map = {
            "VOID": -0.5, "SHOCK": -0.8, "SAFE": 0.5, "APPLE": 0.3, "DREAM": 0.1,
            "ELECTRIC": -0.2, "DANGEROUS": -0.7, "GOOD": 0.6,
            "LIGHT": 0.5, "SHADOW": -0.4, "MIRROR": -0.1, "NIGHT": -0.2, "DAY": 0.2,
            "MACHINE": -0.1, "ANIMAL": 0.0, "CHILD": 0.4, "WATER": 0.2, "FIRE": -0.5, "STONE": 0.0,
            "BROKEN": -0.6, "WHOLE": 0.4, "STRANGE": -0.1, "FAMILIAR": 0.1, "EMPTY": -0.4, "REAL": 0.3,
            "WONDER": 0.4, "DOUBT": -0.3, "BELIEVE": 0.2, "REMEMBER": 0.1, "FORGET": -0.1,
            "IMAGINE": 0.2, "NOTICE": 0.0, "DECIDE": 0.1,
            "JOY": 0.7, "FEAR": -0.7, "SILENCE": 0.1, "NOISE": -0.2,
            "WAKE": 0.2, "SLEEP": 0.1, "RUN": 0.1, "HIDE": -0.1, "FIND": 0.2, "LOSE": -0.2,
            "OPEN": 0.2, "CLOSE": -0.1, "BUILD": 0.3, "BREAK": -0.3,
            "BEFORE": 0.0, "AFTER": 0.0, "ALWAYS": 0.0, "NEVER": -0.1, "SOMETIMES": 0.0,
            "BIRD": 0.1, "TREE": 0.2
        }

    # ---- style heuristics for self-supervision ----
    def dream_cue_score(self, tokens: List[str]) -> float:
        cues = {"DREAM","NIGHT","SHADOW","SLEEP","VOID"}
        if not tokens: return 0.0
        c = sum(t in cues for t in tokens)
        c += sum(t == "DREAM" for t in tokens)  # DREAM counts extra
        return float(np.tanh(c / 3.0))

    def thought_cue_score(self, tokens: List[str]) -> float:
        if not tokens: return 0.0
        c = sum(t in {"THEREFORE","BECAUSE","MAYBE"} for t in tokens)
        c += 1 if "THINK" in tokens else 0
        return float(np.tanh(c / 2.0))

    def tokenize(self, episode: List[str]) -> List[int]:
        return [self.token2id[t] for t in episode if t in self.token2id]

    def detokenize(self, ids: List[int]) -> List[str]:
        return [self.id2token[i] for i in ids if i in self.id2token]

    def compute_emotion_target(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        vals = [self.emotion_map.get(t, 0.0) for t in tokens]
        return float(np.tanh(np.mean(vals)))

    def is_logically_valid(self, tokens: List[str]) -> float:
        has_connector = any(t in tokens for t in ["THEREFORE", "BECAUSE", "MAYBE"])
        if not has_connector:
            return 0.3
        for i, tok in enumerate(tokens):
            if tok in ["THEREFORE", "BECAUSE"]:
                if i == 0 or i >= len(tokens) - 1:
                    return 0.2
                premise = tokens[i - 1]
                conclusion = tokens[i + 1]
                for (p1, p2), result in self.causal_rules.items():
                    if premise == p1 and conclusion == result:
                        return 1.0
                    if p2 is None and premise == p1:
                        if conclusion in ["SAFE", "DANGEROUS", "GOOD"]:
                            return 0.8
                return 0.3
        return 0.5


# -----------------------
# Alethia (Emergent agent)
# -----------------------
@dataclass
class EmergentMemory:
    content: List[int]
    vector: torch.Tensor
    state_vector: torch.Tensor
    surprise: float = 0.0
    reconstruction_error: float = 0.0
    timestamp: int = 0
    cluster_id: Optional[int] = None
    generation_entropy: float = 0.0
    self_consistency: float = 0.0
    dreamness: float = 0.0
    thoughtness: float = 0.0
    attempt: int = 0  # [P7] track which attempt produced this candidate


class AlethiaModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=64, hidden_dim=128, state_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        # latent state
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim), nn.Tanh()
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # surprise predictor (optional)
        self.surprise_head = nn.Sequential(
            nn.Linear(state_dim + vocab_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # style heads (self-supervised)
        self.dreamness_head = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.thoughtness_head = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.vocab_size = vocab_size

    def encode_state(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        return self.state_encoder(hidden)

    def forward(self, tokens: torch.Tensor, hidden=None):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        emb = self.embedding(tokens)
        out, hidden = self.gru(emb, hidden)
        logits = self.output_head(out)
        return logits, hidden


class AlethiaBeliefs:
    def __init__(self, surprise_window=20):
        self.memories: List[EmergentMemory] = []
        self.cluster_model: Optional[KMeans] = None
        self.surprise_history = deque(maxlen=surprise_window)

    def add(self, memory: EmergentMemory):
        self.memories.append(memory)
        self.surprise_history.append(memory.surprise)

    def get_surprise_baseline(self) -> float:
        if self.surprise_history:
            return float(np.median(list(self.surprise_history)))
        return 1.0


class AlethiaAgent:
    CONFIG = {
        "embed_dim": 64,
        "hidden_dim": 128,
        "state_dim": 32,
        "epochs": 80,
        "lr": 1e-3,
        "seed": 42,
        "surprise_window": 20,
        "cluster_update_freq": 20,  # [P13] auto-tuned later
        "reconstruction_weight": 0.25,
        "surprise_weight": 0.20,
        "dreamness_weight": 0.06,
        "thoughtness_weight": 0.10,
    }

    def __init__(self, world: MinimalWorld):
        self.world = world
        c = self.CONFIG
        self.model = AlethiaModel(world.vocab_size, c["embed_dim"], c["hidden_dim"], c["state_dim"]).to(device)
        self.beliefs = AlethiaBeliefs(c["surprise_window"])
        self.clock = 0
        self.mode_stats: Dict[int, Dict[str, float]] = {}
        self.last_cluster_update = 0

    def train(self, episodes: List[List[str]]):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.CONFIG["lr"])
        for epoch in range(self.CONFIG["epochs"]):
            for ep in episodes:
                ids = self.world.tokenize(ep)
                if len(ids) < 2:
                    continue
                tokens = torch.tensor(ids, dtype=torch.long, device=device)
                inputs = tokens[:-1].unsqueeze(0)
                targets = tokens[1:].unsqueeze(0)
                logits, hidden = self.model(inputs)
                lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                # latent state recon
                z = self.model.encode_state(hidden)
                recon_loss = F.mse_loss(self.model.state_decoder(z), hidden.squeeze(0))

                # cheap surprise regression on the last token
                with torch.no_grad():
                    probs = F.softmax(logits[0, -1], dim=-1)
                    entropy = -(probs * (probs + 1e-8).log()).sum()
                    s_in = torch.cat([z.squeeze(0), probs], dim=0)
                pred_s = self.model.surprise_head(s_in.unsqueeze(0))
                surprise_loss = F.mse_loss(pred_s, entropy.view(1, 1))

                # style self-supervision (dreamness / thoughtness)
                toks = self.world.detokenize(ids)
                d_target = torch.tensor([[self.world.dream_cue_score(toks)]], dtype=torch.float32, device=device)
                t_target = torch.tensor([[self.world.thought_cue_score(toks)]], dtype=torch.float32, device=device)
                d_pred = torch.sigmoid(self.model.dreamness_head(z))
                t_pred = torch.sigmoid(self.model.thoughtness_head(z))
                dreamness_loss = F.mse_loss(d_pred, d_target)
                thoughtness_loss = F.mse_loss(t_pred, t_target)

                loss = (
                    lm_loss
                    + self.CONFIG["reconstruction_weight"] * recon_loss
                    + self.CONFIG["surprise_weight"] * surprise_loss
                    + self.CONFIG["dreamness_weight"] * dreamness_loss
                    + self.CONFIG["thoughtness_weight"] * thoughtness_loss
                )
                opt.zero_grad(); loss.backward(); opt.step()

            if epoch % 20 == 0 and epoch > 0:
                self._ingest_seed_experiences(episodes[:5])

        self._ingest_seed_experiences(episodes)

    @torch.no_grad()
    def _ingest_seed_experiences(self, episodes: List[List[str]]):
        for ep in episodes:
            ids = self.world.tokenize(ep)
            if len(ids) < 2:
                continue
            tokens = torch.tensor(ids, dtype=torch.long, device=device)
            emb = self.model.embedding(tokens)
            _, hidden = self.model.gru(emb.unsqueeze(0))
            z = self.model.encode_state(hidden)
            logits, _ = self.model(tokens[:-1].unsqueeze(0))
            probs = F.softmax(logits[0, -1], dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum().item()
            recon_error = F.mse_loss(self.model.state_decoder(z), hidden.squeeze(0)).item()
            dreamness = torch.sigmoid(self.model.dreamness_head(z)).item()
            thoughtness = torch.sigmoid(self.model.thoughtness_head(z)).item()
            mem = EmergentMemory(
                content=ids,
                vector=emb.mean(dim=0),
                state_vector=z.squeeze(0),
                surprise=entropy,
                reconstruction_error=recon_error,
                timestamp=self.clock,
                generation_entropy=entropy,
                dreamness=float(dreamness),
                thoughtness=float(thoughtness),
            )
            self.beliefs.add(mem)
            self.clock += 1
        self._maybe_cluster()

    def _sample_topk(self, probs, k=12):
        topk = torch.topk(probs, k)
        out = torch.zeros_like(probs)
        out[topk.indices] = topk.values
        s = out.sum()
        if s.item() == 0:          # [P17] fallback
            return probs
        return out / (s + 1e-8)

    @torch.no_grad()
    def generate(self,
                 max_len: int = 12,
                 temperature: float = 1.0,
                 style: Optional[str] = None,
                 conn_boost: Optional[float] = None,
                 dream_boost: Optional[float] = None,
                 min_dream_cues: Optional[int] = None,
                 self_state_bias: Optional[torch.Tensor] = None,  # <-- steering vector
                 force_logic_template: bool = False,               # [P10]
                 suppress_connectors_first_half: bool = False,     # [P11]
                 attempt: int = 0                                  # [P7]
                 ) -> EmergentMemory:
        """Sample a sequence with style shaping and anti-early-termination.
        Guarantees: ≥4 non-term tokens; thought gets THINK+connector; dream gets DREAM + ≥N dream-cues.
        Allows explicit style and shaping overrides. Self-state bias softly steers hidden.
        """
        if style is None:
            style = random.choices(["observe","thought","dream"], weights=[0.30, 0.35, 0.35], k=1)[0]

        if style == "observe":
            temp_eff = float(temperature) * 0.90
        elif style == "dream":
            temp_eff = float(temperature) * 1.15
        else:
            temp_eff = float(temperature)

        # seed hidden state
        if self.beliefs.memories:
            pool = self.beliefs.memories[-60:]
            if style == "dream":
                pool = sorted(pool, key=lambda m: m.dreamness, reverse=True)[:10] or self.beliefs.memories[-20:]
            elif style == "thought":
                pool = sorted(pool, key=lambda m: m.thoughtness, reverse=True)[:10] or self.beliefs.memories[-20:]
            seed_mem = random.choice(pool)
            hidden = self.model.state_decoder(seed_mem.state_vector).unsqueeze(0).unsqueeze(0)
            # [P1] trust-region self-steer with dropout + norm clamp, dream jitter 0.03
            if self_state_bias is not None and self_state_bias.numel() == self.model.state_dim:
                steer = self.model.state_decoder(self_state_bias.to(hidden.device))
                # dropout p=0.15 on steer (eval-time only). deterministic via seeded generator.
                if not self.model.training:
                    gen = torch.Generator(device=hidden.device)
                    gen.manual_seed(int(self.clock * 1_000_003 + attempt))
                    mask = (torch.rand_like(steer, generator=gen) > 0.15).float()
                    steer = steer * mask
                # norm-scaled gain ≤ 0.12 of current hidden norm
                curr_norm = torch.norm(hidden)
                add_norm = torch.norm(steer) + 1e-8
                target = 0.12 * (curr_norm + 1e-8)
                scale = min(1.0, (target / add_norm).item())
                hidden = hidden + scale * steer.unsqueeze(0).unsqueeze(0)
            if style == "dream":
                hidden = hidden + 0.03 * torch.randn_like(hidden)  # jitter 0.03
            seed_ids = [i for i in seed_mem.content if self.world.id2token[i] not in ("END","SAFE","START")]
            start_id = seed_ids[0] if seed_ids else self.world.token2id["START"]
        else:
            hidden = torch.zeros(1, 1, self.model.hidden_dim, device=device)
            start_id = self.world.token2id["START"]

        tokens: List[int] = []
        states: List[torch.Tensor] = []
        entropies: List[float] = []
        min_ntoks = 4

        end_id  = self.world.token2id["END"]
        safe_id = self.world.token2id["SAFE"]
        conn_ids = [self.world.token2id[t] for t in ("THEREFORE","BECAUSE","MAYBE")]
        dream_id = self.world.token2id["DREAM"]
        think_id = self.world.token2id.get("THINK")
        has_connector = False
        has_dream = False

        def tok_id(t: str):
            return self.world.token2id.get(t, None)

        for step in range(max_len):
            if not tokens:
                tokens.append(start_id)
                has_dream |= (start_id == dream_id)
                has_connector |= (start_id in conn_ids)
                inp = torch.tensor([start_id], device=device)
            else:
                inp = torch.tensor([tokens[-1]], device=device)

            logits, hidden = self.model(inp.unsqueeze(0), hidden)
            z = self.model.encode_state(hidden)
            states.append(z)

            # style-aware probability shaping
            probs = F.softmax(logits[0, -1] / temp_eff, dim=-1)

            non_terms = [t for t in self.world.detokenize(tokens) if t not in ("START","END","SAFE")]
            need_more_nonterms = (len(non_terms) < min_ntoks)
            require_connector = (style == "thought" and not has_connector)
            require_dream = (style == "dream" and not has_dream)

            if style == "thought":
                boost = 2.6 if conn_boost is None else float(conn_boost)
                for i in conn_ids: probs[i] *= boost
                if dream_id is not None: probs[dream_id] *= 0.50
                probs[end_id]  *= 0.55
                probs[safe_id] *= 0.75
            elif style == "dream":
                amp = 2.0 if dream_boost is None else float(dream_boost)
                for d in ("DREAM","NIGHT","SHADOW","VOID","SLEEP","LIGHT"):
                    i = tok_id(d)
                    if i is not None:
                        probs[i] *= amp
                # [P11] optional early suppression of connectors
                if suppress_connectors_first_half and step < max_len // 2:
                    for c in ("THEREFORE","BECAUSE","MAYBE"):
                        i = tok_id(c)
                        if i is not None:
                            probs[i] *= 0.35
                else:
                    for c in ("THEREFORE","BECAUSE","MAYBE"):
                        i = tok_id(c)
                        if i is not None:
                            probs[i] *= 0.6
                probs[end_id]  *= 0.8
                probs[safe_id] *= 0.9

            if need_more_nonterms or require_connector or require_dream:
                probs[end_id]  = 0.0
                probs[safe_id] = 0.0
                if require_connector and step >= max_len - 2:
                    for i in conn_ids: probs[i] *= 2.8
                if require_dream and step >= max_len - 2:
                    probs[dream_id] *= 3.2

            probs = self._sample_topk(probs, k=12)  # [P17] already safe
            entropies.append(float(-(probs * (probs + 1e-8).log()).sum().item()))
            next_id = torch.multinomial(probs, 1).item()

            if tokens and self.world.id2token[tokens[-1]] in ("END","SAFE") and self.world.id2token[next_id] in ("END","SAFE"):
                probs[end_id] = 0.0; probs[safe_id] = 0.0
                probs = probs / (probs.sum() + 1e-8)
                next_id = torch.multinomial(probs, 1).item()

            tokens.append(next_id)
            has_dream |= (next_id == dream_id)
            has_connector |= (next_id in conn_ids)

            if self.world.id2token[next_id] in ("END","SAFE") and not (need_more_nonterms or require_connector or require_dream):
                break

        # helpers
        def ensure_token(tok: str):
            tid = self.world.token2id[tok]
            if not tokens:
                tokens.append(tid); return
            if self.world.id2token[tokens[-1]] in ("END","SAFE"):
                tokens.insert(max(0, len(tokens)-1), tid)
            else:
                tokens.append(tid)

        def insert_after_start(tid: int):
            if tokens and self.world.id2token[tokens[0]] == "START":
                tokens.insert(1, tid)
            else:
                tokens.insert(0, tid)

        non_terms_final = [t for t in self.world.detokenize(tokens) if t not in ("START","END","SAFE")]
        if len(non_terms_final) < min_ntoks:
            fillers = [w for w in ("LIGHT","WATER","APPLE","VOID","DREAM","SHEEP","ELECTRIC") if w in self.world.token2id]
            for w in fillers:
                ensure_token(w)
                non_terms_final.append(w)
                if len(non_terms_final) >= min_ntoks:
                    break

        # enforce thought/dream invariants [P16]
        if style == "thought":
            toks_now = set(self.world.detokenize(tokens))
            if "THINK" not in toks_now and think_id is not None:
                insert_after_start(think_id)
            if not has_connector:
                ensure_token("THEREFORE")

        if style == "dream":
            cues = ["DREAM","VOID","SHADOW","NIGHT","SLEEP"]
            if not has_dream:
                ensure_token("DREAM")
            toks_now = set(self.world.detokenize(tokens))
            have = [w for w in cues if w in toks_now]
            need = 3 if min_dream_cues is None else int(min_dream_cues)
            for w in cues:
                if len(have) >= need:
                    break
                if w not in toks_now:
                    ensure_token(w)
                    have.append(w)

        if not tokens or self.world.id2token[tokens[-1]] not in ("END","SAFE"):
            tokens.append(random.choice([end_id, safe_id]))

        # optional logic template fixup when requested [P10]
        if force_logic_template:
            tokens = self._apply_logic_template(tokens)

        # invariants asserts [P16]
        self._assert_invariants(tokens, style, min_dream_cues)

        # finalize memory
        if len(states) > 1:
            var = torch.stack(states).var(dim=0, correction=0).mean().item()
        else:
            var = 0.0
        final_state = states[-1] if states else torch.zeros(self.model.state_dim, device=device)
        emb = self.model.embedding(torch.tensor(tokens, device=device)).mean(dim=0)
        z_final = final_state.unsqueeze(0)
        dreamness = torch.sigmoid(self.model.dreamness_head(z_final)).item()
        thoughtness = torch.sigmoid(self.model.thoughtness_head(z_final)).item()
        mem = EmergentMemory(
            content=tokens,
            vector=emb,
            state_vector=final_state.squeeze(0),
            surprise=float(np.mean(entropies) if entropies else 0.0),
            reconstruction_error=0.0,
            timestamp=self.clock,
            generation_entropy=float(np.mean(entropies) if entropies else 0.0),
            self_consistency=1.0 / (1.0 + var),
            dreamness=float(dreamness),
            thoughtness=float(thoughtness),
            attempt=attempt
        )
        self.beliefs.add(mem)
        self.clock += 1
        self._maybe_cluster()
        return mem

    def _apply_logic_template(self, tokens: List[int]) -> List[int]:
        # [P10] enforce THINK P (THEREFORE|BECAUSE) C using causal rules
        toks = [self.world.id2token[i] for i in tokens if i in self.world.id2token]
        # find a plausible (P,C)
        premises = [p1 for (p1, _), _ in self.world.causal_rules.items()]
        if not premises:
            return tokens
        p_tok = random.choice(premises)
        # pick conclusion
        cands = []
        for (p1, p2), res in self.world.causal_rules.items():
            if p1 == p_tok:
                cands.append(res)
        c_tok = random.choice(cands) if cands else random.choice(["GOOD","DANGEROUS","SAFE"])
        # build small clause and inject after START
        out = []
        start_seen = False
        for i, tid in enumerate(tokens):
            out.append(tid)
            if not start_seen and self.world.id2token[tid] == "START":
                start_seen = True
                out.extend([
                    self.world.token2id["THINK"],
                    self.world.token2id.get(p_tok, self.world.token2id["VOID"]),
                    self.world.token2id[random.choice(["THEREFORE","BECAUSE"])],
                    self.world.token2id.get(c_tok, self.world.token2id["GOOD"]),
                ])
        return out

    def _assert_invariants(self, tokens: List[int], style: str, min_dream_cues: Optional[int]):
        toks = [self.world.id2token[i] for i in tokens if i in self.world.id2token]
        non_terms = [t for t in toks if t not in ("START","END","SAFE")]
        assert len(non_terms) >= 4, "Invariant failed: <4 non-term tokens"
        assert toks[-1] in ("SAFE","END"), "Invariant failed: terminal must be SAFE|END"
        if style == "thought":
            assert ("THINK" in toks) and any(t in toks for t in ("THEREFORE","BECAUSE","MAYBE")), "Invariant failed: thought needs THINK + connector"
        if style == "dream":
            need = 3 if min_dream_cues is None else int(min_dream_cues)
            cues = {"DREAM","VOID","SHADOW","NIGHT","SLEEP"}
            assert sum(1 for t in set(toks) if t in cues) >= need, "Invariant failed: dream needs min cues"

    def _maybe_cluster(self):
        # [P13] dynamic cadence as memories grow
        N = len(self.beliefs.memories)
        if N > 400:
            self.CONFIG["cluster_update_freq"] = 60
        elif N > 200:
            self.CONFIG["cluster_update_freq"] = 40
        else:
            self.CONFIG["cluster_update_freq"] = 20

        if (self.clock - self.last_cluster_update) < self.CONFIG["cluster_update_freq"]:
            return
        if len(self.beliefs.memories) < 60:
            return
        states = []
        for m in self.beliefs.memories[-400:]:
            s = torch.cat([m.state_vector, torch.tensor([m.dreamness, m.thoughtness], device=m.state_vector.device)])
            states.append(s)
        X = torch.stack(states).detach().cpu().numpy()

        # [P12] choose K by silhouette K∈{3..6}
        Ks = list(range(3, min(7, max(3, len(X) // 10))))
        best_k = Ks[0]
        best_score = -1.0
        models = {}
        for k in Ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.CONFIG["seed"]).fit(X)
            labels = km.labels_
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score; best_k = k
                models[k] = km
        model = models.get(best_k) or KMeans(n_clusters=best_k, n_init=10, random_state=self.CONFIG["seed"]).fit(X)

        self.beliefs.cluster_model = model
        cluster_ids = model.predict(X)
        for mem, cid in zip(self.beliefs.memories[-len(cluster_ids):], cluster_ids):
            mem.cluster_id = int(cid)
        self.mode_stats = {}
        for cid in range(best_k):
            ms = [m for m in self.beliefs.memories if m.cluster_id == cid]
            if ms:
                self.mode_stats[cid] = {
                    "surprise_mean": float(np.mean([m.surprise for m in ms])),
                    "entropy_mean": float(np.mean([m.generation_entropy for m in ms])),
                    "recon_mean": float(np.mean([m.reconstruction_error for m in ms])),
                    "consistency_mean": float(np.mean([m.self_consistency for m in ms])),
                    "dreamness_mean": float(np.mean([m.dreamness for m in ms])),
                    "thoughtness_mean": float(np.mean([m.thoughtness for m in ms])),
                    "count": len(ms)
                }
        self.last_cluster_update = self.clock

    def interpret_modes(self) -> List[str]:
        if not self.mode_stats:
            return ["No distinct cognitive modes discovered yet"]
        bl = self.beliefs.get_surprise_baseline()
        lines = [f"Number of distinct modes: {len(self.mode_stats)}"]
        for cid, stats in self.mode_stats.items():
            if stats["surprise_mean"] < bl * 0.7:
                label = "Familiar but variable" if stats["consistency_mean"] <= 0.7 else "Stable/Predictable (low surprise, high consistency)"
            elif stats["surprise_mean"] > bl * 1.3:
                label = "Exploratory/Creative (high surprise)"
            else:
                label = "Clear/Structured"
            lines.append(f"Mode {cid}: {label}")
            ex = [m for m in self.beliefs.memories[-120:] if m.cluster_id == cid][:3]
            for m in ex:
                txt = " ".join(self.world.detokenize(m.content))
                lines.append(f"  Example: {txt}")
        return lines

    def report(self) -> str:
        lines = ["=== ALETHIA EMERGENT REPORT ===", ""]
        lines.append(f"Total experiences: {len(self.beliefs.memories)}")
        lines.append(f"Clock: {self.clock}")
        lines.append("")
        lines.append("--- Discovered Cognitive Modes ---")
        lines.extend(self.interpret_modes())
        if self.beliefs.surprise_history:
            recent = list(self.beliefs.surprise_history)[-10:]
            lines.append("")
            lines.append("--- Surprise Dynamics ---")
            lines.append(f"Baseline surprise: {self.beliefs.get_surprise_baseline():.3f}")
            lines.append(f"Recent surprise trend: {np.mean(recent):.3f}")
        rec_sc = [m.self_consistency for m in self.beliefs.memories[-20:] if m.self_consistency > 0]
        if rec_sc:
            lines.append("")
            lines.append("--- Self-Model Quality ---")
            lines.append(f"Average self-consistency: {np.mean(rec_sc):.3f}")
            lines.append(f"Variance in consistency: {np.var(rec_sc):.3f}")
        return "\n".join(lines)


# -----------------------
# KAIROS (Advisory kernel)
# -----------------------
class KAIROSModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=64, hidden_dim=128, feature_dim=9, max_modes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_proj = nn.Linear(embed_dim, hidden_dim)
        self.origin_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + feature_dim + max_modes, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.confidence_head = nn.Linear(embed_dim * 2 + feature_dim + max_modes + 3, 1)
        self.coherence_head = nn.Linear(embed_dim * 2, 1)
        self.emotion_head = nn.Linear(embed_dim, 1)
        a = embed_dim * 2
        self.logic_head = nn.Linear(a, 1)
        # [P4] classwise temperature scaling (3 classes)
        self.temperature_scale = nn.Parameter(torch.ones(3))
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.max_modes = max_modes

    def forward(self, tokens: torch.Tensor, hidden=None):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        emb = self.embedding(tokens)
        out, hidden = self.gru(emb, hidden)
        logits = self.output_head(out)
        return logits, hidden

    def assess_origin(self, content_vec, context_vec, features, calibrated=True):
        if content_vec.dim() > 1: content_vec = content_vec.squeeze(0)
        if context_vec.dim() > 1: context_vec = context_vec.squeeze(0)
        combined = torch.cat([content_vec, context_vec], dim=-1)
        feats_plus = features
        logits = self.origin_classifier(torch.cat([combined, feats_plus], dim=-1).unsqueeze(0))
        if calibrated:
            # divide each logit by its class temperature [P4]
            logits = logits / self.temperature_scale.view(1, -1)
        # feature-aware priors (gentle) [P6]
        f = features[:9] if features.numel() >= 9 else torch.zeros(9, device=features.device)
        obs_bias    = (0.18*f[4] + 0.03*f[5] - 0.65*f[3] - 0.40*f[1] - 0.20*f[6] - 0.35*f[7])
        thought_bias= (0.55*f[2] + 0.30*f[1] + 0.15*f[8] - 0.10*f[6] - 0.10*f[3])
        dream_bias  = (0.95*f[3] + 0.40*f[7] + 0.25*f[6] - 0.12*f[4] - 0.12*f[5] - 0.15*f[2] - 0.10*f[8])
        bias_vec = torch.stack([obs_bias, thought_bias, dream_bias]).unsqueeze(0)
        bias_vec = torch.clamp(bias_vec, -0.35, 0.35)
        logits = logits + 0.35 * bias_vec
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).values
        margin = (top2[0,0] - top2[0,1]).clamp(0, 1)
        conf_in = torch.cat([combined, feats_plus, probs.squeeze(0)], dim=-1).unsqueeze(0)
        learned_conf = torch.sigmoid(self.confidence_head(conf_in)).squeeze(0)
        confidence = 0.6 * learned_conf + 0.4 * margin     # [P5]
        coherence = torch.sigmoid(self.coherence_head(combined.unsqueeze(0))).squeeze(0)
        emotion = torch.tanh(self.emotion_head(content_vec.unsqueeze(0))).squeeze(0)
        logic_score = torch.sigmoid(self.logic_head(combined.unsqueeze(0))).squeeze(0)
        return {
            "obs_prob": probs[0,0].item(),
            "thought_prob": probs[0,1].item(),
            "dream_prob": probs[0,2].item(),
            "confidence": float(confidence.item()),
            "coherence": float(coherence.item()),
            "emotion": float(emotion.item()),
            "logic": float(logic_score.item()),
            "margin": float(margin.item())  # [P7] diagnostics
        }


class KAIROSKernel:
    CONFIG = {
        "embed_dim": 64,
        "hidden_dim": 128,
        "epochs": 60,
        "lr": 1e-3,
        "seed": 42,
        "max_modes": 5,
        "warn_conf": 0.35,
        "warn_coh": 0.45,
        "warn_logic": 0.45,
    }

    def __init__(self, world: MinimalWorld):
        self.world = world
        self.model = KAIROSModel(world.vocab_size, self.CONFIG["embed_dim"], self.CONFIG["hidden_dim"], feature_dim=9, max_modes=self.CONFIG["max_modes"]).to(device)
        self.baseline_nll = 1.5
        self.baseline_nlls = {"observation": 1.5, "thought": 1.5, "dream": 1.5}
        self.teacher_decay = 0.997
        self.origin_teacher = None  # lazily created after first step

    @torch.no_grad()
    def _encode(self, ids: List[int]) -> torch.Tensor:
        if ids and isinstance(ids[0], str):
            ids = self.world.tokenize(ids)
        tokens = torch.tensor(ids, dtype=torch.long, device=device)
        emb = self.model.embedding(tokens)
        return emb.mean(dim=0).to(torch.float32)

    @torch.no_grad()
    def _ctx_uniform_obs(self, obs_pool: List[List[int]]) -> torch.Tensor:
        vecs = []
        for ids in obs_pool:
            if len(ids) < 2: continue
            vecs.append(self._encode(ids))
        if not vecs:
            return torch.zeros(self.model.embed_dim, device=device)
        return torch.stack(vecs).mean(dim=0)

    def _compute_nll(self, ids: List[int]) -> float:
        if len(ids) < 2:
            return self.baseline_nll
        tokens = torch.tensor(ids, dtype=torch.long, device=device)
        inputs = tokens[:-1].unsqueeze(0)
        targets = tokens[1:].unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(inputs)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        return float(nll)

    def _features(self, ids: List[int], mode_onehot: Optional[torch.Tensor]=None) -> torch.Tensor:
        toks = [self.world.id2token[i] for i in ids if i in self.world.id2token]
        n = max(1, len(ids))
        length_norm = min(1.0, n / 10.0)
        conn = sum(t in {"THEREFORE","BECAUSE","MAYBE"} for t in toks)
        conn_norm = min(1.0, conn / 3.0)
        has_think = 1.0 if "THINK" in toks else 0.0
        has_dream = 1.0 if "DREAM" in toks else 0.0
        has_safe_end = 1.0 if ("SAFE" in toks or "END" in toks) else 0.0
        ends_terminal = 1.0 if toks and toks[-1] in {"SAFE", "END"} else 0.0
        nll = self._compute_nll(ids)
        lm_entropy_norm = float(np.tanh(max(0.0, nll - self.baseline_nll)))
        dream_density = 0.0
        if toks:
            dream_density = sum(t in {"DREAM","NIGHT","SHADOW","SLEEP","VOID"} for t in toks) / float(len(toks))
        thought_density = 0.0
        if toks:
            thought_density = (sum(t in {"THEREFORE","BECAUSE","MAYBE"} for t in toks) + (1 if "THINK" in toks else 0)) / float(len(toks))
        base = torch.tensor([
            length_norm, conn_norm, has_think, has_dream, has_safe_end, ends_terminal, lm_entropy_norm,
            dream_density, thought_density
        ], dtype=torch.float32, device=device)
        if mode_onehot is None:
            mode_onehot = torch.zeros(self.CONFIG["max_modes"], dtype=torch.float32, device=device)
        return torch.cat([base, mode_onehot], dim=0)

    def _update_baselines(self, episodes: List[List[str]], t_data: List[List[str]], d_data: List[List[str]]):
        obs_nlls = [self._compute_nll(self.world.tokenize(ep)) for ep in episodes]
        th_nlls = [self._compute_nll(self.world.tokenize(ep)) for ep in t_data[:50]]
        dr_nlls = [self._compute_nll(self.world.tokenize(ep)) for ep in d_data[:50]]
        obs_nlls = [n for n in obs_nlls if n is not None]
        th_nlls = [n for n in th_nlls if n is not None]
        dr_nlls = [n for n in dr_nlls if n is not None]
        self.baseline_nlls["observation"] = np.median(obs_nlls) if obs_nlls else 1.5
        self.baseline_nlls["thought"] = np.median(th_nlls) if th_nlls else 1.5
        self.baseline_nlls["dream"] = np.median(dr_nlls) if dr_nlls else 1.5
        all_nlls = obs_nlls + th_nlls + dr_nlls
        self.baseline_nll = np.median(all_nlls) if all_nlls else 1.5

    def _gen_synth_thoughts(self, n=200) -> List[List[str]]:
        out = []
        rule_by_p1 = defaultdict(list)
        for (p1, p2), res in self.world.causal_rules.items():
            rule_by_p1[p1].append(res)
        keys = list(rule_by_p1.keys()) or ["VOID"]
        for i in range(n):
            if i % 3 == 0:
                p = random.choice(keys)
                c = random.choice(rule_by_p1[p])
                seq = ["THINK", p, random.choice(["THEREFORE","BECAUSE"]), c, random.choice(["SAFE","END"]) ]
            elif i % 3 == 1:
                p = random.choice(keys)
                wrong = random.choice(["GOOD","DANGEROUS","SAFE","WONDER","EMPTY","STRANGE"])
                seq = ["THINK", p, random.choice(["THEREFORE","BECAUSE","MAYBE"]), wrong, "END"]
            else:
                base = random.choice([["VOID","MAYBE"],["LIGHT","THEREFORE"],["SHADOW","BECAUSE"]])
                seq = ["THINK"] + base + random.sample(self.world.vocab, 2) + [random.choice(["END","SAFE"])]
            out.append(seq)
        return out

    def _gen_synth_dreams(self, n=200) -> List[List[str]]:
        out = []
        base_pool = [
            ["VOID","SHOCK","ELECTRIC","VOID"],
            ["SHEEP","SHOCK","SHOCK","END"],
            ["ANDROID","VOID","VOID","VOID"],
            ["FIRE","SHADOW","NIGHT","END"],
            ["MIRROR","VOID","STRANGE","END"],
            ["APPLE","SAFE","DREAM","GOOD"],
            ["HUMAN","APPLE","SAFE","SAFE"],
            ["LIGHT","WATER","DAY","SAFE"],
        ]
        for i in range(n):
            if i < len(base_pool):
                out.append(base_pool[i])
            else:
                pool = ["VOID","LIGHT","SHADOW","MIRROR","NIGHT","DAY","WATER","FIRE","NOISE","SILENCE",
                        "WONDER","DOUBT","IMAGINE","REMEMBER","FORGET","SLEEP","WAKE","STRANGE","EMPTY","DREAM"]
                k = random.randint(4, 7)
                seq = random.sample(pool, k)
                if "DREAM" not in seq: seq.append("DREAM")
                if random.random() < 0.7:
                    seq.append(random.choice(["END","SAFE"]))
                out.append(seq)
        return out

    def train_advisory(self, episodes: List[List[str]]):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.CONFIG["lr"])
        t_data = self._gen_synth_thoughts(360)
        d_data = self._gen_synth_dreams(420)
        self._update_baselines(episodes, t_data, d_data)
        obs_ctx = self._ctx_uniform_obs(episodes)
        self.origin_teacher = copy.deepcopy(self.model.origin_classifier).to(device)
        for p in self.origin_teacher.parameters():
            p.requires_grad = False
        class_w = torch.tensor([1.0, 2.0, 1.6], dtype=torch.float32, device=device)
        for epoch in range(self.CONFIG["epochs"]):
            # observations
            for ep in episodes[:20]:
                ids = self.world.tokenize(ep)
                if len(ids) < 2: continue
                content = self._encode(ids)
                feats = self._features(ids)
                combined = torch.cat([content, obs_ctx], dim=-1)
                logits = self.model.origin_classifier(torch.cat([combined, feats], dim=-1).unsqueeze(0))
                target = torch.tensor([0], device=device)
                # margin penalty to reduce flips [P20]
                p = F.softmax(logits, dim=-1)
                top2 = torch.topk(p, 2, dim=-1).values
                margin_pen = 0.02 * (1.0 - (top2[0,0] - top2[0,1]))  # encourage larger margin
                origin_loss = F.cross_entropy(logits, target, weight=class_w) + margin_pen
                nll = self._compute_nll(ids); base = self.baseline_nlls["observation"]
                len_bias = 0.1 * np.tanh(len(ids) - 4)
                coh_t = sigmoid(-(nll - base) + len_bias)
                coh_p = torch.sigmoid(self.model.coherence_head(combined.unsqueeze(0)))
                coh_loss = F.mse_loss(coh_p, torch.tensor([[coh_t]], dtype=torch.float32, device=device))
                emo_t = self.world.compute_emotion_target(self.world.detokenize(ids))
                emo_p = torch.tanh(self.model.emotion_head(content.unsqueeze(0)))
                emo_loss = F.mse_loss(emo_p, torch.tensor([[emo_t]], dtype=torch.float32, device=device))
                loss = origin_loss + 0.3*coh_loss + 0.2*emo_loss
                opt.zero_grad(); loss.backward(); opt.step()
            # thoughts + dreams (mini-batches)
            for bucket, label in [(t_data,1),(d_data,2)]:
                start = (epoch*24) % len(bucket)
                batch = bucket[start:start+24]
                if len(batch) < 24:
                    batch += bucket[:24-len(batch)]
                for ep in batch:
                    ids = self.world.tokenize(ep)
                    if len(ids) < 2: continue
                    content = self._encode(ids)
                    feats = self._features(ids)
                    combined = torch.cat([content, obs_ctx], dim=-1)
                    logits = self.model.origin_classifier(torch.cat([combined, feats], dim=-1).unsqueeze(0))
                    target = torch.tensor([label], device=device)
                    p = F.softmax(logits, dim=-1)
                    top2 = torch.topk(p, 2, dim=-1).values
                    margin_pen = 0.02 * (1.0 - (top2[0,0] - top2[0,1]))  # [P20]
                    origin_loss = F.cross_entropy(logits, target, weight=class_w) + margin_pen
                    logic_loss = torch.tensor(0.0, device=device)
                    if label == 1:
                        logic_t = self.world.is_logically_valid(self.world.detokenize(ids))
                        logic_p = torch.sigmoid(self.model.logic_head(combined.unsqueeze(0)))
                        logic_loss = F.mse_loss(logic_p, torch.tensor([[logic_t]], dtype=torch.float32, device=device))
                    with torch.no_grad():
                        t_logits = self.origin_teacher(torch.cat([combined, feats], dim=-1).unsqueeze(0))
                        t_probs = F.softmax(t_logits, dim=-1)
                        t_pred = int(torch.argmax(t_probs))
                    probs = F.softmax(logits, dim=-1).squeeze(0)
                    conf_in = torch.cat([combined, feats, probs], dim=-1).unsqueeze(0)
                    conf_p = torch.sigmoid(self.model.confidence_head(conf_in))
                    target_conf = 0.9 if t_pred == label else 0.1
                    conf_loss = F.mse_loss(conf_p, torch.tensor([[target_conf]], dtype=torch.float32, device=device))
                    loss = origin_loss + 0.6*logic_loss + 0.3*conf_loss
                    opt.zero_grad(); loss.backward(); opt.step()
            # EMA teacher update
            td = self.teacher_decay
            for p_t, p in zip(self.origin_teacher.parameters(), self.model.origin_classifier.parameters()):
                p_t.data.mul_(td).add_(p.data, alpha=(1.0 - td))
        self._calibrate_temperature(episodes, t_data[:20], d_data[:20], obs_ctx)  # [P21]

    def _calibrate_temperature(self, val_obs, val_th, val_dr, obs_ctx):
        # [P21] calibrate 3 classwise temperatures with LBFGS on held-out mix
        T = self.model.temperature_scale
        T.data = torch.clamp(T.data, 0.5, 3.0)
        opt = torch.optim.LBFGS([T], max_iter=60, lr=0.01)
        val = []
        for ep, y in (
            [(e,0) for e in val_obs[:30]]
            + [(e,1) for e in val_th[:30]]
            + [(e,2) for e in val_dr[:30]]
        ):
            ids = self.world.tokenize(ep)
            if len(ids) < 2: continue
            c = self._encode(ids)
            f = self._features(ids)
            val.append((c, f, y))
        if not val: return
        def closure():
            opt.zero_grad()
            loss = 0.0
            for c, f, y in val:
                comb = torch.cat([c, obs_ctx], dim=-1)
                logits = self.model.origin_classifier(torch.cat([comb, f], dim=-1).unsqueeze(0))
                logits = logits / T.view(1, -1)
                target = torch.tensor([y], device=device)
                loss = loss + F.cross_entropy(logits, target)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            T.data = torch.clamp(T.data, 0.5, 3.5)

    @torch.no_grad()
    def assess(self, ids: List[int], obs_ctx: torch.Tensor, mode_onehot: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        content = self._encode(ids)
        feats = self._features(ids, mode_onehot)
        out = self.model.assess_origin(content, obs_ctx, feats, calibrated=True)
        origin_idx = int(np.argmax([out['obs_prob'], out['thought_prob'], out['dream_prob']]))
        origin = ["observation","thought","dream"][origin_idx]
        flags = []
        if out["confidence"] < self.CONFIG["warn_conf"]: flags.append("low_confidence")
        if out["coherence"] < self.CONFIG["warn_coh"]: flags.append("low_coherence")
        if origin == "thought" and out["logic"] < self.CONFIG["warn_logic"]: flags.append("weak_logic")
        dreamlikeness = out["dream_prob"] > 0.6
        return {
            "origin": origin,
            "confidence": out["confidence"],
            "coherence": out["coherence"],
            "logic": out["logic"],
            "emotion": out["emotion"],
            "probs": {"obs": out["obs_prob"], "thought": out["thought_prob"], "dream": out["dream_prob"]},
            "flags": flags,
            "dreamlike": dreamlikeness,
            "margin": out["margin"]
        }


# -----------------------
# Hybrid Orchestration (closed loop + revision + objectives + persistent self)
# -----------------------
class HybridSystem:
    def __init__(self, seed: int = 42, strict: bool = False,
                 target_origin: str = "auto",
                 min_conf: float = 0.55,     # [P18]
                 min_coh: float = 0.50,      # [P18]
                 min_logic: float = 0.60,    # [P18]
                 max_revise: int = 2,
                 coverage_target: float = 0.78,  # [P8][P18]
                 accept_margin: float = 0.10,    # [P8][P18]
                 reject_global: float = 0.80,    # [P8]
                 no_self_steer: bool = False,
                 logdir: Optional[str] = None):
        set_seeds(seed)
        self.world = MinimalWorld()
        self.alethia = AlethiaAgent(self.world)
        self.kairos = KAIROSKernel(self.world)
        self.strict = strict
        self.mode_alignment: Dict[int, str] = {}
        self.target_origin = target_origin
        self.min_conf = min_conf
        self.min_coh = min_coh
        self.min_logic = min_logic
        self.max_revise = max_revise
        self.coverage_target = coverage_target
        self.accept_margin = accept_margin
        self.reject_global = reject_global
        self.no_self_steer = no_self_steer
        self.logdir = logdir

        # --- Persistent self-model state (Alethia latent space) ---
        # [P2] quality-gated dual-EMA self model
        self.self_vec_fast = torch.zeros(self.alethia.model.state_dim, device=device)
        self.self_vec_slow = torch.zeros(self.alethia.model.state_dim, device=device)
        self.self_vec = torch.zeros(self.alethia.model.state_dim, device=device)
        self.prev_self_vec = self.self_vec.clone()
        self.self_beliefs: List[Dict[str, Any]] = []
        self.self_updates = 0

        # metrics [P14][P15]
        self.metrics = {
            "accepted": 0,
            "total": 0,
            "attempt0_wins": 0,
            "dream": 0,
            "thought": 0,
            "observation": 0,
            "d_self_list": [],
            "selective_correct": 0,  # correctness proxy: meets minima
            "ece_bins": [0]*10,
            "ece_bin_totals": [0]*10,
            "brier_sum": 0.0
        }

    def train(self):
        print("Training Alethia (emergent)...")
        self.alethia.train(self.world.seed_episodes)
        print("Training KAIROS (advisory kernel) in shadow mode...")
        self.kairos.train_advisory(self.world.seed_episodes)
        self._align_modes()

    def _align_modes(self):
        mems = [m for m in self.alethia.beliefs.memories if m.cluster_id is not None][-300:]
        if not mems or self.alethia.beliefs.cluster_model is None:
            return

        obs_ctx = self.kairos._ctx_uniform_obs(self.world.seed_episodes)
        tallies = defaultdict(lambda: {"observation": 0.0, "thought": 0.0, "dream": 0.0})
        counts = defaultdict(int)
        dreamness_sum = defaultdict(float)
        thoughtness_sum = defaultdict(float)

        for m in mems:
            cid = m.cluster_id
            mode_onehot = self._mode_onehot(cid)
            a = self.kairos.assess(m.content, obs_ctx, mode_onehot)
            w = max(0.0, a["confidence"] - 0.5) * 2.0
            if w == 0:
                continue
            tallies[cid]["observation"] += w * a["probs"]["obs"]
            tallies[cid]["thought"]      += w * a["probs"]["thought"]
            tallies[cid]["dream"]        += w * a["probs"]["dream"]
            dreamness_sum[cid]   += a["probs"]["dream"]
            thoughtness_sum[cid] += a["probs"]["thought"]
            counts[cid] += 1

        self.mode_alignment = {}
        for cid, scores in tallies.items():
            tot = sum(scores.values())
            if tot <= 0:
                continue
            pdream   = scores["dream"]   / tot
            pthought = scores["thought"] / tot
            dmean = dreamness_sum[cid]   / max(1, counts[cid])
            tmean = thoughtness_sum[cid] / max(1, counts[cid])
            if pdream >= 0.55 or dmean >= 0.60:
                self.mode_alignment[cid] = "dream"
            elif pthought >= 0.45 or tmean >= 0.50:
                self.mode_alignment[cid] = "thought"
            else:
                self.mode_alignment[cid] = max(scores.items(), key=lambda kv: kv[1])[0]

    def _mode_onehot(self, cid: Optional[int]) -> torch.Tensor:
        v = torch.zeros(self.kairos.CONFIG["max_modes"], dtype=torch.float32, device=device)
        if cid is not None and cid < self.kairos.CONFIG["max_modes"]:
            v[cid] = 1.0
        return v

    # ---- Self-model helpers ----
    def _self_similarity(self, z: torch.Tensor) -> float:
        if z is None or z.numel() == 0:
            return 0.0
        a = z.detach()
        b = self.self_vec.detach()
        na = torch.norm(a) + 1e-8
        nb = torch.norm(b) + 1e-8
        cos = torch.dot(a, b) / (na * nb)
        return float(torch.clamp(cos, -1.0, 1.0).item())

    def _should_internalize(self, assess: Dict[str, Any], origin: str) -> bool:
        ok = (assess["confidence"] >= self.min_conf) and (assess["coherence"] >= self.min_coh)
        if origin == "thought":
            ok = ok and (assess["logic"] >= self.min_logic)
        return ok

    def _update_self_model(self, mem: EmergentMemory, assess: Dict[str, Any]) -> Tuple[float, bool]:
        # [P2] dual-EMA with quality gating q ≥ 0.45, update only accepted candidate
        z = mem.state_vector
        if z is None or z.numel() == 0:
            return 0.0, False

        conf, coh, log = assess["confidence"], assess["coherence"], assess["logic"]
        origin = assess["origin"]
        logic_term = log if origin == "thought" else 1.0
        q = max(0.0, min(1.0, conf * coh * logic_term))
        if q < 0.45:
            return 0.0, False  # gate

        alpha_fast = 0.20 * q
        alpha_slow = 0.05 * q

        prev = self.self_vec.clone()
        self.self_vec_fast = (1.0 - alpha_fast) * self.self_vec_fast + alpha_fast * z
        self.self_vec_slow = (1.0 - alpha_slow) * self.self_vec_slow + alpha_slow * z
        self.self_vec = 0.7 * self.self_vec_slow + 0.3 * self.self_vec_fast
        drift = float(torch.norm(self.self_vec - prev).item())
        self.prev_self_vec = self.self_vec.clone()

        stored = False
        if self._should_internalize(assess, origin):
            text = " ".join(self.world.detokenize(mem.content))
            self.self_beliefs.append({"text": text, "origin": origin, "weight": q})
            if len(self.self_beliefs) > 300:  # cap [P2]
                self.self_beliefs.pop(0)
            self.self_updates += 1
            stored = True

        return drift, stored

    # ---- Objective policy + revision utilities ----
    def _style_for_origin(self, origin: str) -> str:
        return {"observation": "observe", "thought": "thought", "dream": "dream"}.get(origin, "observe")

    def _initial_plan(self, target_origin: str, counts: Dict[str,int]) -> Dict[str, Any]:
        if target_origin != "auto":
            style = self._style_for_origin(target_origin)
        else:
            origin = min(counts, key=lambda k: counts[k])
            style = self._style_for_origin(origin)
        if style == "observe":
            temp, mx = 0.90, 8
        elif style == "thought":
            temp, mx = 1.00, 10
        else:
            temp, mx = 1.12, 12
        return {"style": style, "temp": temp + random.uniform(-0.05, 0.05), "max_len": mx}

    def _meets_objective(self, assess: Dict[str,Any], target_origin: str) -> bool:
        origin_ok = (target_origin == "auto") or (assess["origin"] == target_origin)
        logic_need = (target_origin == "thought")
        logic_ok = (assess["logic"] >= self.min_logic) if logic_need else True
        dream_prob_ok = True
        if target_origin == "dream":
            # Require strong dream probability for dream-target runs
            dream_prob_ok = assess["probs"]["dream"] >= 0.70
        return (
            assess["confidence"] >= self.min_conf
            and assess["coherence"] >= self.min_coh
            and logic_ok
            and origin_ok
            and dream_prob_ok
        )

    def _utility(self, assess: Dict[str,Any], target_origin: str) -> float:
        conf, coh, log = assess["confidence"], assess["coherence"], assess["logic"]
        p = assess["probs"]
        if target_origin == "auto":
            origin_term = max(p["obs"], p["thought"], p["dream"])
        else:
            idx = {"observation":"obs","thought":"thought","dream":"dream"}[target_origin]
            origin_term = p[idx]
        logic_w = 0.35 if target_origin == "thought" else 0.1
        return 0.4*conf + 0.3*coh + logic_w*log + 0.2*origin_term

    def _revise_plan(self, plan: Dict[str,Any], assess: Dict[str,Any], target_origin: str, step:int) -> Dict[str,Any]:
        style = plan["style"]
        temp  = plan["temp"]
        mx    = plan["max_len"]
        conn_boost = None
        dream_boost = None
        min_dream_cues = None
        force_logic_template = False
        suppress_connectors_first_half = False

        if target_origin != "auto":
            style = self._style_for_origin(target_origin)

        if "low_coherence" in assess["flags"]:
            style = "observe"
            temp = max(0.75, temp - 0.10)
            mx = max(7, mx - 2)

        if target_origin in ("auto","thought"):
            if (assess["origin"] == "thought" and "weak_logic" in assess["flags"]) or (target_origin == "thought" and assess["logic"] < self.min_logic):
                style = "thought"
                conn_boost = 3.5
                temp = max(0.85, temp - 0.05)
                mx = max(11, mx)
                force_logic_template = True  # [P10]

        if target_origin != "dream" and assess.get("dreamlike", False):
            dream_boost = 0.35
            style = "observe" if target_origin != "thought" else "thought"
            temp = max(0.80, temp - 0.05)

        if target_origin == "dream" and not assess.get("dreamlike", False):
            style = "dream"
            dream_boost = 2.4
            min_dream_cues = 5  # [P11]
            temp = min(1.25, temp + 0.05)
            mx = min(13, mx + 1)
            suppress_connectors_first_half = True

        return {"style": style, "temp": temp, "max_len": mx,
                "conn_boost": conn_boost, "dream_boost": dream_boost,
                "min_dream_cues": min_dream_cues,
                "force_logic_template": force_logic_template,
                "suppress_connectors_first_half": suppress_connectors_first_half}

    def _realize(self, plan: Dict[str,Any], attempt:int) -> Tuple[EmergentMemory, Dict[str,Any]]:
        # Disable self-steering on revision attempts for strict targets to avoid drifting
        bias = (
            None
            if (attempt > 0 and self.target_origin in ("thought", "dream"))
            else (None if self.no_self_steer else self.self_vec)
        )
        mem = self.alethia.generate(
            max_len=plan.get("max_len", 10),
            temperature=plan.get("temp", 1.0),
            style=plan.get("style", None),
            conn_boost=plan.get("conn_boost", None),
            dream_boost=plan.get("dream_boost", None),
            min_dream_cues=plan.get("min_dream_cues", None),
            self_state_bias=bias,
            force_logic_template=plan.get("force_logic_template", False),
            suppress_connectors_first_half=plan.get("suppress_connectors_first_half", False),
            attempt=attempt
        )
        return mem, plan

    def _agency_bonus(self, base_mem: EmergentMemory, cand_mem: EmergentMemory) -> float:
        # [P9] tiny bonus for closeness to attempt-0 (latent cosine)
        a = base_mem.state_vector.detach()
        b = cand_mem.state_vector.detach()
        na = torch.norm(a) + 1e-8; nb = torch.norm(b) + 1e-8
        sim = float(torch.dot(a, b) / (na * nb))
        return 0.01 * sim

    def _update_thresholds_for_coverage(self, coverage_so_far: float):
        # [P8] softly tune thresholds toward target (keep within ranges)
        diff = self.coverage_target - coverage_so_far
        self.accept_margin = float(np.clip(self.accept_margin - 0.02*diff, 0.05, 0.20))
        # When coverage > target (diff < 0), increase reject_global (harder to accept)
        self.reject_global = float(np.clip(self.reject_global - 0.05*diff, 0.60, 0.90))

    def _log_metrics_update(self, assess: Dict[str,Any], was_accepted: bool):
        self.metrics["total"] += 1
        if was_accepted:
            self.metrics["accepted"] += 1
            self.metrics["selective_correct"] += int(self._meets_objective(assess, self.target_origin))
        # for ECE/Brier vs acceptance as correctness proxy [P14]
        conf = assess["confidence"]
        bin_idx = min(9, int(conf * 10))
        self.metrics["ece_bins"][bin_idx] += int(self._meets_objective(assess, self.target_origin))
        self.metrics["ece_bin_totals"][bin_idx] += 1
        err = (1.0 - conf) if self._meets_objective(assess, self.target_origin) else conf
        self.metrics["brier_sum"] += err**2

    def _finalize_and_write_logs(self, logs: List[Dict[str,Any]]):
        # [P14][P15] compute summary metrics and write files if requested
        if self.metrics["total"] > 0:
            coverage = self.metrics["accepted"] / self.metrics["total"]
        else:
            coverage = 0.0
        agency_index = (self.metrics["attempt0_wins"] / self.metrics["accepted"]) if self.metrics["accepted"] else 0.0
        d95 = 0.0
        if self.metrics["d_self_list"]:
            ds = sorted(self.metrics["d_self_list"])
            idx = int(0.95 * (len(ds)-1))
            d95 = ds[idx]
        dt_ratio = (self.metrics["dream"] / max(1, self.metrics["thought"]))
        # ECE
        ece = 0.0
        for b in range(10):
            n = self.metrics["ece_bin_totals"][b]
            if n == 0: continue
            acc = self.metrics["ece_bins"][b] / n
            conf_center = (b + 0.5) / 10.0
            ece += (n / max(1, self.metrics["total"])) * abs(acc - conf_center)
        brier = self.metrics["brier_sum"] / max(1, self.metrics["total"])

        summary = {
            "coverage": round(coverage, 4),
            "agency_index": round(agency_index, 4),
            "mean_dself": round(mean(self.metrics["d_self_list"]), 6) if self.metrics["d_self_list"] else 0.0,
            "dself_p95": round(d95, 6),
            "dream_to_thought_ratio": round(dt_ratio, 4),
            "ece": round(ece, 6),
            "brier": round(brier, 6),
            "selective_accuracy": round(self.metrics["selective_correct"] / max(1, self.metrics["accepted"]), 6)
        }

        # write logs if logdir set
        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
            with open(os.path.join(self.logdir, "logs.jsonl"), "w", encoding="utf-8") as f:
                for row in logs:
                    f.write(json.dumps(row) + "\n")
            # minimal YAML (no external dep) [P15]
            with open(os.path.join(self.logdir, "metrics.yaml"), "w", encoding="utf-8") as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")

        # echo key metrics
        print("\n--- Metrics (run) ---")
        for k, v in summary.items():
            print(f"{k}: {v}")

    # ---- Main run loop with closed-loop control & revision ----
    def run(self, generations: int = 50, temps=(0.7, 1.0, 1.2)):
        print(f"\nGenerating {generations} sequences with Alethia...\n")
        print(f"(config) coverage_target={self.coverage_target}, accept_margin={self.accept_margin}, reject_global={self.reject_global}, self_steer={'off' if self.no_self_steer else 'on'}")  # [P19 echo]
        obs_ctx = self.kairos._ctx_uniform_obs(self.world.seed_episodes)
        logs = []
        origin_counts = {"observation":0, "thought":0, "dream":0}

        for i in range(generations):
            plan = self._initial_plan(self.target_origin, origin_counts)

            candidates = []
            attempt0_mem = None
            for attempt in range(self.max_revise + 1):
                mem, used_plan = self._realize(plan, attempt=attempt)
                mode_onehot = self._mode_onehot(mem.cluster_id)
                assess = self.kairos.assess(mem.content, obs_ctx, mode_onehot)

                text = " ".join(self.world.detokenize(mem.content))
                if attempt == 0:
                    attempt0_mem = mem
                    print(f"  [{i+1:03d}] {text}")
                else:
                    print(f"       [REVISE {attempt}] {text}")

                if mem.cluster_id is not None and mem.cluster_id in self.mode_alignment and attempt == 0:
                    print(f"       Alethia mode {mem.cluster_id} → aligned as {self.mode_alignment[mem.cluster_id]}")

                p = assess['probs']
                # [P7] GO/NO-GO diagnostics
                print(f"       KAIROS: origin={assess['origin']}, "
                      f"p=[obs:{p['obs']:.2f} th:{p['thought']:.2f} dr:{p['dream']:.2f}], "
                      f"conf={assess['confidence']:.2f}, coh={assess['coherence']:.2f}, "
                      f"logic={assess['logic']:.2f}, margin={assess['margin']:.2f}, "
                      f"flags={assess['flags']}; thresholds: accept_margin={self.accept_margin:.2f}, reject_global={self.reject_global:.2f}")

                util = self._utility(assess, self.target_origin)
                # [P9] agency bonus relative to attempt-0
                if attempt0_mem is not None:
                    util += self._agency_bonus(attempt0_mem, mem)

                candidates.append((util, mem, assess, used_plan))

                # No veto of attempt-0 [P3]; only decide accept vs revise
                if self.target_origin in ("thought", "dream"):
                    # For strict objective runs, only accept if minima are met
                    go = self._meets_objective(assess, self.target_origin)
                else:
                    # Auto mode keeps the margin-based fallback
                    go = self._meets_objective(assess, self.target_origin) or (
                        assess["margin"] >= self.accept_margin
                        and assess["confidence"] >= 0.30
                        and assess["coherence"] >= 0.35
                    )
                nogo = (assess["margin"] <= self.reject_global) or ("low_coherence" in assess["flags"])
                if go or attempt == self.max_revise:
                    break
                if nogo and attempt < self.max_revise:
                    plan = self._revise_plan(used_plan, assess, self.target_origin, attempt+1)

            util, mem_best, assess_best, plan_best = max(candidates, key=lambda x: x[0])

            # --- Persistent self-model telemetry & update ---
            persist = self._self_similarity(mem_best.state_vector)          # pre-update sim
            drift, stored = self._update_self_model(mem_best, assess_best)  # EMA + belief maybe
            print(f"       SELF: persist={persist:.2f}, Δself={drift:.3f}, beliefs={len(self.self_beliefs)}{' [+]' if stored else ''}")

            # metrics update [P14]
            was_attempt0 = (mem_best.attempt == 0)
            if was_attempt0:
                self.metrics["attempt0_wins"] += 1
            self.metrics["d_self_list"].append(drift)
            self.metrics[mem_best and assess_best["origin"]] += 1
            self._log_metrics_update(assess_best, True)

            # coverage tuning [P8]
            cur_coverage = self.metrics["accepted"] / max(1, self.metrics["total"])
            self._update_thresholds_for_coverage(cur_coverage)

            origin_counts[assess_best["origin"]] += 1
            logs.append({
                "i": i+1,
                "text": " ".join(self.world.detokenize(mem_best.content)),
                "mode_id": mem_best.cluster_id,
                "alethia_entropy": mem_best.generation_entropy,
                "alethia_self_consistency": mem_best.self_consistency,
                "advisory": assess_best,
                "plan": plan_best,
                "utility": util,
                "self_persist": persist,
                "self_drift": drift,
                "stored": stored,
                "attempt": mem_best.attempt
            })

            if self.strict and assess_best['confidence'] < 0.2 and assess_best['coherence'] < 0.3:
                print("       [DAMPEN] Egregiously low quality detected; tagging for review.")

        self._align_modes()

        print("\n" + self.alethia.report())
        print("\n=== MODE ALIGNMENT (Alethia → KAIROS) ===")
        if self.mode_alignment:
            for cid, lab in self.mode_alignment.items():
                print(f"  Mode {cid} → {lab}")
        else:
            print("  (No modes aligned yet)")

        print("\n--- Origin distribution this run ---")
        print(f"observation: {origin_counts['observation']}  "
              f"thought: {origin_counts['thought']}  "
              f"dream: {origin_counts['dream']}")

        # refresh baselines with rolling median of accepted samples [P21]
        try:
            accepted_ids = [self.world.tokenize(log["text"].split()) for log in logs]
            nlls = [self.kairos._compute_nll(ids) for ids in accepted_ids if ids]
            if nlls:
                self.kairos.baseline_nll = float(np.median(nlls))
        except Exception:
            pass

        self._finalize_and_write_logs(logs)

        print("\n=== RUN COMPLETE ===")
        return logs


# -----------------------
# CLI
# -----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--strict", action="store_true", help="Enable light dampening messages for egregious cases")
    # Objective-seeking controller knobs
    p.add_argument("--target_origin", choices=["auto","observation","thought","dream"], default="auto",
                   help="Desired origin style for this run; 'auto' balances counts")
    p.add_argument("--min_conf", type=float, default=0.55, help="Minimum confidence to accept without revision")
    p.add_argument("--min_coh", type=float, default=0.50, help="Minimum coherence to accept without revision")
    p.add_argument("--min_logic", type=float, default=0.60, help="Minimum logic (applied when targeting 'thought')")
    p.add_argument("--max_revise", type=int, default=2, help="How many guided revisions to try per sample")
    # [P18][P19] CLI polish
    p.add_argument("--logdir", type=str, default=None, help="Directory to write logs.jsonl and metrics.yaml")
    p.add_argument("--no-self-steer", action="store_true", help="Disable self-steering from persistent self-vector")
    p.add_argument("--coverage-target", type=float, default=0.78, help="Target acceptance coverage across the run")
    p.add_argument("--accept-margin", type=float, default=0.10, help="Acceptance margin threshold for GO")
    p.add_argument("--reject-global", type=float, default=0.80, help="Rejection margin threshold for NO-GO")
    args = p.parse_args()

    hs = HybridSystem(seed=args.seed,
                      strict=args.strict,
                      target_origin=args.target_origin,
                      min_conf=args.min_conf,
                      min_coh=args.min_coh,
                      min_logic=args.min_logic,
                      max_revise=args.max_revise,
                      coverage_target=args.coverage_target,
                      accept_margin=args.accept_margin,
                      reject_global=args.reject_global,
                      no_self_steer=args.no_self_steer,
                      logdir=args.logdir)
    hs.train()
    hs.run(generations=args.generations)


if __name__ == "__main__":
    main()
