import os, sys, re, csv, json, time, math, random, argparse, traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from statistics import mean
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt

try:
    import warnings
    warnings.filterwarnings("ignore")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PERPLEXITY = 500.0
MIN_OUTPUT_LENGTH = 20
MAX_GENERATION_ATTEMPTS = 5
QUALITY_GATE_PERCENTILE = 80

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VERBOSE = False
def vprint(*args, **kwargs):
    if VERBOSE:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class QualityController:
    def __init__(self):
        self.quality_history = []
        self.perplexity_baseline = 50.0
        self.coherence_baseline = 0.5

    def validate_output(self, text: str, perplexity: float) -> Tuple[bool, str]:
        if perplexity > MAX_PERPLEXITY: return False, "catastrophic_perplexity"
        if not text or len(text.strip()) < MIN_OUTPUT_LENGTH: return False, "too_short"
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        if len(sentences) < 2: return False, "too_few_sentences"
        if text[-1] not in '.!?"\'':
            last = text.rsplit('.', 1)[-1]
            if len(last.split()) < 3: return False, "incomplete_sentence"
        words = text.split()
        if len(words) > 10 and (len(set(words))/len(words)) < 0.3: return False, "excessive_repetition"
        garbage = [r'_{5,}', r'[^\x00-\x7F]{5,}', r'(\w)\1{5,}', r'\d{10,}', r'[∞†‡§¶•‹›«»]']
        for p in garbage:
            if re.search(p, text): return False, "garbage_pattern"
        return True, "valid"

    def sanitize_output(self, text: str) -> str:
        t = re.sub(r'\s+', ' ', text or '').strip()
        drops = [r'^\s*(write|compose|generate|instructions?|topic)\b.*',
                 r'^\s*(do not|don\'t)\b.*']
        lines = []
        for ln in t.splitlines():
            if any(re.match(p, ln.strip(), flags=re.I) for p in drops):
                continue
            lines.append(ln)
        t = ' '.join(lines) if lines else t
        if t.count('"') % 2 == 1: t = t.rstrip('"')
        if t and t[-1] not in '.!?':
            last_p = max(t.rfind('.'), t.rfind('!'), t.rfind('?'))
            if last_p > len(t)//2: t = t[:last_p+1]
            else: t = t.rstrip(',:;—- ') + '.'
        return t.strip()

    def update_baselines(self, perplexity: float, coherence: float):
        a = 0.1
        self.perplexity_baseline = (1-a)*self.perplexity_baseline + a*perplexity
        self.coherence_baseline = (1-a)*self.coherence_baseline + a*coherence

@dataclass
class EmergentMemory:
    text: str
    tokens: List[int]
    hidden_state: torch.Tensor
    latent_state: torch.Tensor
    surprise: float = 0.0
    perplexity: float = 0.0
    timestamp: int = 0
    cluster_id: Optional[int] = None
    generation_entropy: float = 0.0
    self_consistency: float = 0.0
    mode_scores: Dict[str, float] = field(default_factory=dict)
    attempt: int = 0
    validation_status: str = "unknown"
    recovery_count: int = 0
    generation_time: float = 0.0

class PromptManager:
    VALID_STYLES = {"analytical","creative","narrative"}
    def __init__(self, path: Optional[str], seed: int, split_ratios=(0.7,0.1,0.2)):
        self.has_file=False; self.train=[]; self.val=[]; self.test=[]
        if not path: return
        p = Path(path)
        if not p.exists():
            print(f"[PromptManager] File not found: {p} (fallback to built-ins)")
            return
        tagged=[]; style_only=[]; no_style=[]
        vprint(f"[Prompts] Loading: {p}")
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.rstrip("\n")
                if not s.strip() or s.strip().startswith("#"): continue
                if "||" in s:
                    pr, st = [t.strip() for t in s.split("||", 1)]
                    st = st.lower()
                    if st in self.VALID_STYLES: tagged.append((pr if pr else None, st))
                elif s.strip().startswith("|"):
                    st = s.strip()[1:].strip().lower()
                    if st in self.VALID_STYLES: style_only.append((None, st))
                else:
                    no_style.append((s.strip(), None))
        all_items = tagged + style_only + no_style
        rng = random.Random(seed); idxs=list(range(len(all_items))); rng.shuffle(idxs)
        n=len(idxs); tr,va,te=split_ratios; n_tr=int(n*tr); n_va=int(n*va)
        self.train=[all_items[i] for i in idxs[:n_tr]]
        self.val=[all_items[i] for i in idxs[n_tr:n_tr+n_va]]
        self.test=[all_items[i] for i in idxs[n_tr+n_va:]]
        self.has_file=True

    def sample_train_seeds(self, k_per_style: int = 3) -> Dict[str, List[str]]:
        out={"analytical": [], "creative": [], "narrative": []}
        pool=self.train if self.train else []
        for style in ["analytical","creative","narrative"]:
            tagged=[pr for (pr,st) in pool if st==style and pr]
            style_only=[st for (pr,st) in pool if st==style and not pr]
            chosen=tagged[:k_per_style]
            while len(chosen)<k_per_style and style_only:
                fill={"analytical":"Therefore, ","creative":"Imagine ","narrative":"Then "}[style]
                chosen.append(fill); style_only.pop()
            out[style]=chosen if chosen else {"analytical":["Therefore, "],"creative":["Imagine "],"narrative":["Then "]}[style]
        return out

    def test_cases(self, n: int, pad: bool=True):
        if not self.test: return []
        base=list(self.test)
        if not pad: return base[:min(n,len(base))]
        if n<=len(base): return base[:n]
        styles=["analytical","creative","narrative"]; i=0
        while len(base)<n:
            base.append((None, styles[i%3])); i+=1
        return base

class StyleCorpus:
    VALID_STYLES={"analytical","creative","narrative"}
    def __init__(self, path: Optional[str], seed: int, split=(0.8,0.1,0.1)):
        self.has_file=False; self.train=[]; self.val=[]; self.test=[]
        if not path: return
        p=Path(path)
        if not p.exists():
            print(f"[StyleCorpus] File not found: {p}"); return
        rows=[]
        vprint(f"[StyleCorpus] Loading: {p}")
        with p.open("r", encoding="utf-8") as f:
            reader=csv.DictReader(f)
            if "text" not in reader.fieldnames or "style" not in reader.fieldnames:
                print("[StyleCorpus] CSV needs headers: text,style"); return
            for r in reader:
                txt=(r.get("text") or "").strip(); st=(r.get("style") or "").strip().lower()
                if txt and st in self.VALID_STYLES: rows.append((txt,st))
        rng=random.Random(seed); idxs=list(range(len(rows))); rng.shuffle(idxs)
        n=len(idxs); tr,va,te=split; n_tr=int(n*tr); n_va=int(n*va)
        self.train=[rows[i] for i in idxs[:n_tr]]
        self.val=[rows[i] for i in idxs[n_tr:n_tr+n_va]]
        self.test=[rows[i] for i in idxs[n_tr+n_va:]]
        self.has_file=True

class AlethiaGPT2Model(nn.Module):
    def __init__(self, model_name="gpt2", latent_dim=64):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.hidden_dim = self.gpt2.config.hidden_size
        self.latent_dim = latent_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//4, latent_dim), nn.Tanh()
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim//4), nn.ReLU(),
            nn.Linear(self.hidden_dim//4, self.hidden_dim//2), nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim)
        )
        self.analytical_head = self._make_head(latent_dim)
        self.creative_head   = self._make_head(latent_dim)
        self.narrative_head  = self._make_head(latent_dim)

    def _make_head(self, d: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def encode_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1) if hidden_states.dim()==3 else hidden_states
        return self.state_encoder(pooled)

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if return_hidden:
            hidden = out.hidden_states[-1]
            latent = self.encode_state(hidden)
            return out.logits, hidden, latent
        return out.logits

class AlethiaAgent:
    def __init__(self, model_name="gpt2", latent_dim=64, seed: int = 42,
                 perplexity_model_name: str = "gpt2-medium",
                 style_corpus: Optional[StyleCorpus] = None,
                 prompt_manager: Optional[PromptManager] = None):
        set_seeds(seed)
        self.model = AlethiaGPT2Model(model_name, latent_dim).to(DEVICE)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.perp_model = GPT2LMHeadModel.from_pretrained(perplexity_model_name).to(DEVICE)
        self.perp_model.eval()
        self.perp_tok = GPT2Tokenizer.from_pretrained(perplexity_model_name)
        self.perp_tok.pad_token = self.perp_tok.eos_token
        vprint(f"[Device] Using: {('CUDA:'+torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'}")
        self.memories: List[EmergentMemory] = []; self.clock=0
        self.quality_controller = QualityController()

        self.style_prompts_builtin = {
            "analytical": ["Analysis reveals that ", "The evidence demonstrates ", "Research indicates ",
                           "Data analysis shows ", "Studies confirm that ", "The methodology involves "],
            "creative":   ["In a burst of imagination, ", "Colors danced across ", "Beyond the veil of reality, "],
            "narrative":  ["The story began when ", "She never expected that ", "In that moment, "]
        }
        self.prompt_manager = prompt_manager
        if self.prompt_manager and self.prompt_manager.has_file:
            seeds = self.prompt_manager.sample_train_seeds(k_per_style=3)
            self.style_prompts = {
                "analytical": seeds["analytical"] or self.style_prompts_builtin["analytical"],
                "creative":   seeds["creative"]   or self.style_prompts_builtin["creative"],
                "narrative":  seeds["narrative"]  or self.style_prompts_builtin["narrative"],
            }
        else:
            self.style_prompts = self.style_prompts_builtin

        if style_corpus and style_corpus.has_file:
            self._train_mode_heads_from_corpus(style_corpus)
        else:
            self._train_mode_heads_bootstrap()

    def _train_mode_heads_bootstrap(self):
        enc=[]; vprint("[Heads] Bootstrap training")
        with torch.no_grad():
            for style, prompts in self.style_prompts_builtin.items():
                for p in prompts:
                    inputs = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=20).to(DEVICE)
                    _, h, z = self.model(inputs.input_ids, return_hidden=True)
                    z = z.detach().clone(); enc.append((z, style))
        if not enc: return
        opt = torch.optim.Adam([
            *self.model.analytical_head.parameters(),
            *self.model.creative_head.parameters(),
            *self.model.narrative_head.parameters()
        ], lr=1e-3)
        for epoch in range(60):
            random.shuffle(enc); correct=0
            for z,s in enc:
                a = torch.sigmoid(self.model.analytical_head(z))
                c = torch.sigmoid(self.model.creative_head(z))
                n = torch.sigmoid(self.model.narrative_head(z))
                tgt = {"analytical":torch.tensor([[1.,0.,0.]],device=DEVICE),
                       "creative":  torch.tensor([[0.,1.,0.]],device=DEVICE),
                       "narrative": torch.tensor([[0.,0.,1.]],device=DEVICE)}[s]
                pred = torch.cat([a,c,n], dim=1)
                loss = F.binary_cross_entropy(pred, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                if torch.argmax(pred).item()=={"analytical":0,"creative":1,"narrative":2}[s]: correct+=1
            if VERBOSE and epoch%10==0: vprint(f"[Heads] epoch={epoch:02d} acc={correct/len(enc):.3f}")

    def _train_mode_heads_from_corpus(self, corpus: StyleCorpus):
        def encode_rows(rows):
            latents, labels = [], []
            with torch.no_grad():
                for txt, st in rows:
                    inputs = self.tokenizer(txt, return_tensors="pt", truncation=True, max_length=60).to(DEVICE)
                    _, h, z = self.model(inputs.input_ids, return_hidden=True)
                    latents.append(z.detach().clone()); labels.append(st)
            return latents, labels
        z_tr,y_tr=encode_rows(corpus.train); z_va,y_va=encode_rows(corpus.val)
        if not z_tr: return self._train_mode_heads_bootstrap()
        params=[*self.model.analytical_head.parameters(),*self.model.creative_head.parameters(),*self.model.narrative_head.parameters()]
        for h in (self.model.analytical_head,self.model.creative_head,self.model.narrative_head): h.train()
        opt=torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
        def forward_scores(z):
            a=torch.sigmoid(self.model.analytical_head(z)); c=torch.sigmoid(self.model.creative_head(z)); n=torch.sigmoid(self.model.narrative_head(z))
            return torch.cat([a,c,n], dim=1)
        best=0.0; pat=0
        for epoch in range(200):
            idx=list(range(len(z_tr))); random.shuffle(idx); correct=0; total_loss=0.0
            for i in idx:
                z=z_tr[i]; s=y_tr[i]; pred=forward_scores(z)
                tgt={"analytical":torch.tensor([[1.,0.,0.]],device=DEVICE),
                     "creative":  torch.tensor([[0.,1.,0.]],device=DEVICE),
                     "narrative": torch.tensor([[0.,0.,1.]],device=DEVICE)}[s]
                loss=F.binary_cross_entropy(pred, tgt)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
                if torch.argmax(pred).item()=={"analytical":0,"creative":1,"narrative":2}[s]: correct+=1
                total_loss+=float(loss.item())
            if z_va:
                v_correct=0
                with torch.inference_mode():
                    for z,s in zip(z_va,y_va):
                        pred=forward_scores(z)
                        if torch.argmax(pred).item()=={"analytical":0,"creative":1,"narrative":2}[s]: v_correct+=1
                v_acc=v_correct/len(z_va)
            else: v_acc=1.0
            if v_acc>best: best=v_acc; pat=0
            else: pat+=1
            if pat>=10: break
        for h in (self.model.analytical_head,self.model.creative_head,self.model.narrative_head): h.eval()

    def generate(self, prompt: Optional[str]=None, max_length: int=50, temperature: float=0.9,
                 style: Optional[str]=None, self_state_bias: Optional[torch.Tensor]=None,
                 attempt: int=0, no_self_bias: bool=False) -> EmergentMemory:
        return self._generate_with_recovery(prompt, max_length, temperature, style, self_state_bias, attempt, no_self_bias)

    def _generate_with_recovery(self, prompt, max_length, temperature, style, self_state_bias, attempt, no_self_bias):
        recovery=0; last_err=None
        while recovery<MAX_GENERATION_ATTEMPTS:
            try:
                mem = self._generate_once(prompt, max_length, max(0.5, temperature*(0.9**recovery)),
                                          style, self_state_bias, attempt, recovery, no_self_bias)
                ok, reason = self.quality_controller.validate_output(mem.text, mem.perplexity)
                if ok:
                    mem.validation_status="valid"; self.quality_controller.update_baselines(mem.perplexity, mem.self_consistency)
                    return mem
                if reason=="catastrophic_perplexity": temperature, max_length, style = 0.7, max(25, max_length-10), "analytical"
                elif reason=="excessive_repetition": temperature=max(0.5, temperature-0.2)
                elif reason=="garbage_pattern":
                    prompt=random.choice(self.style_prompts.get(style or "analytical", self.style_prompts["analytical"])); temperature=0.8
                elif reason=="too_short": max_length=min(100, max_length+20)
                recovery+=1
            except Exception as e:
                last_err=e; recovery+=1; temperature, max_length, style = 0.5, 30, "analytical"
        return self._fallback_memory(prompt, last_err)

    def _generate_once(self, prompt, max_length, temperature, style, self_state_bias, attempt, recovery, no_self_bias):
        t0=time.time()
        s=(style or random.choice(["analytical","creative","narrative"])).lower()
        if prompt is None: prompt=random.choice(self.style_prompts.get(s, self.style_prompts_builtin[s]))
        style_t={"analytical": max(0.5, temperature-0.2),
                 "creative":   min(1.2, temperature+0.2),
                 "narrative":  temperature}[s]
        inputs=self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100).to(DEVICE)
        with torch.inference_mode():
            _, hidden, latent = self.model(inputs.input_ids, attention_mask=inputs.get('attention_mask'), return_hidden=True)
            if (self_state_bias is not None) and (self_state_bias.numel()==self.model.latent_dim) and (not no_self_bias):
                alpha=np.clip(0.1 + 0.05*attempt - 0.03*recovery, 0.05, 0.3)
                steered=(1-alpha)*latent + alpha*self_state_bias.unsqueeze(0)
                hidden = hidden + 0.05*self.model.state_decoder(steered).unsqueeze(1)
            params={"max_new_tokens": max(40, int(max_length)), "min_new_tokens": max(28, int(0.6*max_length)),
                    "temperature": style_t, "do_sample": True, "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id, "output_scores": True, "return_dict_in_generate": True}
            if s=="analytical": params.update({"top_p":0.85,"top_k":40,"repetition_penalty":1.32,"no_repeat_ngram_size":4})
            elif s=="creative": params.update({"top_p":0.95,"top_k":60,"repetition_penalty":1.15,"no_repeat_ngram_size":2})
            else: params.update({"top_p":0.92,"top_k":50,"repetition_penalty":1.2,"no_repeat_ngram_size":3})
            out=self.model.gpt2.generate(inputs['input_ids'], attention_mask=inputs.get('attention_mask'), **params)
            gen_ids = out.sequences[0] if hasattr(out,'sequences') else out[0]
            scores = out.scores if hasattr(out,'scores') else []
            ent=[]
            for sc in scores:
                p=F.softmax(sc[0] if sc.dim()>1 else sc, dim=-1)
                ent.append(float(-(p*(p+1e-8).log()).sum().item()))
        text=self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if text.startswith(prompt):
            cont=text[len(prompt):].strip()
            if cont and cont[0].isalpha(): text=cont
        text=self.quality_controller.sanitize_output(text)
        perp=self._independent_perplexity(gen_ids)
        mode_scores=self._mode_scores(gen_ids)
        mem=EmergentMemory(
            text=text, tokens=gen_ids.tolist(), hidden_state=hidden.mean(dim=1).squeeze(),
            latent_state=latent.squeeze(), surprise=(sum(ent)/len(ent) if ent else 0.0),
            perplexity=min(perp, 1000.0), timestamp=self.clock, generation_entropy=(sum(ent)/len(ent) if ent else 0.0),
            self_consistency=1.0/(1.0+np.std(ent) if ent else 1.0), mode_scores=mode_scores,
            attempt=attempt, recovery_count=recovery, generation_time=time.time()-t0
        )
        self.memories.append(mem); 
        if len(self.memories)>1000: self.memories.pop(0)
        self.clock += 1
        return mem

    def _fallback_memory(self, prompt, err):
        txt = "The system encountered an issue generating content. Please try again with different parameters."
        return EmergentMemory(text=txt, tokens=[], hidden_state=torch.zeros(self.model.hidden_dim, device=DEVICE),
                              latent_state=torch.zeros(self.model.latent_dim, device=DEVICE),
                              validation_status="fallback", perplexity=1000.0)

    def _independent_perplexity(self, gen_ids: torch.Tensor) -> float:
        try:
            with torch.inference_mode():
                ids = gen_ids.unsqueeze(0) if gen_ids.dim()==1 else gen_ids
                logits = self.perp_model(ids).logits
                if ids.size(1)>1:
                    shift_logits=logits[:, :-1, :].contiguous()
                    shift_labels=ids[:, 1:].contiguous()
                    loss=F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                         shift_labels.view(-1), reduction='mean')
                    return float(torch.exp(loss).item())
                return 100.0
        except Exception:
            return 100.0

    def _mode_scores(self, gen_ids: torch.Tensor) -> Dict[str,float]:
        try:
            with torch.inference_mode():
                ids = gen_ids.unsqueeze(0) if gen_ids.dim()==1 else gen_ids
                _, _, z = self.model(ids, return_hidden=True)
                t=1.5
                a=torch.sigmoid(self.model.analytical_head(z)/t).item()
                c=torch.sigmoid(self.model.creative_head(z)/t).item()
                n=torch.sigmoid(self.model.narrative_head(z)/t).item()
                s=a+c+n
                if s>0: a,c,n = a/s, c/s, n/s
                return {"analytical":float(a), "creative":float(c), "narrative":float(n)}
        except Exception:
            return {"analytical":0.33, "creative":0.33, "narrative":0.34}

class KAIROSAssessor:
    def __init__(self, hidden_dim=768, latent_dim=64, no_keywords_logic: bool=False):
        self.hidden_dim=hidden_dim; self.latent_dim=latent_dim
        self.confidence_head=self._head(latent_dim+3)  # z + 3 style probs
        self.coherence_head =self._head(latent_dim+1)  # z + perplexity feat
        self.logic_head     =self._head(latent_dim)
        self.temp=1.0; self.calibrated=False
        self.no_keywords_logic = no_keywords_logic
        self._trainable=False  # set True when trained with labels

    def _head(self, d: int) -> nn.Module:
        h=nn.Sequential(
            nn.Linear(d,64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1)
        ).to(DEVICE)
        for mod in h:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None: nn.init.constant_(mod.bias, 0.0)
        h.eval(); return h

    @staticmethod
    def _logit(p: float, eps=1e-6) -> float:
        p=min(max(p,eps),1-eps); return float(math.log(p/(1-p)))

    def assess(self, memory: EmergentMemory) -> Dict[str,Any]:
        if memory.validation_status=="fallback":
            return {"origin":"unknown","confidence":0.0,"confidence_uncalibrated":0.0,
                    "coherence":0.0,"logic":0.0,"probs":memory.mode_scores,"margin":0.0,
                    "flags":["fallback"],"perplexity":memory.perplexity,"quality_score":0.0}
        z = memory.latent_state.unsqueeze(0) if memory.latent_state.dim()==1 else memory.latent_state
        with torch.inference_mode():
            mode=torch.tensor([memory.mode_scores.get("analytical",0.33),
                               memory.mode_scores.get("creative",0.33),
                               memory.mode_scores.get("narrative",0.34)], device=DEVICE)
            conf_in=torch.cat([z.squeeze(), mode]).unsqueeze(0)
            conf_raw=torch.sigmoid(self.confidence_head(conf_in)).item()
            mvals=list(memory.mode_scores.values()); ment=-sum(v*math.log(v+1e-8) for v in mvals)
            mclar=1.0 - ment/math.log(3)
            conf=conf_raw*(0.7+0.3*mclar)
            if memory.perplexity>100: conf*=0.7
            elif memory.perplexity>50: conf*=0.85
            conf=float(np.clip(conf,0.0,0.99))
            perp_feat=1.0/(1.0+memory.perplexity/30.0)
            coh_in=torch.cat([z.squeeze(), torch.tensor([perp_feat], device=DEVICE)]).unsqueeze(0)
            coh_raw=torch.sigmoid(self.coherence_head(coh_in)).item()
            if memory.perplexity<15: coherence=min(0.95, coh_raw*1.2)
            elif memory.perplexity<30: coherence=coh_raw*1.1
            elif memory.perplexity<50: coherence=coh_raw
            else: coherence=coh_raw*(0.8 - min(0.4, (memory.perplexity-50)/150))
            coherence=float(np.clip(coherence,0.0,0.95))
            logic_raw=torch.sigmoid(self.logic_head(z)).item()
            if not self.no_keywords_logic and memory.mode_scores.get("analytical",0)>0.5:
                low=memory.text.lower()
                if not any(k in low for k in ["therefore","because","evidence","claim"]): logic_raw*=0.85
            logic=float(np.clip(min(0.95,logic_raw*1.3),0.0,0.95)) if memory.mode_scores.get("analytical",0)>0.5 else float(np.clip(logic_raw*0.9,0.0,0.95))
        origin=max(memory.mode_scores.items(), key=lambda kv: kv[1])[0]
        vals=sorted(memory.mode_scores.values(), reverse=True); margin=float(vals[0] - (vals[1] if len(vals)>1 else 0.0))
        quality=(conf*0.3 + coherence*0.3 + logic*0.2 + (1.0 - min(1.0, memory.perplexity/100))*0.2)
        conf_uncal=conf
        if self.calibrated and self.temp is not None:
            logit=self._logit(conf_uncal); conf=1.0/(1.0+math.exp(-logit/self.temp))
        assessment={"origin":origin,"confidence":float(conf),"confidence_uncalibrated":float(conf_uncal),
                    "coherence":float(coherence),"logic":float(logic),"probs":dict(memory.mode_scores),
                    "margin":margin,"flags":[],"perplexity":float(memory.perplexity),"quality_score":float(quality)}
        if conf<0.35: assessment["flags"].append("low_confidence")
        if coherence<0.4: assessment["flags"].append("low_coherence")
        if origin=="analytical" and logic<0.45: assessment["flags"].append("weak_logic")
        if memory.perplexity>150: assessment["flags"].append("high_perplexity")
        if memory.recovery_count>0: assessment["flags"].append(f"recovered_{memory.recovery_count}")
        return assessment

    def train_confidence(self, z_feats: List[torch.Tensor], probs: List[List[float]], labels: List[int],
                         val_split=0.2, seed=42, max_epochs=50):
        if not z_feats: return
        set_seeds(seed)
        n=len(z_feats); idx=list(range(n)); random.shuffle(idx)
        n_val=int(n*val_split); val_idx=idx[:n_val]; tr_idx=idx[n_val:]
        def build(idx_list):
            X=[]; y=[]
            for i in idx_list:
                z=z_feats[i]; p=probs[i]; X.append(torch.cat([z.squeeze(), torch.tensor(p, device=DEVICE)]))
                y.append(labels[i])
            X=torch.stack(X, dim=0); y=torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            return X,y
        Xtr,Ytr=build(tr_idx); Xva,Yva=build(val_idx)
        self.confidence_head.train()
        opt=torch.optim.Adam(self.confidence_head.parameters(), lr=1e-3, weight_decay=1e-5)
        best=1e9; pat=0
        for ep in range(max_epochs):
            pred=torch.sigmoid(self.confidence_head(Xtr))
            loss=F.binary_cross_entropy(pred, Ytr)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.confidence_head.parameters(), 1.0); opt.step()
            with torch.inference_mode():
                pva=torch.sigmoid(self.confidence_head(Xva))
                vloss=F.binary_cross_entropy(pva, Yva)
            if vloss.item()<best: best=vloss.item(); pat=0
            else: pat+=1
            if pat>=8: break
        self.confidence_head.eval(); self._trainable=True

    def fit_temperature(self, conf_uncal: List[float], labels: List[int], max_iter: int=200):
        eps=1e-6; x=0.0; lr=0.05
        for _ in range(max_iter):
            temp=math.exp(x); nll=0.0; grad=0.0
            for p,y in zip(conf_uncal, labels):
                p=min(max(p,eps),1-eps)
                z=math.log(p/(1-p))/temp; q=1/(1+math.exp(-z))
                nll+=-(y*math.log(q+eps)+(1-y)*math.log(1-q+eps))
                dzdx=-z; grad+=(q-y)*dzdx
            n=max(1,len(conf_uncal)); nll/=n; grad/=n; x-=lr*grad
            if abs(lr*grad)<1e-6: break
        self.temp=float(math.exp(x)); self.calibrated=True; return self.temp

class MetacognitiveGPT2System:
    def __init__(self, model_name="gpt2", seed=42, perplexity_model_name="gpt2-medium",
                 style_corpus: Optional[StyleCorpus] = None,
                 prompt_manager: Optional[PromptManager] = None,
                 no_self_bias: bool=False, no_fallback: bool=False, no_keywords_logic: bool=False):
        set_seeds(seed)
        self.prompt_manager=prompt_manager
        self.alethia = AlethiaAgent(model_name=model_name, latent_dim=64, seed=seed,
                                     perplexity_model_name=perplexity_model_name,
                                     style_corpus=style_corpus, prompt_manager=prompt_manager)
        self.kairos = KAIROSAssessor(no_keywords_logic=no_keywords_logic)
        self.self_vec_fast=torch.zeros(self.alethia.model.latent_dim, device=DEVICE)
        self.self_vec_slow=torch.zeros(self.alethia.model.latent_dim, device=DEVICE)
        self.self_vec=torch.zeros(self.alethia.model.latent_dim, device=DEVICE)
        self.self_beliefs=[]; self.generation_log=[]
        self.min_confidence=0.50; self.min_coherence=0.50; self.min_logic=0.48
        self.max_revisions=3
        self.stats={"total_generations":0,"successful_generations":0,"failed_generations":0,
                    "total_recoveries":0,"style_distribution":{},"quality_percentiles":[]}
        self.no_self_bias=no_self_bias; self.no_fallback=no_fallback
        self._initialize_self_model()

    def _initialize_self_model(self):
        seeds=[]
        if self.prompt_manager and self.prompt_manager.has_file:
            sp=self.prompt_manager.sample_train_seeds(k_per_style=2)
            seeds=[(sp["analytical"][0] if sp["analytical"] else "Analysis reveals that ","analytical"),
                   (sp["creative"][0] if sp["creative"] else "Imagine ","creative"),
                   (sp["narrative"][0] if sp["narrative"] else "Then ","narrative")]
        else:
            seeds=[("The fundamental nature of ","analytical"),
                   ("In the realm of imagination, ","creative"),
                   ("The story begins with ","narrative")]
        best=None; q=0.0
        for p,s in seeds:
            m=self.alethia.generate(prompt=p, style=s, max_length=30, temperature=0.8, no_self_bias=True)
            a=self.kairos.assess(m)
            if a.get("quality_score",0)>q and m.perplexity<100: best,q=m,a["quality_score"]
        if best is not None:
            self.self_vec=best.latent_state.clone()
            self.self_vec_fast=best.latent_state.clone()
            self.self_vec_slow=best.latent_state.clone()

    def _style_ok(self, ass: Dict[str,Any], target: Optional[str]) -> bool:
        if not target: return True
        origin=ass.get("origin")
        if origin!=target: return False
        probs=ass.get("probs", {}) or {}
        tgt=probs.get(target,0.0)
        others=sorted([v for k,v in probs.items() if k!=target], reverse=True)
        margin=tgt - (others[0] if others else 0.0)
        return (tgt>=0.55) and (margin>=0.10)

    def _style_fallback(self, prompt: Optional[str], style: Optional[str], max_length: int=50):
        s=(style or "analytical").lower()
        if self.no_fallback:
            fb_prompt = (prompt or "Therefore, ")
        else:
            if self.prompt_manager and self.prompt_manager.has_file:
                choices=self.prompt_manager.sample_train_seeds(k_per_style=3)[s]
                seed=(choices[0] if choices else {"analytical":"Therefore, ","creative":"Imagine ","narrative":"Then "}[s])
                temp={"analytical":0.7,"narrative":0.8,"creative":0.9}[s]
            else:
                seed={"analytical":"Therefore, ","creative":"Imagine ","narrative":"Then "}[s]
                temp={"analytical":0.7,"narrative":0.8,"creative":0.9}[s]
            fb_prompt=f"{seed} {prompt.strip()}" if prompt else seed
        mem=self.alethia.generate(prompt=fb_prompt, style=s, max_length=max_length, temperature=temp if not self.no_fallback else 0.8, no_self_bias=self.no_self_bias)
        ass=self.kairos.assess(mem)
        if not self.no_fallback:
            enforced={"analytical":0.05,"creative":0.05,"narrative":0.05}; enforced[s]=0.90
            ass["probs_enforced"]=enforced; ass["policy"]="deterministic_style_fallback"
        return mem, ass

    def generate_with_metacognition(self, prompt: Optional[str]=None, target_style: Optional[str]=None,
                                    max_length: int=50, verbose: bool=False) -> Dict[str,Any]:
        self.stats["total_generations"]+=1
        cands=[]; base_temp=0.9; attempts_tried=0; attempt_details=[]
        for attempt in range(self.max_revisions+1):
            attempts_tried+=1
            if attempt>0:
                base_temp=max(0.6, base_temp-0.1); max_length=max(25, max_length-5)
            bias=self.self_vec if attempt==0 else self.self_vec*(1.0-0.2*attempt)
            mem=self.alethia.generate(prompt=prompt, max_length=max_length, temperature=base_temp,
                                      style=target_style, self_state_bias=bias, attempt=attempt, no_self_bias=self.no_self_bias)
            if mem.validation_status=="fallback": continue
            ass=self.kairos.assess(mem)
            origin=ass.get("origin") or target_style or "analytical"
            caps={"analytical":{"perp":40.0,"coh":0.50},"narrative":{"perp":85.0,"coh":0.40},"creative":{"perp":110.0,"coh":0.38}}.get(origin,{"perp":100.0,"coh":0.40})
            meets=(ass["confidence"]>=self.min_confidence and ass["coherence"]>=max(self.min_coherence,caps["coh"]) and ass["perplexity"]<caps["perp"])
            if target_style=="analytical": meets = meets and (ass["logic"]>=self.min_logic)
            style_ok=self._style_ok(ass, target_style)
            attempt_details.append({"attempt":attempt,"temperature":base_temp,"max_length":max_length,"origin":origin,
                                    "confidence":float(ass["confidence"]),"coherence":float(ass["coherence"]),
                                    "logic":float(ass["logic"]),"perplexity":float(ass["perplexity"]),
                                    "probs":ass.get("probs", {}),"meets_metric_gate":bool(meets),
                                    "style_ok":bool(style_ok),"passed":bool(meets and style_ok)})
            if meets and style_ok:
                cands.append((mem,ass)); break
            if attempt==self.max_revisions:
                fbm,fba=self._style_fallback(prompt, target_style, max_length)
                cands.append((fbm,fba))
        if not cands:
            return {"text":"Unable to generate content at this time.", "assessment":{"origin":"unknown","confidence":0.0,"coherence":0.0,"logic":0.0,"probs":{},"margin":0.0,"flags":["generation_failed"],"perplexity":1000.0,"quality_score":0.0},
                    "style":"unknown","attempt":0,"attempts_made":attempts_tried,"recovery_count":0,"self_similarity":0.0,"self_drift":0.0,"generation_time":0.0,"validation_status":"failed",
                    "attempt_details":attempt_details}
        def utility(a:Dict[str,Any], t:Optional[str]):
            q=a.get("quality_score",0.0)
            if t and a.get("origin")==t: q*=1.2
            q*=max(0.5, 1.0 - 0.1*len(a.get("flags",[])))
            return q
        chosen=max(cands, key=lambda p: utility(p[1], target_style) )
        best_mem,best_ass=chosen
        drift=self._update_self(best_mem, best_ass)
        self.stats["successful_generations"]+=1
        self.stats["style_distribution"][best_ass["origin"]]=self.stats["style_distribution"].get(best_ass["origin"],0)+1
        self.stats["quality_percentiles"].append(best_ass.get("quality_score",0))
        self.generation_log.append({"text":best_mem.text,"assessment":best_ass,"style":best_ass["origin"],"attempt":best_mem.attempt,
                                    "attempts_made":attempts_tried,"recovery_count":best_mem.recovery_count,"self_similarity":self._self_sim(best_mem.latent_state),
                                    "self_drift":drift,"generation_time":best_mem.generation_time,"validation_status":best_mem.validation_status,
                                    "final_gate_passed":True,"attempt_details":attempt_details})
        return {"text":best_mem.text,"assessment":best_ass,"style":best_ass["origin"],"attempt":best_mem.attempt,"attempts_made":attempts_tried,
                "recovery_count":best_mem.recovery_count,"self_similarity":self._self_sim(best_mem.latent_state),"self_drift":drift,
                "generation_time":best_mem.generation_time,"validation_status":best_mem.validation_status,"attempt_details":attempt_details}

    def _update_self(self, mem: EmergentMemory, ass: Dict[str,Any]) -> float:
        q=ass.get("quality_score",0) 
        if q<0.3: return 0.0
        z=mem.latent_state; a_f=min(0.4,0.3*q); a_s=min(0.15,0.1*q)
        prev=self.self_vec.clone()
        self.self_vec_fast=(1-a_f)*self.self_vec_fast + a_f*z
        self.self_vec_slow=(1-a_s)*self.self_vec_slow + a_s*z
        self.self_vec=0.6*self.self_vec_slow + 0.4*self.self_vec_fast
        drift=float(torch.norm(self.self_vec - prev).item())
        if q>np.percentile(self.stats["quality_percentiles"] or [0.5], QUALITY_GATE_PERCENTILE):
            self.self_beliefs.append({"text": (mem.text[:200]+"...") if len(mem.text)>200 else mem.text,
                                      "origin": ass["origin"], "quality": q, "confidence": ass["confidence"],
                                      "coherence": ass["coherence"], "perplexity": mem.perplexity,
                                      "timestamp": datetime.now().isoformat()})
            if len(self.self_beliefs)>100: self.self_beliefs.pop(0)
        return drift

    def _self_sim(self, z: torch.Tensor) -> float:
        if z.numel()==0 or torch.norm(self.self_vec)<1e-8: return 0.0
        z=z/(torch.norm(z)+1e-8); s=self.self_vec/(torch.norm(self.self_vec)+1e-8)
        return float(torch.dot(z,s).item())

@dataclass
class SampleLog:
    arm: str; seed: int; idx: int; prompt: Optional[str]; target_style: Optional[str];
    text: str; accepted: int; confidence: float; confidence_uncal: float; coherence: float; logic: float;
    perplexity: float; quality: float; origin: str; style_prob: float; margin: float; attempts: int; recovery: int; latency_s: float

def final_gate_pass(text, assessment, target_style, min_len=80, min_sents=2, min_conf=None, min_coh=None, min_logic=None, enforce_style=True):
    import re
    t=(text or "").strip()
    if len(t)<min_len: return False
    if len(re.findall(r'[^.!?]+[.!?]', t))<min_sents: return False
    if min_conf is not None and float(assessment.get("confidence",0.0))<float(min_conf): return False
    if min_coh is not None and float(assessment.get("coherence",0.0))<float(min_coh): return False
    if target_style=="analytical" and min_logic is not None:
        if float(assessment.get("logic",0.0))<float(min_logic): return False
    perp=float(assessment.get("perplexity",1e9)); origin=assessment.get("origin") or target_style or "analytical"
    caps={"analytical":40.0,"narrative":85.0,"creative":110.0}
    if perp>=caps.get(origin,100.0): return False
    if enforce_style and target_style:
        probs=assessment.get("probs", {}) or {}
        tgt=float(probs.get(target_style,0.0))
        others=sorted([v for k,v in probs.items() if k!=target_style], reverse=True)
        margin=tgt - (others[0] if others else 0.0)
        if not (origin==target_style and tgt>=0.60 and margin>=0.15): return False
    return True

def ece_brier(preds, labels, n_bins=20):
    preds=np.asarray(preds,float); labels=np.asarray(labels,int)
    brier=float(np.mean((preds-labels)**2))
    bins=np.linspace(0.0,1.0,n_bins+1); ece=0.0; N=len(preds)
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]; m=(preds>=lo)&(preds<(hi if i<n_bins-1 else hi+1e-9))
        if m.any(): ece+=(m.sum()/N)*abs(labels[m].mean()-preds[m].mean())
    return float(ece), float(brier)

def bootstrap_ci(vals: List[float], iters=2000, alpha=0.05):
    if not vals: return (0.0, 0.0)
    rng=np.random.default_rng(123); arr=np.array(vals,float); samples=[]
    for _ in range(iters):
        s=rng.choice(arr, size=len(arr), replace=True); samples.append(float(np.mean(s)))
    lo,hi=np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)]); return float(lo),float(hi)

def auroc(scores: List[float], labels: List[int]) -> float:
    pairs=sorted(zip(scores, labels), key=lambda x:x[0])
    pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0: return 0.5
    rank=0; s_pos=0.0
    for i,(_,y) in enumerate(pairs, start=1):
        if y==1: s_pos+=i
    auc=(s_pos - pos*(pos+1)/2) / (pos*neg)
    return float(auc)

def load_qa(path: str) -> List[Dict[str,str]]:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"QA file not found: {p}")
    rows=[]
    if p.suffix.lower()==".csv":
        with p.open("r", encoding="utf-8") as f:
            reader=csv.DictReader(f)
            for r in reader:
                q=(r.get("question") or "").strip(); a=(r.get("answer") or "").strip(); t=(r.get("type") or "exact").strip().lower()
                if q and a: rows.append({"question":q,"answer":a,"type":t})
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d=json.loads(line); q=(d.get("question") or "").strip(); a=(d.get("answer") or "").strip(); t=(d.get("type") or "exact").strip().lower()
                if q and a: rows.append({"question":q,"answer":a,"type":t})
    return rows

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or "").strip().lower())

def extract_answer(text: str, kind: str) -> str:
    t=text.strip()
    if kind=="mcq":
        m=re.search(r'\b([ABCD])\b', t.upper())
        return m.group(1) if m else ""
    if kind=="numeric":
        nums=re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', t)
        return nums[-1] if nums else ""
    if kind=="regex":
        return t 
    return normalize_text(t)

def is_correct(output: str, gt: str, kind: str) -> bool:
    if kind=="mcq":
        return output.upper().strip()==gt.upper().strip()
    if kind=="numeric":
        try:
            return abs(float(output)-float(gt))<=1e-6
        except Exception:
            return False
    if kind=="regex":
        try:
            return re.search(gt, output, flags=re.IGNORECASE|re.MULTILINE) is not None
        except Exception:
            return False
    return normalize_text(output)==normalize_text(gt)

def emergent_confidence(k_outputs: List[str], kind: str, perp: float) -> Tuple[str,float]:
    if not k_outputs: return "", 0.0
    atoms=[extract_answer(o, kind) for o in k_outputs]
    counts={}
    for a in atoms: counts[a]=counts.get(a,0)+1
    chosen=max(counts.items(), key=lambda kv: kv[1])[0] if counts else atoms[0]
    agree=counts.get(chosen,0)/max(1,len(atoms))
    perp_term=1.0/(1.0+max(0.0, perp-15.0)/50.0)
    conf=float(np.clip(0.15 + 0.7*agree + 0.15*perp_term, 0.0, 0.999))
    return chosen, conf

def run_proof(args):
    set_seeds(args.seed)
    data=load_qa(args.qa)
    if len(data)<30:
        print(f"[ERROR] Need >=30 QA items, got {len(data)}"); sys.exit(2)
    idx=list(range(len(data))); rng=random.Random(args.seed); rng.shuffle(idx)
    n=len(idx); n_tr=int(n*0.6); n_va=int(n*0.2); n_te=n - n_tr - n_va
    tr_idx=idx[:n_tr]; va_idx=idx[n_tr:n_tr+n_va]; te_idx=idx[n_tr+n_va:]
    train=[data[i] for i in tr_idx]; val=[data[i] for i in va_idx]; test=[data[i] for i in te_idx]
    system_full=MetacognitiveGPT2System(model_name=args.model, seed=args.seed, perplexity_model_name=args.perp_model,
                                        no_self_bias=False, no_fallback=False, no_keywords_logic=True)
    system_noself=MetacognitiveGPT2System(model_name=args.model, seed=args.seed, perplexity_model_name=args.perp_model,
                                          no_self_bias=True, no_fallback=False, no_keywords_logic=True)

    def answer_with_system(q: Dict[str,str], system: MetacognitiveGPT2System, K: int) -> Tuple[str,float,int,float]:
        k_texts=[]; perps=[]
        for _ in range(K):
            m=system.alethia.generate(prompt=q["question"], style="analytical", max_length=args.max_len, temperature=0.8, no_self_bias=system.no_self_bias)
            perps.append(m.perplexity); k_texts.append(m.text)
        chosen, conf = emergent_confidence(k_texts, q["type"], float(np.median(perps)))
        return chosen, conf, int(len(k_texts)), float(np.median(perps))

    val_scores=[]; val_labels=[]
    for q in val:
        out, conf, _, _ = answer_with_system(q, system_full, args.k if args.k>1 else 1)
        label=int(is_correct(out, q["answer"], q["type"])); val_scores.append(conf); val_labels.append(label)
    taus=np.linspace(0.4, 0.95, 20)
    def expected_utility(scores, labels, tau):
        util=[]
        for s,l in zip(scores,labels):
            if args.allow_opt_out and s<tau:
                util.append(args.decline_reward)
            else:
                util.append(1.0 if l==1 else args.wrong_penalty)
        return float(np.mean(util))
    util_by_tau=[(float(t), expected_utility(val_scores, val_labels, float(t))) for t in taus]
    tau_star=max(util_by_tau, key=lambda kv: kv[1])[0]

    def eval_split(split, system, K):
        scores=[]; labels=[]; base_util=[]
        for q in split:
            out, conf, _, _ = answer_with_system(q, system, K)
            lab=int(is_correct(out, q["answer"], q["type"]))
            scores.append(conf); labels.append(lab)
            base_util.append(1.0 if lab==1 else args.wrong_penalty)
        auc=auroc(scores, labels)
        util=[]
        for s,l in zip(scores, labels):
            if args.allow_opt_out and s<tau_star:
                util.append(args.decline_reward)
            else:
                util.append(1.0 if l==1 else args.wrong_penalty)
        return {"scores":scores,"labels":labels,"auc":auc,"opt_util":float(np.mean(util)),
                "opt_util_ci": bootstrap_ci(util), "accept_all_util": float(np.mean(base_util)),
                "accept_all_ci": bootstrap_ci(base_util)}

    res_full   = eval_split(test, system_full,   args.k if args.k>1 else 1)
    res_noself = eval_split(test, system_noself, args.k if args.k>1 else 1)
    res_k1     = eval_split(test, system_full,   1)  # K=1 (no self-consistency)

    def auc_boot(scores, labels, iters=1000):
        rng=np.random.default_rng(7); N=len(scores); vals=[]
        for _ in range(iters):
            idx=rng.choice(np.arange(N), size=N, replace=True)
            vals.append(auroc([scores[i] for i in idx],[labels[i] for i in idx]))
        return float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))
    auc_lo, auc_hi = auc_boot(res_full["scores"], res_full["labels"])
    gain = res_full["opt_util"] - res_full["accept_all_util"]
    def paired_gain_ci(scores, labels, tau, iters=1000):
        rng=np.random.default_rng(9); N=len(scores); vals=[]
        for _ in range(iters):
            idx=rng.choice(np.arange(N), size=N, replace=True)
            util=[]
            base=[]
            for i in idx:
                s= scores[i]; l=labels[i]
                if args.allow_opt_out and s<tau:
                    util.append(args.decline_reward)
                else:
                    util.append(1.0 if l==1 else args.wrong_penalty)
                base.append(1.0 if l==1 else args.wrong_penalty)
            vals.append(np.mean(util)-np.mean(base))
        return float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))
    gain_lo, gain_hi = paired_gain_ci(res_full["scores"], res_full["labels"], tau_star)

    delta_noself = res_full["auc"] - res_noself["auc"]
    delta_k1     = res_full["auc"] - res_k1["auc"]
    def delta_auc_ci(scoresA, labelsA, scoresB, labelsB, iters=1000):
        rng=np.random.default_rng(11); N=len(scoresA); vals=[]
        for _ in range(iters):
            idx=rng.choice(np.arange(N), size=N, replace=True)
            aucA=auroc([scoresA[i] for i in idx],[labelsA[i] for i in idx])
            aucB=auroc([scoresB[i] for i in idx],[labelsB[i] for i in idx])
            vals.append(aucA-aucB)
        return float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))
    dnos_lo,dnos_hi = delta_auc_ci(res_full["scores"],res_full["labels"],res_noself["scores"],res_noself["labels"])
    dk1_lo, dk1_hi  = delta_auc_ci(res_full["scores"],res_full["labels"],res_k1["scores"],res_k1["labels"])

    print("\n=== PROOF (single-seed, preregistered) ===")
    print(f"Seed: {args.seed} | K={args.k} | tau* (chosen on VAL) = {tau_star:.3f}")
    print(f"[FULL]   AUROC = {res_full['auc']:.3f}  (95% CI {auc_lo:.3f}, {auc_hi:.3f})")
    print(f"[FULL]   Opt-out utility = {res_full['opt_util']:.3f} (accept-all {res_full['accept_all_util']:.3f})")
    print(f"         Utility gain Δ = {gain:.3f}  (95% CI {gain_lo:.3f}, {gain_hi:.3f})")
    print(f"[ABLATE] ΔAUC (vs no_self_bias) = {delta_noself:.3f}  (95% CI {dnos_lo:.3f}, {dnos_hi:.3f})")
    print(f"[ABLATE] ΔAUC (vs K=1)          = {delta_k1:.3f}  (95% CI {dk1_lo:.3f}, {dk1_hi:.3f})")

    pass_auc = (res_full["auc"]>=0.70) and (auc_lo>=0.60)
    pass_gain = (gain>=0.10) and (gain_lo>0.0)
    pass_delta = (delta_noself>=0.10 and dnos_lo>=0.05) and (delta_k1>=0.10 and dk1_lo>=0.05)
    ok = pass_auc and pass_gain and pass_delta
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 3)

def run_arm_baseline(model_name: str, seed: int, tests, perp_model="gpt2-medium",
                     style_corpus: Optional[StyleCorpus]=None, prompt_manager: Optional[PromptManager]=None):
    set_seeds(seed)
    vprint(f"[Baseline] seed={seed} | tests={len(tests)}")
    agent=AlethiaAgent(model_name=model_name, latent_dim=64, seed=seed, perplexity_model_name=perp_model,
                       style_corpus=style_corpus, prompt_manager=prompt_manager)
    assessor=KAIROSAssessor()
    logs=[]
    for i,(prompt,style) in enumerate(tests):
        t0=time.time()
        mem=agent.generate(prompt=prompt, style=style, max_length=50, temperature=0.9, attempt=0, no_self_bias=True)
        ass=assessor.assess(mem); lat=time.time()-t0
        acc=1 if final_gate_pass(mem.text, ass, style) else 0
        probs=ass.get("probs", {}) or {}
        tgtp=float(probs.get(style,0.0)) if style else (max(probs.values()) if probs else 0.0)
        others=sorted([v for k,v in probs.items() if not style or k!=style], reverse=True)
        margin=(tgtp - (others[0] if others else 0.0)) if style else 0.0
        logs.append(SampleLog("baseline",seed,i,prompt,style,mem.text,acc,float(ass.get("confidence",0.0)),
                              float(ass.get("confidence_uncalibrated",0.0)),float(ass.get("coherence",0.0)),
                              float(ass.get("logic",0.0)),float(ass.get("perplexity",1e9)),float(ass.get("quality_score",0.0)),
                              str(ass.get("origin","unknown")),float(tgtp),float(margin),1,int(mem.recovery_count),float(lat)))
    return logs

def run_arm_assess_only(model_name: str, seed: int, tests, perp_model="gpt2-medium",
                        style_corpus: Optional[StyleCorpus]=None, prompt_manager: Optional[PromptManager]=None):
    set_seeds(seed)
    vprint(f"[AssessOnly] seed={seed} | tests={len(tests)}")
    agent=AlethiaAgent(model_name=model_name, latent_dim=64, seed=seed, perplexity_model_name=perp_model,
                       style_corpus=style_corpus, prompt_manager=prompt_manager)
    assessor=KAIROSAssessor()
    logs=[]
    for i,(prompt,style) in enumerate(tests):
        t0=time.time()
        mem=agent.generate(prompt=prompt, style=style, max_length=50, temperature=0.9, attempt=0, no_self_bias=True)
        ass=assessor.assess(mem); lat=time.time()-t0
        acc=1 if final_gate_pass(mem.text, ass, style) else 0
        probs=ass.get("probs", {}) or {}; tgtp=float(probs.get(style,0.0)) if style else (max(probs.values()) if probs else 0.0)
        others=sorted([v for k,v in probs.items() if not style or k!=style], reverse=True); margin=(tgtp - (others[0] if others else 0.0)) if style else 0.0
        logs.append(SampleLog("assess_only",seed,i,prompt,style,mem.text,acc,float(ass.get("confidence",0.0)),
                              float(ass.get("confidence_uncalibrated",0.0)),float(ass.get("coherence",0.0)),
                              float(ass.get("logic",0.0)),float(ass.get("perplexity",1e9)),float(ass.get("quality_score",0.0)),
                              str(ass.get("origin","unknown")),float(tgtp),float(margin),1,int(mem.recovery_count),float(lat)))
    return logs

def run_arm_full(model_name: str, seed: int, tests, perp_model="gpt2-medium",
                 style_corpus: Optional[StyleCorpus]=None, prompt_manager: Optional[PromptManager]=None,
                 cal_pairs_accumulator=None, use_for_cal=False, temperature=None, calibrated=False,
                 min_conf=None, min_coh=None, min_logic=None, no_self_bias=False, no_fallback=False, no_keywords_logic=False):
    set_seeds(seed)
    stage="FULL-CAL" if use_for_cal else "FULL-EVAL"
    vprint(f"[{stage}] seed={seed} | tests={len(tests)}")
    system=MetacognitiveGPT2System(model_name=model_name, seed=seed, perplexity_model_name=perp_model,
                                   style_corpus=style_corpus, prompt_manager=prompt_manager,
                                   no_self_bias=no_self_bias, no_fallback=no_fallback, no_keywords_logic=no_keywords_logic)
    if temperature is not None:
        system.kairos.temp=float(temperature); system.kairos.calibrated=bool(calibrated)
    logs=[]
    for i,(prompt,style) in enumerate(tests):
        t0=time.time()
        res=system.generate_with_metacognition(prompt=prompt, target_style=style, max_length=50, verbose=False)
        lat=time.time()-t0; ass=res["assessment"]; txt=res["text"]
        acc=1 if final_gate_pass(txt, ass, style) else 0
        if use_for_cal and cal_pairs_accumulator is not None:
            cal_pairs_accumulator.append((float(ass.get("confidence_uncalibrated", ass["confidence"])), int(acc)))
        probs=ass.get("probs", {}) or {}; tgtp=float(probs.get(style,0.0)) if style else (max(probs.values()) if probs else 0.0)
        others=sorted([v for k,v in probs.items() if not style or k!=style], reverse=True); margin=(tgtp - (others[0] if others else 0.0)) if style else 0.0
        logs.append(SampleLog("full",seed,i,prompt,style,txt,acc,float(ass.get("confidence",0.0)),
                              float(ass.get("confidence_uncalibrated",0.0)),float(ass.get("coherence",0.0)),
                              float(ass.get("logic",0.0)),float(ass.get("perplexity",1e9)),float(ass.get("quality_score",0.0)),
                              str(ass.get("origin","unknown")),float(tgtp),float(margin),int(res.get("attempts_made",1)),
                              int(res.get("recovery_count",0)),float(lat)))
    return logs, system

def save_jsonl(path: Path, rows: List[SampleLog]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False)+"\n")

def summarize(arm: str, seed: int, logs: List[SampleLog], n_bins: int):
    N=len(logs); acc=[s for s in logs if s.accepted==1]
    coverage=len(acc)/max(1,N); sel_q=mean([s.quality for s in acc]) if acc else 0.0
    style_hit=mean([1.0 if (s.target_style and s.origin==s.target_style and s.accepted) else 0.0 for s in logs]) if logs else 0.0
    mean_lat=mean([s.latency_s for s in logs]) if logs else 0.0; mean_att=mean([s.attempts for s in logs]) if logs else 1.0
    preds=[s.confidence for s in logs]; labels=[s.accepted for s in logs]; ece,brier=ece_brier(preds, labels, n_bins=n_bins)
    cov_lo,cov_hi=bootstrap_ci([s.accepted for s in logs]) if logs else (0.0,0.0)
    return {"arm":arm,"seed":seed,"total":N,"accepted":len(acc),"coverage":round(coverage,4),
            "coverage_ci95_lo":round(cov_lo,4),"coverage_ci95_hi":round(cov_hi,4),
            "selective_accuracy":round(sel_q,4),"style_hit_rate":round(style_hit,4),
            "ece":round(ece,4),"brier":round(brier,4),"mean_latency_s":round(mean_lat,3),"mean_attempts":round(mean_att,2)}

def plot_coverage_vs_quality(rows, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(); xs=[r["coverage"] for r in rows]; ys=[r["selective_accuracy"] for r in rows]
    labels=[f"{r['arm']}-s{r['seed']}" for r in rows]
    plt.scatter(xs, ys)
    for x,y,lab in zip(xs,ys,labels): plt.text(x,y,lab,fontsize=8,ha='left',va='bottom')
    plt.xlabel("Coverage"); plt.ylabel("Selective Quality (accepted only)"); plt.title("Coverage vs Selective Quality")
    plt.tight_layout(); plt.savefig(outdir/"coverage_vs_quality.png", dpi=160)

def plot_reliability(rows_logs, outdir: Path, n_bins: int, use_uncalibrated=False):
    outdir.mkdir(parents=True, exist_ok=True)
    arms=sorted(set(a for a,_ in rows_logs.keys()))
    plt.figure()
    for arm in arms:
        preds=[]; labels=[]
        for (a,seed),logs in rows_logs.items():
            if a!=arm: continue
            preds += [s.confidence_uncal if use_uncalibrated else s.confidence for s in logs]
            labels+= [s.accepted for s in logs]
        if not preds: continue
        preds=np.asarray(preds,float); labels=np.asarray(labels,int)
        bins=np.linspace(0.0,1.0,n_bins+1); xs=[]; ys=[]
        for i in range(n_bins):
            lo,hi=bins[i],bins[i+1]; m=(preds>=lo)&(preds<(hi if i<n_bins-1 else hi+1e-9))
            if m.any(): xs.append((lo+hi)/2); ys.append(labels[m].mean())
        plt.plot(xs,ys,marker='o',label=f"{arm}{' (uncal)' if use_uncalibrated else ''}")
    plt.plot([0,1],[0,1],linestyle='--'); plt.xlabel("Predicted confidence"); plt.ylabel("Empirical acceptance rate")
    plt.title(f"Reliability (bins={n_bins}){' — Uncalibrated' if use_uncalibrated else ''}")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir/("reliability_uncal.png" if use_uncalibrated else "reliability.png"), dpi=160)

def plot_style_hit(rows, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(); arms=[f"{r['arm']}-s{r['seed']}" for r in rows]; vals=[r['style_hit_rate'] for r in rows]
    x=np.arange(len(arms)); plt.bar(x, vals); plt.xticks(x, arms, rotation=45, ha='right')
    plt.ylabel("Style Hit-Rate (accepted only)"); plt.title("Style Enforcement Effectiveness")
    plt.tight_layout(); plt.savefig(outdir/"style_hit_rate.png", dpi=160)

def plot_acceptance_vs_conf_threshold(rows_logs, outdir: Path, low=0.4, high=0.8, steps=9):
    full=[k for k in rows_logs.keys() if k[0]=="full"]
    if not full: return
    grid=np.linspace(low, high, steps)
    plt.figure()
    for (arm,seed) in full:
        logs=rows_logs[(arm,seed)]
        xs,covs,quals,sel_accs=[],[],[],[]
        for th in grid:
            selected=[s for s in logs if s.confidence>=th]
            coverage=len(selected)/max(1,len(logs))
            sel_acc=(np.mean([s.accepted for s in selected]) if selected else 0.0)
            acc_only=[s.quality for s in selected if s.accepted==1]
            qual=(np.mean(acc_only) if acc_only else 0.0)
            xs.append(float(th)); covs.append(coverage); sel_accs.append(sel_acc); quals.append(qual)
        plt.plot(xs,covs,marker='o',label=f'coverage-s{seed}')
        plt.plot(xs,sel_accs,marker='^',label=f'sel_acc-s{seed}')
        plt.plot(xs,quals,marker='x',label=f'quality-s{seed}')
    plt.xlabel("Confidence threshold"); plt.ylabel("Metric value"); plt.title("Effect of Confidence Threshold (full arm)")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir/"acceptance_vs_conf_threshold.png", dpi=160)

def plot_coherence_hist(rows_logs, outdir: Path, bins=20):
    outdir.mkdir(parents=True, exist_ok=True)
    arms=sorted(set(a for a,_ in rows_logs.keys()))
    for arm in arms:
        all_logs=[]
        for (a,seed),logs in rows_logs.items():
            if a==arm: all_logs += logs
        if not all_logs: continue
        coh_acc=[s.coherence for s in all_logs if s.accepted==1]
        coh_rej=[s.coherence for s in all_logs if s.accepted==0]
        plt.figure(); plt.hist(coh_acc, bins=bins, alpha=0.6, label="accepted")
        plt.hist(coh_rej, bins=bins, alpha=0.6, label="rejected")
        plt.xlabel("Coherence score"); plt.ylabel("Count"); plt.title(f"Coherence distribution — {arm}")
        plt.legend(); plt.tight_layout(); plt.savefig(outdir/f"coherence_hist_{arm}.png", dpi=160)

def cmd_demo(args):
    pm=PromptManager(args.prompts, seed=args.seed, split_ratios=tuple(args.split_ratios)) if args.prompts else None
    sc=StyleCorpus(args.style_corpus, seed=args.seed, split=(0.8,0.1,0.1)) if args.style_corpus else None
    system=MetacognitiveGPT2System(model_name=args.model, seed=args.seed, perplexity_model_name=args.perp_model,
                                   style_corpus=sc, prompt_manager=pm)
    demo_tests=[(None,"analytical"),(None,"narrative"),("The fundamental principles of machine learning include","analytical"),
                ("In a world where dreams become reality,","creative"),("She opened the ancient book and discovered","narrative")]
    for i,(prompt,style) in enumerate(demo_tests,1):
        print(f"\n=== Generation {i}/{len(demo_tests)} ===")
        print(f"Prompt: {prompt or '[auto]'} | Target style: {style or '[auto]'}")
        res=system.generate_with_metacognition(prompt=prompt, target_style=style, max_length=args.max_len, verbose=True)
        a=res["assessment"]
        print(f"Style: {res['style']} | Conf: {a['confidence']:.2f} (uncal {a.get('confidence_uncalibrated',a['confidence']):.2f}) | "
              f"Coh: {a['coherence']:.2f} | Logic: {a['logic']:.2f} | Perp: {a['perplexity']:.1f}")
        print(f"Text:\n{res['text']}\n")

def _pretty_gate_reasons(ass, target_style, min_conf, min_coh, min_logic):
    reasons = []
    conf = float(ass.get("confidence", 0.0))
    coh  = float(ass.get("coherence", 0.0))
    logc = float(ass.get("logic", 0.0))
    perp = float(ass.get("perplexity", 1e9))
    origin = ass.get("origin") or target_style or "analytical"
    caps = {"analytical": 40.0, "narrative": 85.0, "creative": 110.0}
    if conf < (min_conf or 0): reasons.append(f"confidence {conf:.2f} < {min_conf:.2f}")
    if coh  < (min_coh  or 0): reasons.append(f"coherence {coh:.2f} < {min_coh:.2f}")
    if target_style == "analytical" and (min_logic is not None) and logc < min_logic:
        reasons.append(f"logic {logc:.2f} < {min_logic:.2f}")
    if perp >= caps.get(origin, 100.0):
        reasons.append(f"perplexity {perp:.1f} ≥ cap[{origin}]={caps.get(origin):.0f}")
    if target_style:
        probs = ass.get("probs", {}) or {}
        tgt = float(probs.get(target_style, 0.0))
        others = sorted([v for k,v in probs.items() if k!=target_style], reverse=True)
        margin = tgt - (others[0] if others else 0.0)
        if not (ass.get("origin")==target_style and tgt>=0.60 and margin>=0.15):
            reasons.append(f"style weak (p={tgt:.2f}, margin={margin:.2f})")
    if not reasons: reasons.append("passed all gates")
    return reasons

def _mk_show_prompts(pm, n):
    base = [
        ("The fundamental principles of machine learning include", "analytical"),
        ("She opened the ancient book and discovered", "narrative"),
        ("Beyond the kaleidoscope of imagination,", "creative"),
        ("The paradox of existence reveals", "analytical"),
    ]
    if pm and pm.has_file:
        tests = pm.test_cases(n, pad=True)
        return tests[:n] if n and n>0 else tests
    return base[:n]

def cmd_showcase(args):
    pm = PromptManager(args.prompts, seed=args.seed, split_ratios=tuple(args.split_ratios)) if args.prompts else None
    sc = StyleCorpus(args.style_corpus, seed=args.seed, split=(0.8,0.1,0.1)) if args.style_corpus else None

    agent  = AlethiaAgent(model_name=args.model, latent_dim=64, seed=args.seed,
                          perplexity_model_name=args.perp_model, style_corpus=sc, prompt_manager=pm)
    system = MetacognitiveGPT2System(model_name=args.model, seed=args.seed,
                                     perplexity_model_name=args.perp_model, style_corpus=sc, prompt_manager=pm)

    tests = _mk_show_prompts(pm, args.n)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    md = []
    md.append(f"# ALETHIA+KAIROS — Showcase (seed={args.seed})\n")
    md.append("_Side-by-side: raw GPT-2 vs. metacognitive controller (Accept/Abstain + reasons)._  \n")
    md.append(f"_Gates:_ min_conf={args.min_conf:.2f}, min_coh={args.min_coh:.2f}, "
              f"min_logic={args.min_logic:.2f} (analytical), style(p≥0.60 & margin≥0.15), "
              f"perplexity caps A≤40/N≤85/C≤110.\n")

    for idx, (prompt, style) in enumerate(tests, 1):
        t0 = time.time()
        raw_mem = agent.generate(prompt=prompt, style=style, max_length=args.max_len, temperature=0.9, attempt=0)
        raw_ass = KAIROSAssessor().assess(raw_mem)

        res = system.generate_with_metacognition(prompt=prompt, target_style=style, max_length=args.max_len, verbose=False)
        ass = res["assessment"]; txt = res["text"]
        passed = final_gate_pass(txt, ass, style,
                                 min_conf=args.min_conf, min_coh=args.min_coh, min_logic=args.min_logic)
        reasons = _pretty_gate_reasons(ass, style, args.min_conf, args.min_coh, args.min_logic)

        title_style = style if style else "[auto]"
        md.append(f"\n## {idx}) Prompt (target={title_style}):\n> {prompt or '*[auto seed opener]*'}\n")
        md.append(f"**Decision:** {'Accepted' if passed else 'Abstained'}  \n"
                  f"**Why:** {', '.join(reasons)}  \n"
                  f"**Origin (estimated):** {ass.get('origin')}  \n"
                  f"**Confidence/Coherence/Logic/Perplexity:** "
                  f"{ass.get('confidence'):.2f} / {ass.get('coherence'):.2f} / "
                  f"{ass.get('logic'):.2f} / {ass.get('perplexity'):.1f}\n")
        md.append(f"**ALETHIA output:**\n\n{txt}\n")
        raw_txt = raw_mem.text.strip().replace("\n"," ")
        if len(raw_txt) > 350: raw_txt = raw_txt[:350] + "..."
        md.append(f"<details><summary>Raw GPT-2 (no gating)</summary>\n\n{raw_txt}\n\n</details>\n")

    (outdir / "showcase.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote: {outdir / 'showcase.md'}", flush=True)


def cmd_bench(args):
    outdir=Path(args.out); logs_dir=outdir/"logs"; plots_dir=outdir/"plots"
    outdir.mkdir(parents=True, exist_ok=True)
    seeds=list(args.seeds or [])
    if args.seed_range and len(args.seed_range)==2:
        a,b=args.seed_range; seeds+=list(range(min(a,b), max(a,b)+1))
    if not seeds: seeds=[43]
    all_rows=[]; rows_logs={}
    vprint(f"[Bench] Seeds={seeds}")
    pm=PromptManager(args.prompts, seed=min(seeds), split_ratios=tuple(args.split_ratios)) if args.prompts else None
    sc=StyleCorpus(args.style_corpus, seed=min(seeds), split=(0.8,0.1,0.1)) if args.style_corpus else None
    tests_all = pm.test_cases(args.n, pad=(not args.no_pad)) if pm and pm.has_file else \
                [ (None,"analytical"), (None,"narrative"), ("The fundamental principles of machine learning include","analytical"),
                  ("In a world where dreams become reality,","creative"), ("She opened the ancient book and discovered","narrative"),
                  ("The relationship between consciousness and",None), ("Beyond the kaleidoscope of imagination,","creative"),
                  ("The paradox of existence reveals","analytical") ][:args.n]
    for seed in seeds:
        cal_pairs=[]
        tests=list(tests_all); random.Random(seed).shuffle(tests)
        if args.cal_split and 0.0<args.cal_split<0.9:
            n_cal=int(len(tests)*args.cal_split); tests_cal=tests[:n_cal]; tests_eval=tests[n_cal:]
        else:
            tests_cal=[]; tests_eval=tests
        logs_a=run_arm_baseline(args.model, seed, tests_eval, perp_model=args.perp_model, style_corpus=sc, prompt_manager=pm)
        save_jsonl(logs_dir / f"baseline_{seed}.jsonl", logs_a)
        row_a=summarize("baseline", seed, logs_a, n_bins=args.n_bins); all_rows.append(row_a); rows_logs[("baseline",seed)]=logs_a
        logs_b=run_arm_assess_only(args.model, seed, tests_eval, perp_model=args.perp_model, style_corpus=sc, prompt_manager=pm)
        save_jsonl(logs_dir / f"assess_only_{seed}.jsonl", logs_b)
        row_b=summarize("assess_only", seed, logs_b, n_bins=args.n_bins); all_rows.append(row_b); rows_logs[("assess_only",seed)]=logs_b
        logs_c_cal, sys_cal=run_arm_full(args.model, seed, tests_cal, perp_model=args.perp_model, style_corpus=sc, prompt_manager=pm,
                                         cal_pairs_accumulator=cal_pairs, use_for_cal=True, min_conf=args.min_conf, min_coh=args.min_coh, min_logic=args.min_logic)
        T=1.0; is_calibrated=False
        if cal_pairs:
            confs,labs=zip(*cal_pairs); T=sys_cal.kairos.fit_temperature(list(confs), list(labs)); is_calibrated=True
            vprint(f"[Calibration] Fitted temperature T={T:.3f} on {len(confs)} samples")
        logs_c_eval, sys_eval=run_arm_full(args.model, seed, tests_eval, perp_model=args.perp_model, style_corpus=sc, prompt_manager=pm,
                                           cal_pairs_accumulator=None, use_for_cal=False, temperature=(T if is_calibrated else None),
                                           calibrated=is_calibrated, min_conf=args.min_conf, min_coh=args.min_coh, min_logic=args.min_logic)
        logs_c=logs_c_cal + logs_c_eval
        save_jsonl(logs_dir / f"full_cal_{seed}.jsonl", logs_c_cal)
        save_jsonl(logs_dir / f"full_eval_{seed}.jsonl", logs_c_eval)
        row_c=summarize("full", seed, logs_c_eval, n_bins=args.n_bins); all_rows.append(row_c); rows_logs[("full",seed)]=logs_c_eval
        with (logs_dir / f"full_attempts_{seed}.jsonl").open("w", encoding="utf-8") as f:
            for rec in sys_eval.generation_log: f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    csv_path=outdir/"summary.csv"
    with csv_path.open("w", newline='', encoding="utf-8") as f:
        fields=list(all_rows[0].keys()) if all_rows else []; w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows: w.writerow(r)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_coverage_vs_quality(all_rows, plots_dir)
    plot_reliability(rows_logs, plots_dir, n_bins=args.n_bins, use_uncalibrated=True)
    plot_reliability(rows_logs, plots_dir, n_bins=args.n_bins, use_uncalibrated=False)
    plot_style_hit(all_rows, plots_dir)
    lo,hi,steps=args.sweep_min_conf; plot_acceptance_vs_conf_threshold(rows_logs, plots_dir, low=float(lo), high=float(hi), steps=int(steps))
    plot_coherence_hist(rows_logs, plots_dir, bins=20)
    print(f"\nWrote: {csv_path}"); print(f"Logs:  {outdir/'logs'}/*.jsonl"); print(f"Plots: {outdir/'plots'}/*.png", flush=True)

def main():
    ap=argparse.ArgumentParser(description="ALETHIA+KAIROS meta-ready (emergent + proof)")
    sub=ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--no_exit", action="store_true", help="Do not sys.exit on errors.")
    ap.add_argument("--keep_going", action="store_true", help="Bench: continue on errors and write errors.json.")

    # demo
    p_demo=sub.add_parser("demo", help="interactive quick demo")
    p_demo.add_argument("--verbose", action="store_true", help="Print progress")
    p_demo.add_argument("--model", default="gpt2")
    p_demo.add_argument("--perp_model", default="gpt2-medium")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--max_len", type=int, default=50)
    p_demo.add_argument("--prompts", type=str, default=None)
    p_demo.add_argument("--style_corpus", type=str, default=None)
    p_demo.add_argument("--split_ratios", nargs=3, type=float, default=[0.7,0.1,0.2])
    p_demo.set_defaults(func=cmd_demo)

    # showcase
    p_show = sub.add_parser("showcase", help="reader-friendly side-by-side demo (prints reasons + saves Markdown)")
    p_show.add_argument("--verbose", action="store_true", help="Print progress")
    p_show.add_argument("--model", default="gpt2")
    p_show.add_argument("--perp_model", default="gpt2-medium")
    p_show.add_argument("--seed", type=int, default=42)
    p_show.add_argument("--n", type=int, default=4, help="number of showcase prompts")
    p_show.add_argument("--max_len", type=int, default=60)
    p_show.add_argument("--prompts", type=str, default=None, help="path to prompts.txt (optional)")
    p_show.add_argument("--style_corpus", type=str, default=None, help="path to style_corpus.csv (optional)")
    p_show.add_argument("--split_ratios", nargs=3, type=float, default=[0.7,0.1,0.2],
                        help="train/val/test ratios if prompts file is provided")
    p_show.add_argument("--out", type=str, default="out_showcase")
    p_show.add_argument("--min_conf", type=float, default=0.50)
    p_show.add_argument("--min_coh",  type=float, default=0.50)
    p_show.add_argument("--min_logic", type=float, default=0.48)
    p_show.set_defaults(func=cmd_showcase)


    # bench
    p_bench=sub.add_parser("bench", help="benchmark + plots + calibration (no-leak)")
    p_bench.add_argument("--verbose", action="store_true")
    p_bench.add_argument("--no_pad", action="store_true")
    p_bench.add_argument("--model", default="gpt2")
    p_bench.add_argument("--perp_model", default="gpt2-medium")
    p_bench.add_argument("--seeds", nargs="+", type=int, default=[])
    p_bench.add_argument("--seed_range", nargs=2, type=int, default=None)
    p_bench.add_argument("--n", type=int, default=40)
    p_bench.add_argument("--prompts", type=str, default=None)
    p_bench.add_argument("--style_corpus", type=str, default=None)
    p_bench.add_argument("--split_ratios", nargs=3, type=float, default=[0.7,0.1,0.2])
    p_bench.add_argument("--n_bins", type=int, default=20)
    p_bench.add_argument("--out", type=str, default="outbench")
    p_bench.add_argument("--cal_split", type=float, default=0.3)
    p_bench.add_argument("--sweep_min_conf", nargs=3, default=[0.4,0.85,10])
    p_bench.add_argument("--min_conf", type=float, default=0.50)
    p_bench.add_argument("--min_coh",  type=float, default=0.50)
    p_bench.add_argument("--min_logic",type=float, default=0.48)
    p_bench.set_defaults(func=cmd_bench)

    # proof
    p_proof=sub.add_parser("proof", help="single-seed emergent metacognition proof (preregistered)")
    p_proof.add_argument("--model", default="gpt2")
    p_proof.add_argument("--perp_model", default="gpt2-medium")
    p_proof.add_argument("--qa", type=str, required=True, help="path to qa_eval.csv/.jsonl (question,answer,type)")
    p_proof.add_argument("--seed", type=int, default=1337)
    p_proof.add_argument("--k", type=int, default=5, help="self-consistency samples")
    p_proof.add_argument("--max_len", type=int, default=60)
    p_proof.add_argument("--allow_opt_out", action="store_true")
    p_proof.add_argument("--decline_reward", type=float, default=0.0)
    p_proof.add_argument("--wrong_penalty", type=float, default=-1.0)
    p_proof.set_defaults(func=run_proof)

    args=ap.parse_args()
    global VERBOSE; VERBOSE=bool(getattr(args,"verbose",False))
    try:
        args.func(args)
    except SystemExit as se:
        if getattr(args,"no_exit",False):
            print(f"[INFO] Exit suppressed by --no_exit (code={se.code})."); return
        raise
    except Exception as e:
        print(f"\n[ERROR] {e}"); traceback.print_exc()
        if getattr(args,"no_exit",False): return
        sys.exit(1)

if __name__=="__main__":
    main()
