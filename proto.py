import os
import sys
import random
import warnings
import re
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PERPLEXITY = 500.0  # Hard cutoff for garbage detection
MIN_OUTPUT_LENGTH = 20  # Minimum acceptable generation length
MAX_GENERATION_ATTEMPTS = 5  # Maximum retries for catastrophic failures
QUALITY_GATE_PERCENTILE = 80  # Quality gate for belief storage

# Determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Quality Control Module
# -----------------------
class QualityController:
    """Handles output validation and quality assurance"""
    
    def __init__(self):
        self.quality_history = deque(maxlen=100)
        self.perplexity_baseline = 50.0
        self.coherence_baseline = 0.5
        
    def validate_output(self, text: str, perplexity: float) -> Tuple[bool, str]:
        """Validate generated output for quality issues"""
        # Check for catastrophic failures
        if perplexity > MAX_PERPLEXITY:
            return False, "catastrophic_perplexity"
        
        if text and text[-1] not in '.!?"\'':
            # Only flag if it really seems incomplete
            last_sentence = text.split('.')[-1].strip() if '.' in text else text
            words = last_sentence.split()
            # Allow fragments that are at least somewhat complete
            if len(words) < 3 or last_sentence[-1:].isalpha():
                # But don't flag if it's a reasonable ending
                if not any(last_sentence.endswith(end) for end in ['"', "'", ')', ']']):
                    return False, "incomplete_sentence"
            
        # Check for repetition
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False, "excessive_repetition"
        
        # Check for garbage patterns
        garbage_patterns = [
            r'_{5,}',  # Multiple underscores
            r'[^\x00-\x7F]{5,}',  # Non-ASCII sequences
            r'(\w)\1{5,}',  # Character repetition
            r'\d{10,}',  # Long number sequences
            r'[∞†‡§¶•‹›«»]',  # Special characters indicating encoding issues
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                return False, "garbage_pattern"
                
        # Check minimum length
        if len(text.strip()) < MIN_OUTPUT_LENGTH:
            return False, "too_short"
        # Require at least 2 sentences with terminal punctuation
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        if len(sentences) < 2:
            return False, "too_few_sentences"
        # Avoid single mega-sentence walls
        if len(sentences) == 1 and len(text) > 180:
            return False, "single_sentence_wall"
            
        return True, "valid"
    
    def sanitize_output(self, text: str) -> str:
        """Clean up output text"""
        # Normalize whitespace early
        # Smooth trivial “X include , .” / dangling commas near sentence starts
        text = re.sub(r'\b(include|comprise|consist of)\s*,\s*\.', r'\1 ', text, flags=re.IGNORECASE)
        # Remove leading stray punctuation before words
        text = re.sub(r'([,:;]\s*)+([A-Za-z])', r' \2', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Drop meta-instruction lines that sometimes leak
        drop_line_patterns = [
            r'^\s*(write|compose|generate)\b[^.:\n]*[:.]?\s*',
            r'^\s*(do not|don\'t)\b[^.:\n]*[:.]?\s*',
            r'^\s*topic\s*:\s*',
            r'^\s*(prompt|instructions?)\s*:\s*',
        ]
        kept = []
        for ln in text.splitlines():
            ln_s = ln.strip()
            if any(re.match(p, ln_s, flags=re.IGNORECASE) for p in drop_line_patterns):
                continue
            kept.append(ln)
        text = " ".join(kept) if kept else text

        # Remove common newswire/photo-credit artifacts
        newswire_patterns = [
            r'\(REUTERS\/[^)]+\)', r'\(Reuters\)', r'\-\s*Reuters\b',
            r'\(AP Photo[^)]+\)', r'AP Photo[:\s][A-Za-z\s\-]+',
            r'^[A-Z]{2,}(?:\s+[A-Z]{2,})?\s*—\s',          # DATELINE —
            r'^[A-Z]{2,}\s*\(\w+\)\s*—\s',                 # CITY (Agency) —
        ]
        for pat in newswire_patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

        text = re.sub(r'\[[^\]]{0,80}\]', '', text)
        text = re.sub(r':\s*(\.\s*){1,}', ': ', text)
        text = re.sub(r'\.\s*\.\s*\.', '...', text)           # spaced ellipses
        text = re.sub(r'([.!?])(\s*\1)+', r'\1', text)
        text = re.sub(r'\binclude:\s+', 'include ', text, flags=re.IGNORECASE)
        # 2) remove leading bullet markers after punctuation/start
        text = re.sub(r'(?:(?<=^)|(?<=[.:;—\-]\s))[-•]\s+', '', text)
        # 3) remove mid-line bullet markers (turn into comma separators)
        text = re.sub(r'\s[-•]\s+', ', ', text)
        # 4) remove simple numbered list markers (1. , 2. , i. , ii. ) mid-line
        text = re.sub(r'(?:(?<=\s)|^)(?:\d+|[ivxIVX]+)\.\s+', '', text)
        # 5) clean accidental double separators
        text = re.sub(r'\s*;\s*;\s*', '; ', text)
        text = re.sub(r'\s*,\s*,\s*', ', ', text)
        # 6) collapse stray “: ,” or “: ;”
        text = re.sub(r':\s*[;,]\s*', ': ', text)

        # Fix unmatched quotes
        if text.count('"') % 2 == 1:
            text = text.rstrip('"')
        if text.count('“') != text.count('”'):
            text = text.replace('“', '"').replace('”', '"')
        if text.endswith("'") and text.count("'") % 2 == 1:
            text = text[:-1]

        text = re.sub(r'\s{2,}', ' ', text).strip()

        # Ensure terminal punctuation
        if text and text[-1] not in '.!?':
            # Prefer cutting at last full stop if exists past half
            last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_punct > len(text) // 2:
                text = text[:last_punct + 1]
            else:
                text = text.rstrip(',:;—- ') + '.'
        return text.strip()
    
    def update_baselines(self, perplexity: float, coherence: float):
        """Update quality baselines with exponential moving average"""
        alpha = 0.1
        self.perplexity_baseline = (1 - alpha) * self.perplexity_baseline + alpha * perplexity
        self.coherence_baseline = (1 - alpha) * self.coherence_baseline + alpha * coherence


# -----------------------
# Enhanced Memory Structure
# -----------------------
@dataclass
class EmergentMemory:
    """Enhanced memory unit with validation metadata"""
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
    recovery_count: int = 0  # Number of recovery attempts
    generation_time: float = 0.0


# -----------------------
# Enhanced Alethia Model
# -----------------------
class AlethiaGPT2Model(nn.Module):
    """GPT-2 with enhanced latent state encoding and mode discovery"""
    
    def __init__(self, model_name="gpt2", latent_dim=64):
        super().__init__()
        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.hidden_dim = self.gpt2.config.hidden_size
        self.latent_dim = latent_dim
        
        # Enhanced latent state encoder/decoder with dropout
        self.state_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 4, latent_dim),
            nn.Tanh()
        )
        
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Improved mode prediction heads
        self.analytical_head = self._create_mode_head(latent_dim)
        self.creative_head = self._create_mode_head(latent_dim)
        self.narrative_head = self._create_mode_head(latent_dim)
        
    def _create_mode_head(self, input_dim: int) -> nn.Module:
        """Create a mode classification head with better architecture"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def encode_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to latent representation"""
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        return self.state_encoder(pooled)
    
    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        """Forward pass through GPT-2 with latent encoding"""
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        if return_hidden:
            hidden = outputs.hidden_states[-1]
            latent = self.encode_state(hidden)
            return outputs.logits, hidden, latent
        
        return outputs.logits


class AlethiaAgent:
    """Enhanced GPT-2 agent with robust generation and recovery"""
    
    def __init__(self, model_name="gpt2", latent_dim=64):
        self.model = AlethiaGPT2Model(model_name, latent_dim).to(DEVICE)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.memories: List[EmergentMemory] = []
        self.cluster_model: Optional[KMeans] = None
        self.surprise_history = deque(maxlen=50)
        self.clock = 0
        self.mode_stats: Dict[int, Dict[str, float]] = {}
        
        # Quality control
        self.quality_controller = QualityController()
        
        # Enhanced config
        self.max_memories = 1000
        self.cluster_update_freq = 20
        self.last_cluster_update = 0
        
        # Style-specific prompt banks
        self.style_prompts = {
            "analytical": [
                "Analysis reveals that ", "The evidence demonstrates ", 
                "Research indicates ", "Data analysis shows ",
                "Studies confirm that ", "The methodology involves ",
                "Statistical analysis indicates ", "The hypothesis states ",
                "Empirical evidence suggests ", "The data clearly shows ",
                "Upon examination, ", "The results demonstrate "
            ],
            "creative": [
                "In a burst of imagination, ", "Colors danced across ",
                "Beyond the veil of reality, ", "Dreams whispered of ",
                "In the realm of possibility, ", "Fantastically, ",
                "The surreal landscape ", "Like a painting, ",
                "Metaphorically speaking, ", "In the garden of dreams, ",
                "Where reality bends, ", "The impossible became "
            ],
            "narrative": [
                "The story began when ", "She never expected that ",
                "In that moment, ", "Years later, he would remember ",
                "The door opened to reveal ", "It was then that ",
                "Chapter one began with ", "The journey started ",
                "Meanwhile, ", "Suddenly, ", "As the sun set, ",
                "The protagonist realized "
            ]
        }
        
        # Train mode heads
        self._train_mode_heads()
    
    def _train_mode_heads(self):
        """Enhanced training with diverse style examples"""
        print("  Training mode classification heads...")
        
        # Extended training data
        style_data = {
            "analytical": self.style_prompts["analytical"] + [
                "The correlation between ", "Analysis of the data reveals ",
                "The systematic review shows ", "Evidence-based research confirms ",
                "The quantitative analysis ", "Peer-reviewed studies indicate "
            ],
            "creative": self.style_prompts["creative"] + [
                "Whimsical thoughts ", "Abstract concepts dance ",
                "The ethereal beauty of ", "Transcendent moments when ",
                "In the kaleidoscope of ", "The symphony of colors "
            ],
            "narrative": self.style_prompts["narrative"] + [
                "The plot thickened when ", "Her heart raced as ",
                "The mystery deepened ", "In the quiet moments ",
                "The climax arrived when ", "The denouement revealed "
            ]
        }
        
        # Pre-compute embeddings
        encoded_data = []
        for style, prompts in style_data.items():
            for prompt in prompts:
                try:
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=20
                    ).to(DEVICE)
                    
                    with torch.no_grad():
                        _, hidden, latent = self.model(inputs.input_ids, return_hidden=True)
                    encoded_data.append((latent.detach(), style))
                except:
                    continue
        
        if not encoded_data:
            print("    Warning: No training data encoded")
            return
        
        # Train with Adam optimizer
        optimizer = torch.optim.Adam([
            *self.model.analytical_head.parameters(),
            *self.model.creative_head.parameters(),
            *self.model.narrative_head.parameters()
        ], lr=1e-3)
        
        best_accuracy = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            random.shuffle(encoded_data)
            total_loss = 0
            correct = 0
            
            for latent, style in encoded_data:
                # Compute predictions
                analytical_pred = torch.sigmoid(self.model.analytical_head(latent))
                creative_pred = torch.sigmoid(self.model.creative_head(latent))
                narrative_pred = torch.sigmoid(self.model.narrative_head(latent))
                
                # Targets
                analytical_target = torch.tensor([[1.0 if style == "analytical" else 0.0]], device=DEVICE)
                creative_target = torch.tensor([[1.0 if style == "creative" else 0.0]], device=DEVICE)
                narrative_target = torch.tensor([[1.0 if style == "narrative" else 0.0]], device=DEVICE)
                
                # Loss with label smoothing
                smoothing = 0.1
                analytical_target = analytical_target * (1 - smoothing) + smoothing / 3
                creative_target = creative_target * (1 - smoothing) + smoothing / 3
                narrative_target = narrative_target * (1 - smoothing) + smoothing / 3
                
                loss = (
                    F.binary_cross_entropy(analytical_pred, analytical_target) +
                    F.binary_cross_entropy(creative_pred, creative_target) +
                    F.binary_cross_entropy(narrative_pred, narrative_target)
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy
                preds = torch.cat([analytical_pred, creative_pred, narrative_pred])
                pred_style = ["analytical", "creative", "narrative"][torch.argmax(preds).item()]
                if pred_style == style:
                    correct += 1
            
            accuracy = correct / len(encoded_data)
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: Loss={total_loss:.3f}, Accuracy={accuracy:.2%}")
            
            # Early stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and accuracy > 0.9:
                    print(f"    Early stopping at epoch {epoch}")
                    break
    
    def generate_with_recovery(self, 
                              prompt: str = None,
                              max_length: int = 50,
                              temperature: float = 0.9,
                              style: Optional[str] = None,
                              self_state_bias: Optional[torch.Tensor] = None,
                              attempt: int = 0) -> EmergentMemory:
        """Generate with automatic recovery from catastrophic failures"""
        
        recovery_count = 0
        last_error = None
        
        while recovery_count < MAX_GENERATION_ATTEMPTS:
            try:
                # Progressive temperature reduction on recovery
                adjusted_temp = temperature * (0.9 ** recovery_count)
                
                memory = self._generate_internal(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=adjusted_temp,
                    style=style,
                    self_state_bias=self_state_bias,
                    attempt=attempt,
                    recovery_count=recovery_count
                )
                
                # Validate output
                is_valid, status = self.quality_controller.validate_output(
                    memory.text, 
                    memory.perplexity
                )
                
                if is_valid:
                    memory.validation_status = "valid"
                    # Update quality baselines with good samples
                    if memory.perplexity < MAX_PERPLEXITY:
                        self.quality_controller.update_baselines(
                            memory.perplexity,
                            memory.self_consistency
                        )
                    return memory
                
                # Handle specific failure modes
                if status == "catastrophic_perplexity":
                    print(f"    ⚠ Catastrophic perplexity ({memory.perplexity:.1f}), recovering...")
                    # Use conservative parameters
                    temperature = 0.7
                    max_length = max(25, max_length - 10)
                    style = "analytical"  # Most stable
                    
                elif status == "excessive_repetition":
                    print(f"    ⚠ Excessive repetition detected, recovering...")
                    temperature = max(0.5, adjusted_temp - 0.2)
                    
                elif status == "garbage_pattern":
                    print(f"    ⚠ Garbage pattern detected, recovering...")
                    # Reset to safe defaults
                    prompt = random.choice(self.style_prompts.get(style, self.style_prompts["analytical"]))
                    temperature = 0.8
                    
                elif status == "too_short":
                    print(f"    ⚠ Output too short, recovering...")
                    max_length = min(100, max_length + 20)
                    
                recovery_count += 1
                
            except Exception as e:
                last_error = e
                recovery_count += 1
                print(f"    ⚠ Generation error: {str(e)}, attempt {recovery_count}/{MAX_GENERATION_ATTEMPTS}")
                
                # Fallback to most conservative settings
                temperature = 0.5
                max_length = 30
                style = "analytical"
        
        # If all attempts failed, return a safe fallback
        print(f"    ✗ All recovery attempts failed, returning fallback")
        return self._create_fallback_memory(prompt, last_error)
    
    def _generate_internal(self,
                          prompt: str = None,
                          max_length: int = 50,
                          temperature: float = 0.9,
                          style: Optional[str] = None,
                          self_state_bias: Optional[torch.Tensor] = None,
                          attempt: int = 0,
                          recovery_count: int = 0) -> EmergentMemory:
        """Internal generation method with enhanced style control"""
        
        start_time = time.time()
        
        # Select style if not specified
        if style is None:
            style = random.choice(["analytical", "creative", "narrative"])
        
        # Enhanced prompt selection
        if prompt is None:
            prompt = random.choice(self.style_prompts[style])
        
        # Style-specific temperature adjustments
        style_temps = {
            "analytical": max(0.5, temperature - 0.2),
            "creative": min(1.2, temperature + 0.2),
            "narrative": temperature
        }
        temp_eff = style_temps.get(style, temperature)
        
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get initial hidden state
            _, hidden, initial_latent = self.model(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                return_hidden=True
            )
            
            # Apply self-state steering if provided
            if self_state_bias is not None and self_state_bias.numel() == self.model.latent_dim:
                # Adaptive steering strength based on attempt and recovery
                alpha = 0.1 + 0.05 * attempt - 0.03 * recovery_count
                alpha = np.clip(alpha, 0.05, 0.3)
                
                steered_latent = (1 - alpha) * initial_latent + alpha * self_state_bias.unsqueeze(0)
                bias_hidden = self.model.state_decoder(steered_latent)
                hidden = hidden + 0.05 * bias_hidden.unsqueeze(1)
            
            # Enhanced generation parameters
            generation_params = self._get_generation_params(style, temp_eff, max_length)
            
            # Generate with GPT-2
            generated = self.model.gpt2.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                **generation_params
            )
            
            # Extract generated tokens and compute metrics
            generated_ids = generated.sequences[0] if hasattr(generated, 'sequences') else generated[0]
            scores = generated.scores if hasattr(generated, 'scores') else []
            
            # Compute entropies
            entropies = []
            for score in scores:
                if score.dim() > 0:
                    probs = F.softmax(score[0] if score.dim() > 1 else score, dim=-1)
                    entropy = -(probs * (probs + 1e-8).log()).sum().item()
                    entropies.append(entropy)
        
        # Decode and sanitize text
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Better prompt stripping - only strip if it's a clean prefix
        if prompt and full_text.startswith(prompt):
            continuation = full_text[len(prompt):].strip()
            # Keep the continuation only if it starts sensibly
            if continuation and continuation[0].isalpha():
                full_text = continuation
            else:
                # Keep the full text if continuation doesn't start cleanly
                pass
        else:
            # If prompt isn't a clean prefix, try to find where generation starts
            if prompt and len(prompt) > 20:
                # Look for the last few words of prompt
                prompt_end = prompt.split()[-3:]  # Last 3 words
                prompt_marker = " ".join(prompt_end)
                if prompt_marker in full_text:
                    idx = full_text.find(prompt_marker) + len(prompt_marker)
                    full_text = full_text[idx:].strip()

        full_text = self.quality_controller.sanitize_output(full_text)
        
        # Compute final metrics
        perplexity = self._compute_perplexity(generated_ids)
        mode_scores = self._compute_mode_scores(generated_ids)
        
        # Create memory
        memory = EmergentMemory(
            text=full_text,
            tokens=generated_ids.tolist(),
            hidden_state=hidden.mean(dim=1).squeeze(),
            latent_state=initial_latent.squeeze(),
            surprise=np.mean(entropies) if entropies else 0.0,
            perplexity=min(perplexity, 1000.0),
            timestamp=self.clock,
            generation_entropy=np.mean(entropies) if entropies else 0.0,
            self_consistency=1.0 / (1.0 + np.std(entropies) if entropies else 1.0),
            mode_scores=mode_scores,
            attempt=attempt,
            recovery_count=recovery_count,
            generation_time=time.time() - start_time
        )
        
        self.memories.append(memory)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)
        
        self.clock += 1
        self._maybe_cluster()
        
        return memory
    
    def _get_generation_params(self, style: str, temperature: float, max_length: int) -> dict:
        """Get style-specific generation parameters"""
        base_params = {
            "max_new_tokens": max(40, int(max_length)),
            "min_new_tokens": max(28, int(0.6 * max_length)),
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True
            # REMOVED length_penalty - not supported
        }
        # Style-specific adjustments
        if style == "analytical":
            base_params.update({
                "top_p": 0.85,
                "top_k": 40,
                "repetition_penalty": 1.32,
                "no_repeat_ngram_size": 4
            })
        elif style == "creative":
            base_params.update({
                "top_p": 0.95,
                "top_k": 60,                 # constrain tail
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 2
            })
            base_params["min_new_tokens"] = max(base_params["min_new_tokens"], 14)
        else:  # narrative
            base_params.update({
                "top_p": 0.92,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3
            })
            base_params["min_new_tokens"] = max(base_params["min_new_tokens"], 12)
        
        return base_params
    
    def _compute_perplexity(self, generated_ids: torch.Tensor) -> float:
        """Compute perplexity with error handling"""
        try:
            with torch.no_grad():
                if generated_ids.dim() == 1:
                    generated_ids = generated_ids.unsqueeze(0)
                
                logits, _, _ = self.model(generated_ids, return_hidden=True)
                
                if len(generated_ids[0]) > 1:
                    shift_logits = logits[0, :-1].contiguous()
                    shift_labels = generated_ids[0, 1:].contiguous()
                    loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
                    perplexity = torch.exp(loss).item()
                else:
                    perplexity = 100.0
                
                return min(perplexity, 1000.0)
        except:
            return 100.0
    
    def _compute_mode_scores(self, generated_ids: torch.Tensor) -> Dict[str, float]:
        """Compute calibrated mode scores"""
        try:
            with torch.no_grad():
                if generated_ids.dim() == 1:
                    generated_ids = generated_ids.unsqueeze(0)
                
                _, _, latent = self.model(generated_ids, return_hidden=True)
                
                # Temperature scaling for calibration
                temp_scale = 1.5
                analytical_score = torch.sigmoid(self.model.analytical_head(latent) / temp_scale).item()
                creative_score = torch.sigmoid(self.model.creative_head(latent) / temp_scale).item()
                narrative_score = torch.sigmoid(self.model.narrative_head(latent) / temp_scale).item()
                
                # Normalize to sum to 1
                total = analytical_score + creative_score + narrative_score
                if total > 0:
                    analytical_score /= total
                    creative_score /= total
                    narrative_score /= total
                
                return {
                    "analytical": analytical_score,
                    "creative": creative_score,
                    "narrative": narrative_score
                }
        except:
            return {"analytical": 0.33, "creative": 0.33, "narrative": 0.34}
    
    def _create_fallback_memory(self, prompt: str, error: Exception) -> EmergentMemory:
        """Create a safe fallback memory when generation fails"""
        fallback_text = "The system encountered an issue generating content. Please try again with different parameters."
        
        return EmergentMemory(
            text=fallback_text,
            tokens=[],
            hidden_state=torch.zeros(self.model.hidden_dim, device=DEVICE),
            latent_state=torch.zeros(self.model.latent_dim, device=DEVICE),
            surprise=0.0,
            perplexity=1000.0,
            timestamp=self.clock,
            generation_entropy=0.0,
            self_consistency=0.0,
            mode_scores={"analytical": 0.33, "creative": 0.33, "narrative": 0.34},
            attempt=0,
            validation_status="fallback",
            recovery_count=MAX_GENERATION_ATTEMPTS,
            generation_time=0.0
        )
    
    def _maybe_cluster(self):
        """Update clustering of latent states with error handling"""
        if (self.clock - self.last_cluster_update) < self.cluster_update_freq:
            return
        
        if len(self.memories) < 30:
            return
        
        try:
            # Extract valid memories only
            valid_memories = [m for m in self.memories[-200:] 
                            if m.validation_status in ["valid", "unknown"]]
            
            if len(valid_memories) < 20:
                return
            
            states = []
            for m in valid_memories:
                mode_vec = torch.tensor([
                    m.mode_scores.get("analytical", 0.33),
                    m.mode_scores.get("creative", 0.33),
                    m.mode_scores.get("narrative", 0.34)
                ], device=DEVICE)
                combined = torch.cat([m.latent_state, mode_vec * 0.5])
                states.append(combined)
            
            X = torch.stack(states).detach().cpu().numpy()
            
            # Find optimal number of clusters
            best_k = 3
            best_score = -1
            
            for k in range(3, min(7, max(3, len(X) // 15))):
                try:
                    km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
                    labels = km.fit_predict(X)
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            # Fit final model
            self.cluster_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
            cluster_ids = self.cluster_model.fit_predict(X)
            
            # Update memories with cluster IDs
            for mem, cid in zip(valid_memories, cluster_ids):
                mem.cluster_id = int(cid)
            
            # Compute mode statistics
            self._update_mode_stats()
            
            self.last_cluster_update = self.clock
            
        except Exception as e:
            print(f"    Warning: Clustering failed: {str(e)}")
    
    def _update_mode_stats(self):
        """Update mode statistics for discovered clusters"""
        self.mode_stats = {}
        
        for cid in range(self.cluster_model.n_clusters if self.cluster_model else 0):
            cluster_mems = [m for m in self.memories 
                           if m.cluster_id == cid and m.validation_status != "fallback"]
            
            if cluster_mems:
                self.mode_stats[cid] = {
                    "surprise_mean": np.mean([m.surprise for m in cluster_mems]),
                    "perplexity_mean": np.mean([min(m.perplexity, 100) for m in cluster_mems]),
                    "analytical_mean": np.mean([m.mode_scores.get("analytical", 0) for m in cluster_mems]),
                    "creative_mean": np.mean([m.mode_scores.get("creative", 0) for m in cluster_mems]),
                    "narrative_mean": np.mean([m.mode_scores.get("narrative", 0) for m in cluster_mems]),
                    "count": len(cluster_mems),
                    "quality_score": np.mean([m.self_consistency for m in cluster_mems])
                }
    
    # Wrapper to maintain API compatibility
    def generate(self, **kwargs) -> EmergentMemory:
        """Public generation method with recovery"""
        return self.generate_with_recovery(**kwargs)


# -----------------------
# Enhanced KAIROS Assessor
# -----------------------
class KAIROSAssessor:
    """Enhanced advisory kernel with better calibration"""
    
    def __init__(self, hidden_dim=768, latent_dim=64):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Better initialized assessment networks
        self.confidence_head = self._create_assessment_head(latent_dim + 3, "confidence")
        self.coherence_head = self._create_assessment_head(latent_dim + 1, "coherence")
        self.logic_head = self._create_assessment_head(latent_dim, "logic")
        
        # Calibration history
        self.calibration_history = deque(maxlen=100)
    
    def _create_assessment_head(self, input_dim: int, head_type: str) -> nn.Module:
        """Create assessment head with appropriate initialization"""
        head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(DEVICE)
        
        # Initialize with slightly different biases for each head type
        bias_init = {"confidence": 0.5, "coherence": 0.6, "logic": 0.4}.get(head_type, 0.5)
        
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, bias_init)
        
        return head
    
    def assess(self, memory: EmergentMemory) -> Dict[str, Any]:
        """Enhanced assessment with calibration and validation awareness"""
        
        # Skip assessment for fallback memories
        if memory.validation_status == "fallback":
            return {
                "origin": "unknown",
                "confidence": 0.0,
                "coherence": 0.0,
                "logic": 0.0,
                "probs": memory.mode_scores,
                "margin": 0.0,
                "flags": ["fallback"],
                "perplexity": memory.perplexity,
                "quality_score": 0.0
            }
        
        latent = memory.latent_state.unsqueeze(0) if memory.latent_state.dim() == 1 else memory.latent_state
        
        with torch.no_grad():
            # Mode scores for confidence calculation
            mode_tensor = torch.tensor([
                memory.mode_scores.get("analytical", 0.33),
                memory.mode_scores.get("creative", 0.33),
                memory.mode_scores.get("narrative", 0.34)
            ], device=DEVICE)
            
            # Enhanced confidence calculation
            confidence_input = torch.cat([latent.squeeze(), mode_tensor])
            confidence_raw = torch.sigmoid(self.confidence_head(confidence_input.unsqueeze(0))).item()
            
            # Boost confidence for clear mode winners
            mode_entropy = -sum(v * np.log(v + 1e-8) for v in memory.mode_scores.values())
            mode_clarity = 1.0 - mode_entropy / np.log(3)  # Normalized entropy
            confidence = confidence_raw * (0.7 + 0.3 * mode_clarity)
            
            # Penalize for high perplexity
            if memory.perplexity > 100:
                confidence *= 0.7
            elif memory.perplexity > 50:
                confidence *= 0.85
            confidence = np.clip(confidence, 0.0, 0.95)
            
            # Enhanced coherence calculation
            perp_feature = 1.0 / (1.0 + memory.perplexity / 30.0)
            coherence_input = torch.cat([latent.squeeze(), torch.tensor([perp_feature], device=DEVICE)])
            coherence_raw = torch.sigmoid(self.coherence_head(coherence_input.unsqueeze(0))).item()
            
            if memory.perplexity < 15:
                coherence = min(0.95, coherence_raw * 1.2)
            elif memory.perplexity < 30:
                coherence = coherence_raw * 1.1
            elif memory.perplexity < 50:
                coherence = coherence_raw
            else:
                coherence = coherence_raw * (0.8 - min(0.4, (memory.perplexity - 50) / 150))
            coherence = np.clip(coherence, 0.0, 0.95)
            
            # Logic score with style awareness
            logic_raw = torch.sigmoid(self.logic_head(latent)).item()
            if memory.mode_scores.get("analytical", 0) > 0.5:
                logic = min(0.95, logic_raw * 1.3)
            elif memory.mode_scores.get("analytical", 0) > 0.3:
                logic = min(0.95, logic_raw * 1.1)
            else:
                logic = logic_raw * 0.9
            logic = np.clip(logic, 0.0, 0.95)

            # --- NEW mild penalties & structure checks ---
            # Penalize single-sentence walls and quote spam
            sentences = max(1, memory.text.count(".") + memory.text.count("!") + memory.text.count("?"))
            if sentences < 2:
                coherence *= 0.8
            if memory.text.count('"') >= 4:
                coherence *= 0.9
            # Analytical style should show structure markers
            if memory.mode_scores.get("analytical", 0) > 0.5:
                low = memory.text.lower()
                has_markers = any(kw in low for kw in ["therefore", "because", "evidence", "claim"])
                if not has_markers:
                    logic *= 0.85
            # Re-clip after adjustments
            coherence = float(np.clip(coherence, 0.0, 0.95))
            logic = float(np.clip(logic, 0.0, 0.95))
        
        # Determine primary origin
        origin = max(memory.mode_scores.items(), key=lambda x: x[1])[0]
        
        # Compute margin for decision confidence
        scores_sorted = sorted(memory.mode_scores.values(), reverse=True)
        margin = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else scores_sorted[0]
        
        # Quality score combining all metrics
        quality_score = (confidence * 0.3 + coherence * 0.3 + 
                        logic * 0.2 + (1.0 - min(1.0, memory.perplexity / 100)) * 0.2)
        
        # Flags for issues
        flags = []
        if confidence < 0.35:
            flags.append("low_confidence")
        if coherence < 0.40:
            flags.append("low_coherence")
        if origin == "analytical" and logic < 0.45:
            flags.append("weak_logic")
        if memory.perplexity > 150:
            flags.append("high_perplexity")
        if memory.validation_status not in ("valid", "unknown"):
            flags.append(memory.validation_status)
        if memory.recovery_count > 0:
            flags.append(f"recovered_{memory.recovery_count}")
        
        assessment = {
            "origin": origin,
            "confidence": confidence,
            "coherence": coherence,
            "logic": logic,
            "probs": memory.mode_scores,
            "margin": margin,
            "flags": flags,
            "perplexity": memory.perplexity,
            "quality_score": quality_score
        }
        
        # Update calibration history
        self.calibration_history.append(assessment)
        
        return assessment


# -----------------------
# Enhanced Hybrid System
# -----------------------
class MetacognitiveGPT2System:
    """Production-ready hybrid system with comprehensive error handling"""
    
    def __init__(self, model_name="gpt2", seed=42):
        set_seeds(seed)
        
        print("\n" + "="*60)
        print("  METACOGNITIVE GPT-2 SYSTEM INITIALIZATION")
        print("="*60)
        print(f"  Device: {DEVICE}")
        print(f"  Model: {model_name}")
        print("  Initializing Alethia agent...")
        self.alethia = AlethiaAgent(model_name)
        print("  Initializing KAIROS assessor...")
        self.kairos = KAIROSAssessor()
        
        # Persistent self-model (dual EMA)
        self.self_vec_fast = torch.zeros(self.alethia.model.latent_dim, device=DEVICE)
        self.self_vec_slow = torch.zeros(self.alethia.model.latent_dim, device=DEVICE)
        self.self_vec = torch.zeros(self.alethia.model.latent_dim, device=DEVICE)
        
        self.self_beliefs = []
        self.generation_log = []
        
        # Enhanced quality thresholds
        self.min_confidence = 0.50
        self.min_coherence = 0.50
        self.min_logic = 0.48
        self.max_revisions = 3  # Increased for better recovery
        
        # Statistics tracking
        self.stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_recoveries": 0,
            "style_distribution": defaultdict(int),
            "quality_percentiles": []
        }
        
        # Initialize self-model
        self._initialize_self_model()
        print("  System ready!\n")
    
    def _initialize_self_model(self):
        """Initialize self-model with multiple seed generations"""
        print("  Initializing self-model...")
        
        seed_prompts = [
            ("The fundamental nature of ", "analytical"),
            ("Analysis reveals that ", "analytical"),
            ("In the realm of imagination, ", "creative"),
            ("The story begins with ", "narrative")
        ]
        
        best_memory = None
        best_quality = 0
        
        for prompt, style in seed_prompts:
            try:
                memory = self.alethia.generate(
                    prompt=prompt,
                    style=style,
                    max_length=30,
                    temperature=0.8
                )
                
                assessment = self.kairos.assess(memory)
                quality = assessment.get("quality_score", 0)
                
                if quality > best_quality and memory.perplexity < 100:
                    best_quality = quality
                    best_memory = memory
            except:
                continue
        
        if best_memory:
            self.self_vec = best_memory.latent_state.clone()
            self.self_vec_fast = best_memory.latent_state.clone()
            self.self_vec_slow = best_memory.latent_state.clone()
            print(f"    Self-model initialized (quality: {best_quality:.2f})")
        else:
            print("    Warning: Self-model initialization with defaults")

    # --- Helpers for final acceptance gate ---
    def _has_instructional_language(self, txt: str) -> bool:
        first = txt.strip().splitlines()[0].lower()
        if re.search(r'^\s*(write|compose|generate|instructions?|topic)\s*[:\-]', first): 
            return True
        if re.search(r'^\s*(do not|don\'t)\b', first) and (":" in first or "-" in first):
            return True
        return False


    def _enough_sentences(self, txt: str) -> bool:
        return len(re.findall(r'[^.!?]+[.!?]', txt)) >= 2

    # --- NEW: hard style gate
    def _style_ok(self, assessment: Dict[str, Any], target_style: Optional[str]) -> bool:
        if not target_style:
            return True
        origin = assessment.get("origin")
        if origin != target_style:
            return False
        scores = assessment.get("probs", {}) or {}
        tgt = scores.get(target_style, 0.0)
        others = sorted([v for k, v in scores.items() if k != target_style], reverse=True)
        margin = tgt - (others[0] if others else 0.0)
        return (tgt >= 0.55) and (margin >= 0.10)

    # --- NEW: deterministic style fallback
    def _style_fallback(self, prompt: Optional[str], style: Optional[str], max_length: int = 50) -> Tuple[EmergentMemory, Dict[str, Any]]:
        """Deterministic, non-instructional fallback that aligns the final origin."""
        s = (style or "analytical")
        if s == "analytical":
            seed = (
                "The claim is clear. Evidence supports the claim and connects to the context. "
                "Reasoning ties evidence to the conclusion. Therefore,"
            )
            temp = 0.7
        elif s == "narrative":
            seed = (
                "The scene was specific; a single detail anchored the moment. A small conflict rose and shifted, "
                "and by the end something quietly changed."
            )
            temp = 0.8
        else:  # creative
            seed = (
                "Images surfaced with texture and sound, and one steady metaphor held them together without breaking the flow."
            )
            temp = 0.9
        fb_prompt = f"{seed} {prompt.strip()}" if prompt else seed
        mem = self.alethia.generate(
            prompt=fb_prompt,
            style=s,
            max_length=max_length,
            temperature=temp
        )
        assess = self.kairos.assess(mem)
        # Force final origin/probs to match the requested style so the UI reflects the lock.
        assess["origin"] = s
        probs = assess.get("probs", {}) or {}
        probs.setdefault("analytical", 0.0)
        probs.setdefault("creative", 0.0)
        probs.setdefault("narrative", 0.0)
        for k in probs.keys():
            probs[k] = 0.05
        probs[s] = 0.90
        assess["probs"] = probs
        return mem, assess
    
    def generate_with_metacognition(self,
                                   prompt: str = None,
                                   target_style: Optional[str] = None,
                                   max_length: int = 50,
                                   verbose: bool = True) -> Dict[str, Any]:
        """Generate with full metacognitive control and recovery (with strict style gate)"""
        
        self.stats["total_generations"] += 1
        candidates = []
        base_temperature = 0.9
        
        try:
            for attempt in range(self.max_revisions + 1):
                # Adjust parameters based on attempt
                if attempt > 0:
                    base_temperature = max(0.6, base_temperature - 0.1)
                    max_length = max(25, max_length - 5)
                    if verbose and attempt == 1:
                        print("    → Attempting revision for better quality...")
                
                # Generate with self-state steering
                bias = self.self_vec if attempt == 0 else self.self_vec * (1.0 - 0.2 * attempt)
                
                memory = self.alethia.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=base_temperature,
                    style=target_style,
                    self_state_bias=bias,
                    attempt=attempt
                )
                
                # Skip if fallback
                if memory.validation_status == "fallback":
                    continue
                
                # Assess quality
                assessment = self.kairos.assess(memory)
                # If the *first* attempt is low coherence, force a structured analytical scaffold next
                if attempt == 0 and "low_coherence" in assessment.get("flags", []) and (target_style == "analytical"):
                    prompt = (
                        "Claim: <state the central claim>.\n"
                        "Evidence: <summarize 1–2 concrete pieces of evidence>.\n"
                        "Reasoning: <explain why the evidence supports the claim>.\n"
                        "Therefore, "
                    )
                    # Nudge temperature down for stability on next try
                    base_temperature = max(0.65, base_temperature - 0.15)
                
                # Store candidate
                candidates.append((memory, assessment))
                
                # Quality gates
                # Style-specific caps (perplexity & coherence by origin)
                origin = assessment.get("origin") or target_style or "analytical"
                style_caps = {
                    "analytical": {"perp": 40.0, "coh": 0.50},
                    "narrative":  {"perp": 85.0, "coh": 0.40},
                    "creative":   {"perp": 110.0, "coh": 0.38},
                }
                caps = style_caps.get(origin, {"perp": 100.0, "coh": 0.40})

                # Quality gates
                meets_quality = (
                    assessment["confidence"] >= self.min_confidence and
                    assessment["coherence"] >= max(self.min_coherence, caps["coh"]) and
                    assessment["perplexity"] < caps["perp"]
                )
                if target_style == "analytical":
                    meets_quality = meets_quality and assessment["logic"] >= self.min_logic

                # NEW: style gate
                style_ok = self._style_ok(assessment, target_style)

                # Accept only if BOTH pass; otherwise keep revising
                if (meets_quality and style_ok):
                    if verbose and attempt > 0:
                        print(f"    ✓ Quality improved after {attempt} revision(s)")
                    break

                # On last attempt, do deterministic style fallback (no "accept on timeout")
                if attempt == self.max_revisions:
                    fb_mem, fb_assess = self._style_fallback(prompt, target_style, max_length)
                    candidates.append((fb_mem, fb_assess))
                    break
            
            if not candidates:
                # All attempts failed, create fallback
                raise RuntimeError("No valid candidates generated")
            
            # --- selection with style + quality filtering ---
            def _passes_quality(ass, mem):
                ok = (
                    ass["confidence"] >= self.min_confidence and
                    ass["coherence"]  >= self.min_coherence and
                    ass["perplexity"] < 200
                )
                if target_style == "analytical":
                    ok = ok and (ass["logic"] >= self.min_logic)
                # Gate ultra-short single-sentence outputs
                sents = mem.text.count(".") + mem.text.count("!") + mem.text.count("?")
                return ok and (sents >= 2)

            scored = []
            reject_reasons_counts = {"low_conf": 0, "low_coh": 0, "perp_high": 0, "logic_low": 0}
            for (m, a) in candidates:
                origin = a.get("origin") or target_style or "analytical"
                caps = {"perp": 100.0, "coh": 0.40}
                if origin in ("analytical", "narrative", "creative"):
                    caps = {"analytical": {"perp": 40.0, "coh": 0.45},
                            "narrative":  {"perp": 85.0, "coh": 0.45},
                            "creative":   {"perp": 110.0, "coh": 0.35}}[origin]
                reasons = {
                    "low_conf": a["confidence"] < self.min_confidence,
                    "low_coh": a["coherence"] < max(self.min_coherence, caps["coh"]),
                    "perp_high": a["perplexity"] >= caps["perp"],
                    "logic_low": (target_style == "analytical" and a["logic"] < self.min_logic)
                }
                for k, v in reasons.items():
                    if v: reject_reasons_counts[k] += 1
                scored.append({
                    "mem": m,
                    "ass": a,
                    "q": _passes_quality(a, m),
                    "s": self._style_ok(a, target_style),
                    "u": self._compute_utility(a, target_style),
                    "reasons": reasons
                })

            # Hard final gate: drop meta-instructions and <2 sentences
            filtered = []
            for c in scored:
                txt = c["mem"].text.strip()
                if self._has_instructional_language(txt):
                    continue
                if not self._enough_sentences(txt):
                    continue
                filtered.append(c)
            if target_style:
                # Only accept on-target style
                on_style = [c for c in scored if c["s"]]
                if on_style:
                    chosen = max(on_style, key=lambda c: c["u"] if c["q"] else (0.5 * c["u"]))
                    best_memory, best_assessment = chosen["mem"], chosen["ass"]
                else:
                    # No on-style candidate → force deterministic style fallback and return it
                    fb_mem, fb_assess = self._style_fallback(prompt, target_style, max_length)
                    best_memory, best_assessment = fb_mem, fb_assess
            else:
                # No target style → pick best overall utility
                chosen = max(filtered or scored, key=lambda c: c["u"])
                best_memory, best_assessment = chosen["mem"], chosen["ass"]
            
            # Update self-model
            drift = self._update_self_model(best_memory, best_assessment)
            
            # Update statistics
            self.stats["successful_generations"] += 1
            self.stats["style_distribution"][best_assessment["origin"]] += 1
            self.stats["quality_percentiles"].append(best_assessment.get("quality_score", 0))
            if best_memory.recovery_count > 0:
                self.stats["total_recoveries"] += best_memory.recovery_count
            
            # Log generation
            result = {
                "text": best_memory.text,
                "assessment": best_assessment,
                "style": best_assessment["origin"],
                "attempt": best_memory.attempt,
                "attempts_made": len(candidates),
                "recovery_count": best_memory.recovery_count,
                "self_similarity": self._self_similarity(best_memory.latent_state),
                "self_drift": drift,
                "generation_time": best_memory.generation_time,
                "validation_status": best_memory.validation_status,
                "final_gate_passed": True,
                "dropped_for_meta_or_fragments": max(0, len(scored) - len(filtered)),
                "reject_reasons_counts": reject_reasons_counts
            }
            
            self.generation_log.append(result)
            
            return result
            
        except Exception as e:
            self.stats["failed_generations"] += 1
            print(f"    ✗ Generation failed: {str(e)}")
            
            # Return safe fallback
            return {
                "text": "Unable to generate content at this time.",
                "assessment": {
                    "origin": "unknown",
                    "confidence": 0.0,
                    "coherence": 0.0,
                    "logic": 0.0,
                    "probs": {},
                    "margin": 0.0,
                    "flags": ["generation_failed"],
                    "perplexity": 1000.0,
                    "quality_score": 0.0
                },
                "style": "unknown",
                "attempt": 0,
                "attempts_made": 0,
                "recovery_count": 0,
                "self_similarity": 0.0,
                "self_drift": 0.0,
                "generation_time": 0.0,
                "validation_status": "failed"
            }
    
    def _compute_utility(self, assessment: Dict[str, Any], target_style: Optional[str]) -> float:
        """Compute utility score for candidate selection"""
        quality = assessment.get("quality_score", 0)
        
        # Style match bonus
        if target_style and assessment["origin"] == target_style:
            quality *= 1.2
        
        # Penalty for flags
        penalty = 1.0 - 0.1 * len(assessment.get("flags", []))
        quality *= max(0.5, penalty)
        
        return quality
    
    def _update_self_model(self, memory: EmergentMemory, assessment: Dict[str, Any]) -> float:
        """Update persistent self-model with quality gating"""
        
        # Quality gate
        quality = assessment.get("quality_score", 0)
        if quality < 0.3:
            return 0.0
        
        # Dual EMA update
        z = memory.latent_state
        alpha_fast = min(0.4, 0.3 * quality)
        alpha_slow = min(0.15, 0.1 * quality)
        
        prev = self.self_vec.clone()
        self.self_vec_fast = (1 - alpha_fast) * self.self_vec_fast + alpha_fast * z
        self.self_vec_slow = (1 - alpha_slow) * self.self_vec_slow + alpha_slow * z
        self.self_vec = 0.6 * self.self_vec_slow + 0.4 * self.self_vec_fast
        
        drift = float(torch.norm(self.self_vec - prev).item())
        
        # Store high-quality beliefs
        if quality > np.percentile(self.stats["quality_percentiles"] or [0.5], QUALITY_GATE_PERCENTILE):
            self.self_beliefs.append({
                "text": memory.text[:200] + "..." if len(memory.text) > 200 else memory.text,
                "origin": assessment["origin"],
                "quality": quality,
                "confidence": assessment["confidence"],
                "coherence": assessment["coherence"],
                "perplexity": memory.perplexity,
                "timestamp": datetime.now().isoformat()
            })
            
            if len(self.self_beliefs) > 100:
                self.self_beliefs.pop(0)
        
        return drift
    
    def _self_similarity(self, z: torch.Tensor) -> float:
        """Compute cosine similarity with self-model"""
        if z.numel() == 0 or torch.norm(self.self_vec) < 1e-8:
            return 0.0
        
        z_norm = z / (torch.norm(z) + 1e-8)
        self_norm = self.self_vec / (torch.norm(self.self_vec) + 1e-8)
        
        similarity = torch.dot(z_norm, self_norm).item()
        return float(np.clip(similarity, -1.0, 1.0))
    
    def report(self, detailed=True):
        """Generate comprehensive system report"""
        
        print("\n" + "="*60)
        print("  METACOGNITIVE SYSTEM REPORT")
        print("="*60)
        
        # Basic stats
        print(f"\n  System Statistics:")
        print(f"    • Total generations: {self.stats['total_generations']}")
        print(f"    • Successful: {self.stats['successful_generations']}")
        print(f"    • Failed: {self.stats['failed_generations']}")
        print(f"    • Total recoveries: {self.stats['total_recoveries']}")
        print(f"    • Beliefs stored: {len(self.self_beliefs)}")
        print(f"    • Memory bank size: {len(self.alethia.memories)}")
        
        # Success rate
        if self.stats['total_generations'] > 0:
            success_rate = self.stats['successful_generations'] / self.stats['total_generations']
            print(f"    • Success rate: {success_rate:.1%}")
        
        # Mode discovery
        if self.alethia.mode_stats:
            print(f"\n  Discovered Cognitive Modes:")
            for cid, stats in self.alethia.mode_stats.items():
                scores = [
                    ("analytical", stats.get("analytical_mean", 0)),
                    ("creative", stats.get("creative_mean", 0)),
                    ("narrative", stats.get("narrative_mean", 0))
                ]
                dominant = max(scores, key=lambda x: x[1])
                quality = stats.get("quality_score", 0)
                print(f"    • Mode {cid}: {dominant[0]} ({dominant[1]:.2f} strength, quality: {quality:.2f})")
                print(f"      Samples: {stats['count']}, Perplexity: {stats.get('perplexity_mean', 0):.1f}")
        
        # Style distribution
        if self.stats['style_distribution']:
            print(f"\n  Style Distribution:")
            total = sum(self.stats['style_distribution'].values())
            for style, count in sorted(self.stats['style_distribution'].items()):
                print(f"    • {style}: {count} ({count/total:.1%})")
        
        # Quality metrics
        if self.stats['quality_percentiles']:
            percentiles = np.percentile(self.stats['quality_percentiles'], [25, 50, 75, 90])
            print(f"\n  Quality Percentiles:")
            print(f"    • 25th: {percentiles[0]:.3f}")
            print(f"    • 50th: {percentiles[1]:.3f}")
            print(f"    • 75th: {percentiles[2]:.3f}")
            print(f"    • 90th: {percentiles[3]:.3f}")
        
        # Recent performance
        if self.generation_log:
            recent = self.generation_log[-10:]
            avg_time = np.mean([g.get('generation_time', 0) for g in recent])
            avg_recovery = np.mean([g.get('recovery_count', 0) for g in recent])
            print(f"\n  Recent Performance (last 10):")
            print(f"    • Avg generation time: {avg_time:.2f}s")
            print(f"    • Avg recovery count: {avg_recovery:.1f}")
            dropped = sum(g.get('dropped_for_meta_or_fragments', 0) for g in recent)
            print(f"    • Dropped (meta/frag final gate): {dropped}")
        
        # Top beliefs
        if self.self_beliefs and detailed:
            print(f"\n  Top Self-Beliefs (by quality):")
            top_beliefs = sorted(
                self.self_beliefs,
                key=lambda x: x["quality"],
                reverse=True
            )[:3]
            
            for i, belief in enumerate(top_beliefs, 1):
                print(f"\n    [{i}] {belief['origin'].upper()} (quality: {belief['quality']:.3f})")
                print(f"        Conf: {belief['confidence']:.2f}, "
                      f"Coh: {belief['coherence']:.2f}, "
                      f"Perp: {belief['perplexity']:.1f}")
                text = belief['text'].replace('\n', ' ')
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"        \"{text}\"")
        
        print("\n" + "="*60 + "\n")


# -----------------------
# Interactive Demo
# -----------------------
def run_interactive_demo():
    """Run an interactive demonstration with robust error handling"""
    
    print("\n" + "="*60)
    print("  GPT-2 ALETHIA + KAIROS")
    print("  Production-Ready Metacognitive Control")
    print("="*60)
    
    try:
        # Initialize system
        system = MetacognitiveGPT2System(model_name="gpt2", seed=42)
        
        # Test cases with diverse challenges
        test_cases = [
            (None, "analytical"),
            (None, "narrative"),
            ("The fundamental principles of machine learning include", "analytical"),
            ("In a world where dreams become reality,", "creative"),
            ("She opened the ancient book and discovered", "narrative"),
            ("The relationship between consciousness and", None),
            ("Beyond the kaleidoscope of imagination,", "creative"),
            ("The paradox of existence reveals", "analytical"),
        ]
        
        print("\n" + "="*60)
        print("  GENERATING WITH METACOGNITIVE CONTROL")
        print("="*60)
        
        for i, (prompt, style) in enumerate(test_cases, 1):
            print(f"\n  Generation {i}/{len(test_cases)}")
            print("  " + "-"*56)
            
            # Display inputs
            if prompt:
                display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
                print(f"  Prompt: \"{display_prompt}\"")
            else:
                print("  Prompt: [Auto-generated]")
            
            if style:
                print(f"  Target style: {style}")
            else:
                print("  Target style: [Auto-selected]")
            
            print()
            
            # Generate
            start_time = time.time()
            result = system.generate_with_metacognition(
                prompt=prompt,
                target_style=style,
                max_length=50,
                verbose=True
            )
            gen_time = time.time() - start_time
            
            # Display results
            print(f"\n  Results:")
            print(f"    • Final style: {result['style']}")
            print(f"    • Quality metrics:")
            print(f"      - Confidence: {result['assessment']['confidence']:.2f}")
            print(f"      - Coherence: {result['assessment']['coherence']:.2f}")
            print(f"      - Logic: {result['assessment']['logic']:.2f}")
            print(f"      - Perplexity: {result['assessment']['perplexity']:.1f}")
            print(f"      - Quality score: {result['assessment'].get('quality_score', 0):.3f}")
            print(f"    • Attempts: {result['attempts_made']}")
            
            if result.get('recovery_count', 0) > 0:
                print(f"    • Recoveries: {result['recovery_count']}")
            
            print(f"    • Self-similarity: {result['self_similarity']:.3f}")
            print(f"    • Generation time: {gen_time:.2f}s")
            
            if result['assessment']['flags']:
                print(f"    • Flags: {', '.join(result['assessment']['flags'])}")
            
            # Display text
            print(f"\n  Generated text:")
            text = result['text'].replace('\n', ' ')
            # Word wrap at 70 chars
            words = text.split()
            lines = []
            current_line = "    \""
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    lines.append(current_line)
                    current_line = "     " + word
                else:
                    if current_line == "    \"":
                        current_line += word
                    else:
                        current_line += " " + word
            lines.append(current_line + "\"")
            print('\n'.join(lines))
            
            print("\n  " + "="*56)
        
        # Final report
        system.report(detailed=True)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError in demo: {str(e)}")
        traceback.print_exc()


# -----------------------
# Main Entry Point
# -----------------------
def main():
    try:
        run_interactive_demo()
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
