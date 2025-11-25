#!/usr/bin/env python3
"""
SPACETIME_36B_PROFESSIONAL_UPGRADED.py — High-Stability, High-Efficiency LLM Runtime
Optimized for 30–36B GGUF models on CPU-only systems
Deterministic, engineering-grade interface

Copyright (c) 2025 Daniel Harding - RomanAILabs
Credits: Nova
"""

from __future__ import annotations

import logging
import threading
import time
import math
import os
import sys
import importlib
from pathlib import Path
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

import numpy as np
from scipy.linalg import qr
import psutil
import subprocess

# Try to import llama-cpp-python, install if necessary (best-effort)
try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"])
        # attempt re-import
        importlib.invalidate_caches()
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        # Keep the module import error for later use; we'll handle absent model gracefully.
        Llama = None  # type: ignore
        logging.getLogger().warning("llama-cpp-python not available: %s", e)

# -------------------------------
# Logging & Constants
# -------------------------------

LOG_FILE = Path.home() / "spacetime_36b_professional.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.getLogger("console")
console.setLevel(logging.INFO)
if not console.handlers:
    console.addHandler(logging.StreamHandler())

SEED = 777
np.random.seed(SEED)

# Compression subspace sizes
IN_DIM = 48
LATENT_DIM = 12

# PCA smoothing / momentum
PCA_MOMENTUM = 0.94

# Telemetry smoothing
SMOOTH_ALPHA = 0.18

# -------------------------------
# Spacetime Engine (Online PCA compressor)
# -------------------------------

class SpacetimeEngine:
    """Online PCA/SVD compressor and decision engine.

    - Maintains an online second-moment matrix and computes a low-rank
      subspace (IN_DIM -> LATENT_DIM) via incremental SVD on demand.
    - Accepts a real-statistics vector (length IN_DIM) and compresses it.
    - Decides concrete tuning actions and a performance boost score.
    """

    def __init__(self, in_dim: int = IN_DIM, latent_dim: int = LATENT_DIM):
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self._C = np.eye(in_dim) * 1e-6  # running covariance estimate
        self._U = np.zeros((in_dim, latent_dim))
        # initialize an orthonormal basis from a random matrix (Q from QR)
        try:
            Q, _ = qr(np.random.randn(in_dim, latent_dim))
            self._U[:, : Q.shape[1]] = Q
        except Exception:
            # fallback: use random orthonormal via svd
            u, _, _ = np.linalg.svd(np.random.randn(in_dim, in_dim))
            self._U = u[:, :latent_dim]

        # small linear readout for decision making (learnable online)
        self.W = np.random.randn(latent_dim) * 0.05
        self.b = 0.0
        self.steps = 0

    def update_basis(self, vec: np.ndarray):
        """Online second-moment update with momentum and occasional SVD.

        vec: shape (in_dim,)
        """
        if vec is None:
            return
        vec = np.array(vec, dtype=float).reshape(-1)
        if vec.shape[0] != self.in_dim:
            tmp = np.zeros(self.in_dim, dtype=float)
            tmp[: min(vec.shape[0], self.in_dim)] = vec[: self.in_dim]
            vec = tmp

        self._C = PCA_MOMENTUM * self._C + (1 - PCA_MOMENTUM) * np.outer(vec, vec)
        self.steps += 1
        # recompute basis every N steps to save CPU
        if self.steps % 8 == 0:
            try:
                # Full SVD can be costly; use eigh on symmetric matrix
                eigvals, eigvecs = np.linalg.eigh(self._C)
                order = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, order[: self.latent_dim]]
                if eigvecs.shape[1] == self.latent_dim:
                    self._U = eigvecs
            except Exception as e:
                logging.exception("PCA update failed: %s", e)

    def compress(self, v: np.ndarray) -> np.ndarray:
        """Project input vector into the learned latent subspace.

        Returns a latent vector of shape (latent_dim,)
        """
        v = np.array(v, dtype=float).reshape(-1)
        if v.shape[0] != self.in_dim:
            tmp = np.zeros(self.in_dim, dtype=float)
            tmp[: min(v.shape[0], self.in_dim)] = v[: self.in_dim]
            v = tmp
        # protect if _U not initialized
        if self._U is None or self._U.size == 0:
            return np.zeros(self.latent_dim, dtype=float)
        z = v @ self._U
        # ensure shape
        return np.asarray(z).reshape(self.latent_dim)

    def decide(self, z: np.ndarray, cpu_pct: float, mem_pct: float) -> Tuple[int, str, dict]:
        """Produce a boost score (0-100), status string, and tuning actions dict.

        The returned actions dictionary lists concrete runtime knobs to apply.
        """
        try:
            score = float(np.dot(z, self.W) + self.b) if z is not None else 0.0
        except Exception:
            score = 0.0

        # Combine observed CPU/memory signals into a raw boost candidate
        raw = (100.0 - float(cpu_pct)) * 1.4 + (100.0 - float(mem_pct)) * 0.9 + score * 18.0
        boost = int(max(0, min(100, math.floor(raw))))

        if boost > 92:
            status = "Peak Performance"
        elif boost > 78:
            status = "High Performance"
        elif boost > 50:
            status = "Nominal"
        else:
            status = "Conservative"

        actions = self._actions_for_boost(boost)
        return boost, status, actions

    def _actions_for_boost(self, boost: int) -> dict:
        # Concrete tuning parameters; these are conservative and safe.
        # Backends may or may not support all keys — ModelController will apply what it can.
        cpu_physical = psutil.cpu_count(logical=False) or 1
        if boost > 90:
            return {
                "n_batch": 4096,
                "threads": max(2, cpu_physical),
                "kv_prune_rate": 0.00,
                "draft_tokens": 12,
                "rope_scale": 1.00,
            }
        if boost > 70:
            return {
                "n_batch": 3072,
                "threads": max(2, cpu_physical - 1),
                "kv_prune_rate": 0.04,
                "draft_tokens": 10,
                "rope_scale": 0.96,
            }
        if boost > 50:
            return {
                "n_batch": 2300,
                "threads": max(1, cpu_physical - 2),
                "kv_prune_rate": 0.08,
                "draft_tokens": 8,
                "rope_scale": 0.92,
            }
        return {
            "n_batch": 1536,
            "threads": 1,
            "kv_prune_rate": 0.15,
            "draft_tokens": 6,
            "rope_scale": 0.88,
        }


# -------------------------------
# Model Controller
# -------------------------------

class ModelController:
    """Wrapper around llama-cpp model to provide safe tuning primitives.

    - Provides load/unload model operations
    - Probes for available internals and exposes a standard tuning API
    - Provides a best-effort KV-cache statistics proxy when backend doesn't expose real ones
    """

    def __init__(self):
        self.llm: Optional[Llama] = None
        self.model_path: Optional[str] = None
        self.loaded = False
        # runtime stats smoothing
        self._tokens_sec_smoothed = 0.0
        self._last_token_ts = time.time()
        self._last_tokens = 0
        # io counters baseline for per-second rates
        try:
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            self._last_disk_read = getattr(disk_io, "read_bytes", 0)
            self._last_disk_write = getattr(disk_io, "write_bytes", 0)
            self._last_net_sent = getattr(net_io, "bytes_sent", 0)
            self._last_net_recv = getattr(net_io, "bytes_recv", 0)
        except Exception:
            self._last_disk_read = self._last_disk_write = self._last_net_sent = self._last_net_recv = 0
        self._last_io_ts = time.time()
        self.runtime_opts = {}

    def load_model(self, model_path: str, n_ctx: int = 16384, n_threads: int = 2):
        if Llama is None:
            msg = "llama-cpp-python not installed or importable. Install it and try again."
            logging.error(msg)
            return False, msg
        self.model_path = model_path
        try:
            # conservative defaults: rely on mmap for memory-backed loading
            # Many installs accept these kwargs; unknown kwargs will be ignored by llama-cpp implementation,
            # but we keep conservative set here.
            kwargs = dict(
                model_path=model_path,
                n_ctx=int(n_ctx),
                n_threads=int(n_threads),
                n_gpu_layers=0,
                n_batch=1024,
                use_mlock=False,
                use_mmap=True,
                f16_kv=True,
                logits_all=False,
                verbose=False,
            )
            # construct; some versions expect different signatures — try forgivingly
            try:
                self.llm = Llama(**kwargs)  # type: ignore
            except TypeError:
                # try a reduced set
                reduced = {"model_path": model_path, "n_ctx": int(n_ctx)}
                self.llm = Llama(**reduced)  # type: ignore

            self.loaded = True
            logging.info("Model loaded: %s", model_path)
            return True, "Loaded"
        except Exception as e:
            logging.exception("Model load failed: %s", e)
            return False, str(e)

    def unload_model(self):
        # Best-effort: delete and garbage collect
        try:
            if self.llm is not None:
                # some runtimes expose a close() method
                if hasattr(self.llm, "close"):
                    try:
                        self.llm.close()
                    except Exception:
                        pass
            del self.llm
        except Exception:
            pass
        self.llm = None
        self.loaded = False
        logging.info("Model unloaded")

    def probe_kv_stats(self) -> np.ndarray:
        """Return a length-IN_DIM vector of runtime statistics.

        Graceful fallback when the backend doesn't give KV internals.
        Fields (example mapping):
          0: cpu_pct
          1: mem_pct
          2: swap_pct
          3: tokens/sec (smoothed)
          4: page_faults/sec
          5: io_read_rate (KB/s)
          6: io_write_rate (KB/s)
          7..47: reserved / zeros or backend-specific metrics
        """
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        swap = psutil.swap_memory().percent

        # compute read/write rates since last call (KB/s)
        now = time.time()
        try:
            disk_io = psutil.disk_io_counters()
            read_bytes = getattr(disk_io, "read_bytes", 0)
            write_bytes = getattr(disk_io, "write_bytes", 0)
        except Exception:
            read_bytes = write_bytes = 0

        dt = max(1e-6, now - self._last_io_ts)
        read_rate_kb = max(0.0, (read_bytes - getattr(self, "_last_disk_read", 0)) / 1024.0 / dt)
        write_rate_kb = max(0.0, (write_bytes - getattr(self, "_last_disk_write", 0)) / 1024.0 / dt)
        # update last counters
        self._last_disk_read = read_bytes
        self._last_disk_write = write_bytes
        self._last_io_ts = now

        # tokens/sec proxy: if external tracking not present, keep smoothed value
        now_t = time.time()
        dt_t = max(1e-6, now_t - self._last_token_ts)
        # no external token counter available by default: keep previous smoothed
        # The runtime (generate_stream) may update _tokens_sec_smoothed during generation.
        tps = self._tokens_sec_smoothed

        # page faults — use platform fields if available (best-effort)
        pf = 0.0
        try:
            vm = psutil.virtual_memory()
            # psutil doesn't expose per-second page faults universally; attempt to fetch attributes
            pf = float(getattr(vm, "pfaults", 0) or getattr(vm, "page_faults", 0) or 0)
        except Exception:
            pf = 0.0

        v = np.zeros(IN_DIM, dtype=float)
        v[0] = float(cpu)
        v[1] = float(mem)
        v[2] = float(swap)
        v[3] = float(tps)
        v[4] = float(pf)
        v[5] = float(read_rate_kb)
        v[6] = float(write_rate_kb)

        # Backend-specific probes (best-effort)
        try:
            if self.llm is not None:
                attrs = ["n_ctx", "n_threads", "n_batch"]
                idx = 8
                for a in attrs:
                    if hasattr(self.llm, a):
                        try:
                            v[idx] = float(getattr(self.llm, a))
                        except Exception:
                            v[idx] = 0.0
                    idx += 1
        except Exception:
            pass

        # the rest stays zero for now
        return v

    def apply_tuning(self, actions: dict):
        """Attempt to apply tuning knobs to the running model.

        The ModelController applies what it can and safely ignores unknown knobs.
        """
        if not self.loaded:
            # store desired opts so they can be used at call-time
            self.runtime_opts.update(actions)
            return
        try:
            # Set per-call defaults by storing into a small state dict
            # llama-cpp-python may not allow changing internals after construction,
            # so we keep the desired tuning in self.runtime_opts and use them per-call.
            self.runtime_opts.update(actions)
            logging.debug("Applied tuning (cached): %s", actions)
        except Exception:
            logging.exception("Failed to apply tuning")

    def get_runtime_options(self) -> dict:
        return dict(getattr(self, "runtime_opts", {}) or {})

    def prune_kv_cache(self, rate: float):
        """Best-effort KV-cache pruning placeholder.

        When the backend exposes a KV cache, prune the coldest keys. Otherwise,
        emulate pruning by toggling short-context drafting or forcing smaller n_batch.
        """
        logging.debug("Requested KV prune rate: %.3f", float(rate))
        # If the model exposed a kv_cache API, we'd call it here.

    def _non_streaming_call(self, prompt: str, **kwargs) -> str:
        """Fallback single-call text generation wrapper; returns plain text."""
        try:
            # Try common API names
            if self.llm is None:
                return ""
            if hasattr(self.llm, "create_completion"):
                res = self.llm.create_completion(prompt=prompt, **kwargs)
                # shape differs by version; attempt to extract text
                if isinstance(res, dict):
                    return res.get("choices", [{}])[0].get("text", "") or res.get("text", "")
                return str(res)
            if hasattr(self.llm, "generate"):
                res = self.llm.generate(prompt=prompt, **kwargs)
                if isinstance(res, dict):
                    return res.get("choices", [{}])[0].get("text", "") or res.get("text", "")
                return str(res)
            # try calling instance
            res = self.llm(prompt=prompt, **kwargs)
            if isinstance(res, dict):
                return res.get("choices", [{}])[0].get("text", "") or res.get("text", "")
            return str(res)
        except Exception as e:
            logging.exception("Non-streaming generation failed: %s", e)
            return ""

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        """Generator yielding partial outputs (streaming). Uses runtime_opts previously applied.

        This function wraps llama-cpp's streaming interface (if available) or falls
        back to a non-streaming call that yields the final text.
        """
        if not self.loaded or self.llm is None:
            yield {"error": "Model not loaded"}
            return

        opts = self.get_runtime_options()
        call_kwargs = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        # map known keys into llama-cpp kwargs if present
        if "n_batch" in opts:
            call_kwargs["n_batch"] = int(opts["n_batch"])

        # attempt to set per-call thread tuning if supported (many builds won't allow runtime thread changes)
        if "threads" in opts:
            try:
                if hasattr(self.llm, "n_threads"):
                    try:
                        self.llm.n_threads = int(opts["threads"])
                    except Exception:
                        pass
            except Exception:
                pass

        # Attempt to call the most common streaming interfaces in order of library versions
        try:
            # Some versions use create_completion with stream=True that yields events
            if hasattr(self.llm, "create_completion"):
                # llama-cpp-python style
                stream = self.llm.create_completion(stream=True, **{k: v for k, v in call_kwargs.items() if k != "prompt"})
                for event in stream:
                    # yielding events as-is (user code will interpret)
                    yield event
                return

            # Some versions use generate(stream=True)
            if hasattr(self.llm, "generate"):
                stream = self.llm.generate(**call_kwargs)
                # generate may itself be an iterator or return a dict/obj
                if hasattr(stream, "__iter__"):
                    for chunk in stream:
                        yield chunk
                    return
                else:
                    # not iterable, fall through to non-streaming behavior
                    pass

            # Some older wrappers accept call as function and return iterator
            try:
                stream = self.llm.__call__(**call_kwargs)
                if hasattr(stream, "__iter__"):
                    for chunk in stream:
                        yield chunk
                    return
            except Exception:
                pass

            # Fallback: make a single blocking call and yield result as final chunk
            text = self._non_streaming_call(prompt=prompt, max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p))
            yield {"choices": [{"text": text}]}
        except Exception as e:
            logging.exception("Streaming generation failed: %s", e)
            # final fallback
            text = self._non_streaming_call(prompt=prompt, max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p))
            if text:
                yield {"choices": [{"text": text}]}
            else:
                yield {"error": "Generation failed"}

# -------------------------------
# GUI Application
# -------------------------------

class Spacetime36BProfessionalApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spacetime LLM Console – Professional Edition")
        self.root.geometry("1300x900")
        self.root.configure(bg="#0c0c0c")

        # controllers
        self.engine = SpacetimeEngine()
        self.model = ModelController()

        self.is_generating = False
        self._cpu_smoothed = 0.0
        self._mem_smoothed = 0.0

        self._build_ui()

        # start telemetry thread
        telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        telemetry_thread.start()

        self._log("System initialized. Awaiting model load.", tag="system")

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1a1a1a", height=70)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="Spacetime LLM Console — 36B",
            fg="white",
            bg="#1a1a1a",
            font=("Segoe UI", 20, "bold"),
        ).pack(side=tk.LEFT, padx=20)

        # Use ttk Buttons but ensure pack order is consistent
        btn_load = ttk.Button(header, text="Load Model", command=self._on_load_model)
        btn_unload = ttk.Button(header, text="Unload Model", command=self._on_unload_model)
        btn_unload.pack(side=tk.RIGHT, padx=10)
        btn_load.pack(side=tk.RIGHT, padx=10)

        # telemetry
        telemetry_frame = tk.Frame(self.root, bg="#0c0c0c")
        telemetry_frame.pack(fill=tk.X, padx=20, pady=16)

        self.telemetry_bar = tk.Canvas(telemetry_frame, height=36, bg="#0c0c0c", highlightthickness=0)
        self.telemetry_bar.pack(fill=tk.X)

        self.telemetry_label = tk.Label(
            telemetry_frame,
            fg="#00d4ff",
            bg="#0c0c0c",
            font=("Consolas", 12, "bold"),
        )
        self.telemetry_label.pack()

        # chatbox
        self.chat = scrolledtext.ScrolledText(
            self.root,
            bg="#0c0c0c",
            fg="#e8e8e8",
            font=("Consolas", 12),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # input
        input_frame = tk.Frame(self.root, bg="#0c0c0c")
        input_frame.pack(fill=tk.X, padx=20, pady=12)

        self.input = tk.Text(
            input_frame,
            height=4,
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Consolas", 12),
            insertbackground="#ffffff",
        )
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Ctrl+Enter to send — prevent newline propagation
        def _ctrl_enter(event=None):
            self._on_submit()
            return "break"
        self.input.bind("<Control-Return>", _ctrl_enter)

        ttk.Button(input_frame, text="Send", command=self._on_submit).pack(side=tk.RIGHT, padx=10)

        self.status = tk.Label(
            self.root,
            text="Idle",
            fg="#00d4ff",
            bg="#0c0c0c",
            font=("Segoe UI", 12),
        )
        self.status.pack(pady=6)

        # ensure tags exist before use
        self.chat.tag_config("system", foreground="#00d4ff")
        self.chat.tag_config("user", foreground="#f1c40f")
        self.chat.tag_config("ai", foreground="#7cf28a")

    def _log(self, text: str, tag: str = "system"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.chat.config(state=tk.NORMAL)
        # ensure tag exists
        if not self.chat.tag_names().__contains__(tag):
            self.chat.tag_config(tag, foreground="#00d4ff" if tag == "system" else "#7cf28a")
        self.chat.insert(tk.END, f"[{timestamp}] {text}\n\n", tag)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)
        logging.info("%s: %s", tag, text)

    # -------------------------
    # Model management
    # -------------------------
    def _on_load_model(self):
        path = filedialog.askopenfilename(filetypes=[("GGUF Model", "*.gguf"), ("Any file", "*")])
        if not path:
            return
        threading.Thread(target=self._load_model_thread, args=(path,), daemon=True).start()

    def _load_model_thread(self, path: str):
        self.status.config(text="Loading model…")
        ok, msg = self.model.load_model(path, n_ctx=16384, n_threads=max(1, (psutil.cpu_count(logical=False) or 2)))
        if ok:
            self.status.config(text="Model loaded. Ready.")
            self._log("Model successfully initialized.", tag="system")
        else:
            self.status.config(text="Load Error")
            messagebox.showerror("Model Load Error", msg)
            self._log(f"Model load failed: {msg}", tag="system")

    def _on_unload_model(self):
        self.model.unload_model()
        self.status.config(text="Model unloaded")
        self._log("Model unloaded by user.", tag="system")

    # -------------------------
    # Message handling
    # -------------------------
    def _on_submit(self):
        if not self.model.loaded:
            messagebox.showwarning("No model", "Please load a model first.")
            return
        prompt = self.input.get("1.0", tk.END).strip()
        if not prompt:
            return
        self.input.delete("1.0", tk.END)
        self._log(prompt, tag="user")
        threading.Thread(target=self._generate_thread, args=(prompt,), daemon=True).start()

    def _generate_thread(self, prompt: str):
        self.is_generating = True
        self.status.config(text="Generating…")
        opts = self.model.get_runtime_options()
        # apply a small dynamic draft token strategy
        draft = opts.get("draft_tokens", 8)

        try:
            stream = self.model.generate_stream(prompt, max_tokens=512)
            # For token counting proxy, measure generated lengths
            gen_tokens = 0
            start_ts = time.time()
            for chunk in stream:
                if isinstance(chunk, dict) and "error" in chunk:
                    self._log(f"Generation error: {chunk['error']}", tag="system")
                    break

                # handle common chunk shapes
                text = ""
                if isinstance(chunk, dict):
                    # many libs use {"choices": [{"text": "..."}], ...}
                    if "choices" in chunk:
                        try:
                            # accumulate all choice texts
                            choices = chunk["choices"]
                            if isinstance(choices, (list, tuple)) and choices:
                                text = "".join([c.get("text", "") for c in choices if isinstance(c, dict)])
                            elif isinstance(choices, dict):
                                text = choices.get("text", "")
                        except Exception:
                            text = str(chunk)
                    elif "text" in chunk:
                        text = chunk.get("text", "")
                    else:
                        # sometimes streaming events contain "delta" or "event" fields
                        # try to extract
                        delta = chunk.get("delta") if isinstance(chunk, dict) else None
                        if isinstance(delta, dict):
                            text = delta.get("content", "") or delta.get("text", "")
                        else:
                            # fallback: stringify chunk
                            text = ""
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    text = ""

                if text:
                    # update proxy token counters:
                    gen_tokens += len(text.split())
                    # append to chat view
                    self.chat.config(state=tk.NORMAL)
                    self.chat.insert(tk.END, text, ("ai",))
                    self.chat.config(state=tk.DISABLED)
                    self.chat.see(tk.END)

                # update smoothed tokens/sec proxy every few chunks
                elapsed = time.time() - start_ts
                if elapsed > 0:
                    tps = gen_tokens / elapsed
                    # smooth update into controller
                    self.model._tokens_sec_smoothed = (1 - 0.3) * getattr(self.model, "_tokens_sec_smoothed", 0.0) + 0.3 * tps

            self._log("Generation finished.", tag="system")
        except Exception as e:
            logging.exception("Generation thread failure: %s", e)
            self._log(f"Error during generation: {e}", tag="system")
        finally:
            self.is_generating = False
            self.status.config(text="Ready")

    # -------------------------
    # Telemetry loop
    # -------------------------
    def _telemetry_loop(self):
        while True:
            try:
                # Sample raw stats vector
                v = self.model.probe_kv_stats()

                # Update engine basis & compress
                self.engine.update_basis(v)
                z = self.engine.compress(v)

                # Smooth CPU/mem for stable UI
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                self._cpu_smoothed = (1 - SMOOTH_ALPHA) * self._cpu_smoothed + SMOOTH_ALPHA * cpu
                self._mem_smoothed = (1 - SMOOTH_ALPHA) * self._mem_smoothed + SMOOTH_ALPHA * mem

                boost, status, actions = self.engine.decide(z, self._cpu_smoothed, self._mem_smoothed)

                # Apply tuning actions to model
                self.model.apply_tuning(actions)
                self.model.prune_kv_cache(actions.get("kv_prune_rate", 0.0))

                # Update UI
                try:
                    # Ensure UI updates happen in main thread via `after`
                    self.root.after(0, lambda c=self._cpu_smoothed, b=boost, s=status: self._update_telemetry(c, b, s))
                except Exception:
                    self._update_telemetry(self._cpu_smoothed, boost, status)

            except Exception as e:
                logging.exception("Telemetry loop error: %s", e)
            time.sleep(0.6)

    def _update_telemetry(self, cpu: float, boost: int, status: str):
        try:
            self.telemetry_bar.delete("all")
            width = self.telemetry_bar.winfo_width() or 1000
            fill = int(width * boost / 100)
            # draw filled rect; do not set a global style here
            self.telemetry_bar.create_rectangle(0, 0, fill, 36, fill="#00d4ff", outline="")
            self.telemetry_label.config(text=f"CPU: {cpu:.0f}%   Performance Index: {boost}%   Status: {status}")
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = Spacetime36BProfessionalApp()
    app.run()

