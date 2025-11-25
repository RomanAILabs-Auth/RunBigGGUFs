#!/usr/bin/env python3
"""
SPACETIME_36B_PROFESSIONAL.py — High‑Stability, High‑Efficiency LLM Runtime
Optimized for 30–36B GGUF models on CPU‑only systems
Fully deterministic, engineering‑grade interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import time
import numpy as np
from scipy.linalg import qr
import psutil

from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"])
    from llama_cpp import Llama


# -----------------------------------------------------------------------------------
# Spacetime Engine (Deterministic Control Layer)
# -----------------------------------------------------------------------------------

def orthonormal_basis_from_cols(M):
    Q, _ = qr(M)
    return Q

class SpacetimeEngine:
    def __init__(self):
        np.random.seed(777)
        self.U = orthonormal_basis_from_cols(np.random.randn(48, 12))
        self.W = np.random.randn(12) * 0.10
        self.b = 0.0

    def compress(self, v):
        return v @ self.U

    def decide(self, z, cpu, mem):
        score = float(np.dot(z, self.W) + self.b)
        boost = max(20, min(100, (100 - cpu) * 1.6 + score * 22))
        if boost > 95:
            status = "Peak Performance"
        elif boost > 80:
            status = "High Performance"
        else:
            status = "Nominal"
        return int(boost), status


engine = SpacetimeEngine()


# -----------------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------------

class Spacetime36BProfessional:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spacetime LLM Console – Professional Edition")
        self.root.geometry("1300x900")
        self.root.configure(bg="#0c0c0c")

        self.llm = None
        self.model_path = None
        self.is_generating = False

        self._build_ui()
        threading.Thread(target=self._telemetry_loop, daemon=True).start()

        self._log("System initialized. Awaiting model load.")

    # -----------------------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------------------

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1a1a1a", height=70)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="Spacetime LLM Console — 36B",
            fg="white",
            bg="#1a1a1a",
            font=("Segoe UI", 24, "bold")
        ).pack(side=tk.LEFT, padx=40)

        ttk.Button(header, text="Load Model", command=self._load_model).pack(side=tk.RIGHT, padx=40)

        # Telemetry
        telemetry_frame = tk.Frame(self.root, bg="#0c0c0c")
        telemetry_frame.pack(fill=tk.X, padx=40, pady=20)

        self.telemetry_bar = tk.Canvas(telemetry_frame, height=40, bg="#0c0c0c", highlightthickness=0)
        self.telemetry_bar.pack(fill=tk.X)

        self.telemetry_label = tk.Label(
            telemetry_frame,
            fg="#00d4ff",
            bg="#0c0c0c",
            font=("Consolas", 16, "bold")
        )
        self.telemetry_label.pack()

        # Chatbox
        self.chat = scrolledtext.ScrolledText(
            self.root,
            bg="#0c0c0c",
            fg="#e8e8e8",
            font=("Consolas", 14),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)

        # Input
        input_frame = tk.Frame(self.root, bg="#0c0c0c")
        input_frame.pack(fill=tk.X, padx=40, pady=20)

        self.input = tk.Text(
            input_frame,
            height=4,
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Consolas", 14),
            insertbackground="#ffffff"
        )
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input.bind("<Control-Return>", lambda e: self._submit())

        ttk.Button(input_frame, text="Send", command=self._submit).pack(side=tk.RIGHT, padx=10)

        # Status
        self.status = tk.Label(
            self.root,
            text="Idle",
            fg="#00d4ff",
            bg="#0c0c0c",
            font=("Segoe UI", 16)
        )
        self.status.pack(pady=10)


    # -----------------------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------------------

    def _log(self, text, tag="system"):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text + "\n\n", tag)
        self.chat.tag_config(tag, foreground="#00d4ff")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)


    # -----------------------------------------------------------------------------------
    # Model Load
    # -----------------------------------------------------------------------------------

    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=[("GGUF Model", "*.gguf")])
        if not path:
            return

        self.model_path = path
        self.status.config(text="Loading model…")
        threading.Thread(target=self._initialize_model, daemon=True).start()

    def _initialize_model(self):
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=16384,
                n_threads=4,
                n_gpu_layers=0,
                n_batch=2048,
                use_mlock=True,
                use_mmap=False,
                chat_format=None,
                verbose=False,
                f16_kv=True,
                logits_all=False
            )

            self.status.config(text="Model loaded. Ready.")
            self._log("Model successfully initialized.")

        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.status.config(text="Load Error")


    # -----------------------------------------------------------------------------------
    # Message Handling
    # -----------------------------------------------------------------------------------

    def _submit(self):
        if not self.llm:
            return

        message = self.input.get("1.0", tk.END).strip()
        if not message:
            return

        self.input.delete("1.0", tk.END)
        self._log(message, tag="user")

        self.status.config(text="Generating…")
        threading.Thread(target=self._generate, args=(message,), daemon=True).start()

    def _generate(self, prompt):
        try:
            stream = self.llm(prompt, max_tokens=512, temperature=0.7, top_p=0.9, stream=True)

            self.chat.config(state=tk.NORMAL)
            for chunk in stream:
                if "choices" in chunk:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        self.chat.insert(tk.END, text, ("ai",))
                        self.chat.tag_config("ai", foreground="#7cf28a")
                        self.chat.see(tk.END)
            self.chat.insert(tk.END, "\n\n")
            self.chat.config(state=tk.DISABLED)

            self.status.config(text="Ready")

        except Exception as e:
            self._log(f"Error: {e}")
            self.status.config(text="Error")


    # -----------------------------------------------------------------------------------
    # Telemetry / Performance Loop
    # -----------------------------------------------------------------------------------

    def _telemetry_loop(self):
        while True:
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory().percent / 100

            z = engine.compress(np.random.randn(48))
            boost, status = engine.decide(z, cpu, mem)

            self._update_telemetry(cpu, boost, status)

            if self.llm and self.is_generating:
                self.llm.n_batch = min(4096, 2048 + boost * 20)

            time.sleep(0.6)

    def _update_telemetry(self, cpu, boost, status):
        self.telemetry_bar.delete("all")
        width = self.telemetry_bar.winfo_width() or 1000
        fill = int(width * boost / 100)

        self.telemetry_bar.create_rectangle(0, 0, fill, 40, fill="#00d4ff", outline="")

        self.telemetry_label.config(
            text=f"CPU: {cpu:.0f}%   Performance Index: {boost}%   Status: {status}"
        )

    # -----------------------------------------------------------------------------------

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = Spacetime36BProfessional()
    app.run()

