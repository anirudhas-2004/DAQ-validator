# simple_ui.py

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "scoring_layer.py"
MANIFEST = ROOT / "manifest.json"
OUTPUT_PREFIX = "daq_validation"

COLORS = {"FAIL": "#cc0000", "SUSPECT": "#b8860b", "PASS": "#2e7d32"}


class DAQValidatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DAQ Validator")
        self.root.geometry("1200x700")

        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        self.run_button = tk.Button(
            top_frame, text="Run Validation", command=self.run_validation,
            bg="#2d89ef", fg="white", font=("Arial", 11, "bold"), padx=20, pady=8,
        )
        self.run_button.pack(side=tk.LEFT)

        self.status_label = tk.Label(top_frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=20)

        # --- Treeview ---
        tree_frame = tk.Frame(root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        columns = ("channel", "verdict", "combined_score", "dc_score", "sine_score",
                   "sinad", "thd", "sfdr")

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for col, text, width in [
            ("channel", "Channel", 90), ("verdict", "Verdict", 80),
            ("combined_score", "Combined", 90), ("dc_score", "DC Score", 80),
            ("sine_score", "Sine Score", 85), ("sinad", "SINAD (dB)", 90),
            ("thd", "THD (dB)", 85), ("sfdr", "SFDR (dB)", 85),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, minwidth=width)

        self.tree.tag_configure("PASS", background="#dff0d8")
        self.tree.tag_configure("SUSPECT", background="#fcf8e3")
        self.tree.tag_configure("FAIL", background="#f2dede")

        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)
        self._reasons = {}

        # --- Verdict color in the Verdict cell via a hack: overlay with a canvas is not
        #     possible in ttk natively, so we use a style per tag on the text color instead.
        style = ttk.Style()
        style.map("Treeview", foreground=[])  # reset any platform override

        # --- Reasons panel ---
        detail_frame = tk.Frame(root, padx=10, pady=6)
        detail_frame.pack(fill=tk.X, padx=10, pady=(6, 10))

        tk.Label(detail_frame, text="Reasons", font=("Arial", 9, "bold"), fg="#333").pack(anchor=tk.W)

        self.detail_box = tk.Text(
            detail_frame, height=4, wrap=tk.WORD, font=("Arial", 9),
            state=tk.DISABLED, bg="#f5f5f5", relief=tk.GROOVE, bd=1,
            padx=8, pady=6,
        )
        self.detail_box.pack(fill=tk.X)

        for verdict, color in COLORS.items():
            self.detail_box.tag_configure(verdict, foreground=color, font=("Arial", 9, "bold"))
        self.detail_box.tag_configure("text", foreground="#222222")

    def on_row_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        reasons = self._reasons.get(iid, "")
        verdict = self.tree.item(iid, "tags")[0]

        # Split "DC: ...; SINE: ..." into individual reasons for nicer display
        parts = [r.strip() for r in reasons.split(";") if r.strip()]

        self.detail_box.config(state=tk.NORMAL)
        self.detail_box.delete("1.0", tk.END)
        self.detail_box.insert(tk.END, f"{verdict}  ", verdict)
        self.detail_box.insert(tk.END, "\n".join(f"  • {p}" for p in parts), "text")
        self.detail_box.config(state=tk.DISABLED)

    def run_validation(self):
        self.status_label.config(text="Running...")
        self.detail_box.config(state=tk.NORMAL)
        self.detail_box.delete("1.0", tk.END)
        self.detail_box.config(state=tk.DISABLED)
        self._reasons.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)

        if not ANALYZER.exists():
            messagebox.showerror("Error", f"Analyzer not found:\n{ANALYZER}")
            return
        if not MANIFEST.exists():
            messagebox.showerror("Error", f"Manifest not found:\n{MANIFEST}")
            return

        result = subprocess.run(
            ["python", str(ANALYZER), "--manifest", str(MANIFEST), "--out-prefix", OUTPUT_PREFIX],
            cwd=ROOT, capture_output=True, text=True,
        )

        if result.returncode != 0:
            messagebox.showerror("Analyzer Error", result.stderr)
            self.status_label.config(text="Failed")
            return

        self.load_results()
        self.status_label.config(text="Completed")

    def load_results(self):
        csv_path = ROOT / f"{OUTPUT_PREFIX}.csv"
        if not csv_path.exists():
            messagebox.showerror("Error", f"Result CSV not found:\n{csv_path}")
            return

        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                verdict = row["verdict"]
                iid = self.tree.insert("", tk.END, tags=(verdict,), values=(
                    row["channel"], verdict, row["combined_score"], row["dc_score"],
                    row["sine_score"], row["sine_sinad_db"], row["sine_thd_db"],
                    row["sine_sfdr_db"],
                ))
                self._reasons[iid] = row["reasons"]


if __name__ == "__main__":
    root = tk.Tk()
    app = DAQValidatorUI(root)
    root.mainloop()