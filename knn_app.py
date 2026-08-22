# knn_app.py
# โปรแกรม AI ด้วยเทคนิค k-Nearest Neighbor (kNN) สำหรับทำนายระดับฝีมือนักยกน้ำหนัก (TIER)
# จาก 6 features: SEX, AGE, BODYWEIGHT, SQUAT, BENCH, DEADLIFT
# GUI ทำด้วย Tkinter

import os
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
# ข้อมูลจริงจาก OpenPowerlifting: 6 raw features (ไม่ต้องคำนวณเพิ่ม)
FEATURE_COLS = ["SEX", "AGE", "BODYWEIGHT", "SQUAT", "BENCH", "DEADLIFT"]
LABEL_COL = "TIER"

# ---------- สี/ธีม ให้ใกล้เคียงภาพตัวอย่าง ----------
PURPLE = "#3d2170"
LIGHT_PURPLE = "#c9b8f0"
GREEN = "#4caf27"
WHITE = "#ffffff"
DARK = "#1b1030"


class KNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KNN APP")
        self.geometry("980x640")
        self.configure(bg=WHITE)
        self.resizable(False, False)

        # ---------- โหลดข้อมูล + เทรนโมเดลเริ่มต้น ----------
        self.df = self.load_data()
        self.scaler = StandardScaler()
        self.model = None
        self.train_model(k=3)

        self.build_ui()

    # ---------------- Data / Model ----------------
    def load_data(self):
        if not os.path.exists(DATA_PATH):
            messagebox.showerror("ไม่พบไฟล์ข้อมูล", f"ไม่พบไฟล์ {DATA_PATH}")
            self.destroy()
            raise SystemExit
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            messagebox.showerror(
                "คอลัมน์ไม่ครบ",
                "ไฟล์ data.csv ต้องมีคอลัมน์: " + ", ".join(FEATURE_COLS + [LABEL_COL]) +
                "\nคอลัมน์ที่ขาด: " + ", ".join(missing)
            )
            self.destroy()
            raise SystemExit
        return df

    def train_model(self, k):
        X = self.df[FEATURE_COLS].values
        y = self.df[LABEL_COL].values
        X_scaled = self.scaler.fit_transform(X)
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(X_scaled, y)

    # ---------------- UI ----------------
    def build_ui(self):
        # ----- Header -----
        header = tk.Frame(self, bg=WHITE)
        header.pack(fill="x", padx=20, pady=(15, 5))
        tk.Label(header, text="KNN APP", font=("Segoe UI", 24, "bold"),
                  bg=WHITE, fg=DARK).pack(side="left")

        main = tk.Frame(self, bg=WHITE)
        main.pack(fill="both", expand=True, padx=20, pady=5)

        # ----- ซ้าย: ตารางข้อมูลฝึกฝน -----
        left = tk.Frame(main, bg=DARK)
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))

        tk.Label(left, text=f"ข้อมูลฝึกฝนทั้งหมด {len(self.df)} รายการ",
                  bg=DARK, fg=WHITE, font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        table_frame = tk.Frame(left, bg=DARK)
        table_frame.pack(fill="both", expand=True, padx=8, pady=8)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview.Heading", background=LIGHT_PURPLE,
                          foreground=DARK, font=("Segoe UI", 9, "bold"))
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 9))

        cols = FEATURE_COLS + [LABEL_COL]
        col_widths = {"SEX": 45, "AGE": 50, "BODYWEIGHT": 85, "SQUAT": 65, "BENCH": 65, "DEADLIFT": 70, "TIER": 95}
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=col_widths.get(c, 75), anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for _, row in self.df.iterrows():
            self.tree.insert("", "end", values=list(row[cols]))

        # search box (ฟีเจอร์เสริม)
        search_frame = tk.Frame(left, bg=DARK)
        search_frame.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(search_frame, text="ค้นหา (TIER):", bg=DARK, fg=WHITE).pack(side="left")
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=15)
        search_entry.pack(side="left", padx=6)
        tk.Button(search_frame, text="ค้นหา", command=self.search_table).pack(side="left")
        tk.Button(search_frame, text="แสดงทั้งหมด", command=self.refresh_table).pack(side="left", padx=4)

        # ----- ขวา: Input form -----
        right = tk.Frame(main, bg=PURPLE, width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="INPUT", font=("Segoe UI", 18, "bold"),
                  bg=PURPLE, fg=WHITE).pack(anchor="e", padx=15, pady=(15, 10))

        self.entries = {}
        field_labels = {
            "SEX": "SEX (1=ชาย, 0=หญิง)",
            "AGE": "AGE (ปี)",
            "BODYWEIGHT": "BODYWEIGHT (kg)",
            "SQUAT": "BEST SQUAT (kg)",
            "BENCH": "BEST BENCH (kg)",
            "DEADLIFT": "BEST DEADLIFT (kg)",
        }
        for feat in FEATURE_COLS:
            row = tk.Frame(right, bg=PURPLE)
            row.pack(fill="x", padx=15, pady=5)
            tk.Label(row, text=field_labels[feat], bg=PURPLE, fg=WHITE,
                      font=("Segoe UI", 9, "bold"), width=17, anchor="w").pack(side="left")
            e = tk.Entry(row, width=7, font=("Segoe UI", 11), justify="center")
            e.pack(side="right")
            self.entries[feat] = e

        # ค่า k
        k_row = tk.Frame(right, bg=PURPLE)
        k_row.pack(fill="x", padx=15, pady=(15, 6))
        tk.Label(k_row, text="เลือกจำนวนเพื่อนบ้าน (k)", bg=PURPLE, fg=WHITE,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.k_var = tk.StringVar(value="3")
        k_spin = tk.Spinbox(k_row, from_=1, to=25, increment=1, width=6,
                              textvariable=self.k_var, font=("Segoe UI", 12), justify="center")
        k_spin.pack(anchor="w", pady=4)

        # ----- ล่าง: Predict + ผลลัพธ์ -----
        bottom = tk.Frame(self, bg=GREEN)
        bottom.pack(fill="x", side="bottom", padx=20, pady=15, ipady=15)

        predict_btn = tk.Button(bottom, text="Predict", font=("Segoe UI", 14, "bold"),
                                  bg=LIGHT_PURPLE, fg=DARK, activebackground="#b39ddb",
                                  relief="flat", padx=25, pady=8, command=self.predict)
        predict_btn.pack(side="left", padx=(30, 15))

        result_frame = tk.Frame(bottom, bg=WHITE, relief="flat")
        result_frame.pack(side="left", padx=10, ipadx=10, ipady=6)
        tk.Label(result_frame, text="Predict", bg=GREEN if False else "#e8f5e9",
                  fg=DARK, font=("Segoe UI", 12, "bold")).pack(side="left")

        self.result_var = tk.StringVar(value="-")
        self.result_label = tk.Label(result_frame, textvariable=self.result_var,
                                        bg=WHITE, fg=PURPLE, font=("Segoe UI", 14, "bold"))
        self.result_label.pack(side="left", padx=10)

    # ---------------- Actions ----------------
    def refresh_table(self):
        self.search_var.set("")
        self.tree.delete(*self.tree.get_children())
        for _, row in self.df.iterrows():
            self.tree.insert("", "end", values=list(row[FEATURE_COLS + [LABEL_COL]]))

    def search_table(self):
        q = self.search_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        subset = self.df if not q else self.df[self.df[LABEL_COL].str.lower() == q]
        for _, row in subset.iterrows():
            self.tree.insert("", "end", values=list(row[FEATURE_COLS + [LABEL_COL]]))

    def validate_inputs(self):
        """ตรวจสอบว่าข้อมูลที่กรอกเป็นตัวเลขและอยู่ในช่วงที่สมเหตุสมผล"""
        values = {}
        for feat in FEATURE_COLS:
            raw = self.entries[feat].get().strip()
            if raw == "":
                messagebox.showwarning("ข้อมูลไม่ครบ", f"กรุณากรอกค่า {feat}")
                return None
            try:
                val = float(raw)
            except ValueError:
                messagebox.showwarning("ข้อมูลไม่ถูกต้อง", f"{feat} ต้องเป็นตัวเลขเท่านั้น")
                return None
            values[feat] = val

        if values["SEX"] not in (0, 1):
            messagebox.showwarning("ข้อมูลไม่ถูกต้อง", "SEX ต้องเป็น 0 หรือ 1 เท่านั้น")
            return None
        if not (10 < values["AGE"] < 90):
            messagebox.showwarning("ข้อมูลไม่ถูกต้อง", "AGE ต้องอยู่ระหว่าง 10-90 ปี")
            return None
        if not (30 < values["BODYWEIGHT"] < 200):
            messagebox.showwarning("ข้อมูลไม่ถูกต้อง", "BODYWEIGHT ไม่สมเหตุสมผล")
            return None
        for lift in ("SQUAT", "BENCH", "DEADLIFT"):
            if not (0 < values[lift] < 500):
                messagebox.showwarning("ข้อมูลไม่ถูกต้อง", f"{lift} ต้องอยู่ระหว่าง 0-500 กก.")
                return None
        return values

    def predict(self):
        values = self.validate_inputs()
        if values is None:
            return

        try:
            k = int(self.k_var.get())
            if k < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("ค่า k ไม่ถูกต้อง", "k ต้องเป็นจำนวนเต็มบวก")
            return

        if k > len(self.df):
            messagebox.showwarning("ค่า k มากเกินไป", f"k ต้องไม่เกินจำนวนข้อมูลฝึกฝน ({len(self.df)})")
            return

        # เทรนโมเดลใหม่ทุกครั้งที่ k เปลี่ยน (รองรับการเลือก k อิสระ)
        self.train_model(k)

        X_new = [[values[f] for f in FEATURE_COLS]]
        X_new_scaled = self.scaler.transform(X_new)
        pred = self.model.predict(X_new_scaled)[0]

        self.result_var.set(pred)


if __name__ == "__main__":
    app = KNNApp()
    app.mainloop()
