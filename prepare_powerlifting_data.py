# prepare_powerlifting_data.py
# แปลงข้อมูลจริงจาก Kaggle/OpenPowerlifting (openpowerlifting.csv)
# ให้อยู่ในรูปแบบที่ knn_app.py ใช้ทำนาย "ระดับฝีมือ" (Skill Tier) ของนักยกน้ำหนัก
#
# แนวคิด:
#   - ใช้ 6 raw features จริงจากข้อมูล (ไม่ต้องคำนวณเพิ่ม): SEX, AGE, BODYWEIGHT,
#     SQUAT, BENCH, DEADLIFT
#   - label (TIER) คำนวณจากคะแนน Wilks (มาตรฐานที่วงการยกน้ำหนักใช้เทียบฟอร์มข้ามน้ำหนักตัว/เพศ)
#     แบ่งช่วงตามเกณฑ์ที่นิยมใช้กันทั่วไปในวงการ:
#       < 200        -> Beginner
#       200 - 300     -> Novice
#       300 - 400     -> Intermediate
#       400 - 500     -> Advanced
#       >= 500        -> Elite
#   - หมายเหตุสำคัญ: Wilks score เอง "ไม่ถูกใช้เป็น feature" (เพราะ label คำนวณมาจากมันโดยตรง
#     ถ้าใส่ Wilks เป็น feature ด้วยโมเดลจะ "โกง" ได้คำตอบง่ายเกินไป ไม่ใช่การเรียนรู้จริง)

import pandas as pd

SRC = "/mnt/user-data/uploads/openpowerlifting.csv"
DST = "data.csv"

RAW_COLS = ["Sex", "Age", "BodyweightKg", "BestSquatKg", "BestBenchKg", "BestDeadliftKg", "Wilks"]

print("กำลังอ่านไฟล์ (ขนาดใหญ่ อาจใช้เวลาสักครู่)...")
df = pd.read_csv(SRC, usecols=RAW_COLS)
print(f"อ่านข้อมูลดิบทั้งหมด {len(df)} แถว")

# ---- ทำความสะอาดข้อมูล ----
df = df.dropna()
df = df[(df.BestSquatKg > 0) & (df.BestBenchKg > 0) & (df.BestDeadliftKg > 0)]
df = df[(df.Age >= 10) & (df.Age <= 90)]
df = df[(df.BodyweightKg >= 30) & (df.BodyweightKg <= 200)]
print(f"หลังทำความสะอาดข้อมูล เหลือ {len(df)} แถว")

# ---- เข้ารหัส SEX ----
df["SEX"] = df["Sex"].map({"M": 1, "F": 0})

# ---- ตั้งชื่อคอลัมน์ features ----
df["AGE"] = df["Age"]
df["BODYWEIGHT"] = df["BodyweightKg"].round(1)
df["SQUAT"] = df["BestSquatKg"].round(1)
df["BENCH"] = df["BestBenchKg"].round(1)
df["DEADLIFT"] = df["BestDeadliftKg"].round(1)

# ---- คำนวณ label จาก Wilks ----
bins = [0, 200, 300, 400, 500, 10000]
labels = ["Beginner", "Novice", "Intermediate", "Advanced", "Elite"]
df["TIER"] = pd.cut(df["Wilks"], bins=bins, labels=labels)

feature_cols = ["SEX", "AGE", "BODYWEIGHT", "SQUAT", "BENCH", "DEADLIFT"]
out = df[feature_cols + ["TIER"]].dropna()

# ---- สุ่มตัวอย่างแบบ stratified ให้แต่ละคลาสสมดุลกัน และขนาดพอเหมาะกับ GUI ----
MAX_PER_CLASS = 300
parts = []
for tier in labels:
    g = out[out["TIER"] == tier]
    parts.append(g.sample(n=min(len(g), MAX_PER_CLASS), random_state=42))
sampled = pd.concat(parts, ignore_index=True)
sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)  # สลับลำดับ

sampled.to_csv(DST, index=False, encoding="utf-8-sig")
print(f"\nสร้างไฟล์ {DST} สำเร็จ: {len(sampled)} แถว (สุ่มแบบสมดุลจากข้อมูลจริง {len(out)} แถว)")
print(sampled["TIER"].value_counts())
