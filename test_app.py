# test_app.py
# สคริปต์ทดสอบเบื้องต้น เพื่อยืนยันว่าโมเดลและข้อมูลทำงานถูกต้อง
# (ใช้ประกอบข้อ 6 "มีการตรวจสอบการแก้ไขโปรแกรม")
# รันด้วย: python test_app.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
FEATURE_COLS = ["SEX", "AGE", "BODYWEIGHT", "SQUAT", "BENCH", "DEADLIFT"]
LABEL_COL = "TIER"


def test_data_loaded():
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 500, f"ต้องมีข้อมูล >= 500 แถว แต่มี {len(df)}"
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in df.columns, f"ขาดคอลัมน์ {col}"
    print(f"[PASS] ข้อมูลมี {len(df)} แถว, {len(FEATURE_COLS)} features")


def test_model_accuracy():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    for k in [1, 3, 5, 7]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        print(f"[INFO] k={k}  accuracy={acc:.3f}")
        assert acc > 0.5, f"ความแม่นยำที่ k={k} ต่ำเกินไป ({acc:.3f})"
    print("[PASS] โมเดล kNN ทำนายได้แม่นยำในระดับที่ยอมรับได้ทุกค่า k ที่ทดสอบ")


def test_single_prediction():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    scaler = StandardScaler().fit(X)
    model = KNeighborsClassifier(n_neighbors=3).fit(scaler.transform(X), y)

    # เคสนักยกน้ำหนักมือใหม่ชัดเจน: น้ำหนักตัวมาก แต่ยกได้น้อย -> ควรถูกจัดเป็น Beginner/Novice
    sample = [[1, 25, 90, 60, 40, 70]]  # SEX, AGE, BODYWEIGHT, SQUAT, BENCH, DEADLIFT
    pred = model.predict(scaler.transform(sample))[0]
    print(f"[INFO] ตัวอย่างทำนาย (มือใหม่ ยกน้ำหนักน้อย): {pred}")
    assert pred in ("Beginner", "Novice"), f"คาดว่าจะทำนายเป็นกลุ่มมือใหม่ แต่ได้ '{pred}'"
    print("[PASS] ทำนายกรณีทดสอบตัวอย่างได้ถูกต้อง")


if __name__ == "__main__":
    test_data_loaded()
    test_model_accuracy()
    test_single_prediction()
    print("\n=== ทดสอบทั้งหมดผ่าน ===")
