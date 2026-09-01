# รายงานโครงงาน — งานที่ 1: การสร้างโปรแกรม AI ด้วย kNN 

**รายวิชา:** IN403103 Artificial Intelligence
**ชื่อกลุ่ม / สมาชิก:**
1. ____นายธนกร ทิพเนตร____ รหัสนักศึกษา __673450191-2__
2. _นายกิตติศักดิ์ ขันเเข็ง_ รหัสนักศึกษา _673450031-4_
3. _นางสาวพัชราภา สุขเดช_ รหัสนักศึกษา __673450398-0_


# 🏋️ KNN Powerlifting Skill Classification

> **โปรแกรม AI สำหรับทำนายระดับฝีมือนักกีฬายกน้ำหนักด้วย K-Nearest Neighbors (kNN)**

โปรเจกต์นี้เป็นส่วนหนึ่งของรายวิชา **IN403103 Artificial Intelligence**
มีวัตถุประสงค์เพื่อประยุกต์ใช้ Machine Learning Algorithm แบบ **K-Nearest Neighbors (kNN)** ในการจำแนกระดับฝีมือของนักกีฬายกน้ำหนักจากข้อมูลสถิติการแข่งขัน

โปรแกรมจะรับข้อมูลของนักกีฬา เช่น เพศ อายุ น้ำหนักตัว และน้ำหนักที่ยกได้ใน 3 ท่าหลัก ได้แก่ **Squat, Bench Press และ Deadlift** จากนั้นนำข้อมูลไปเปรียบเทียบกับนักกีฬาที่มีลักษณะใกล้เคียงกันใน Dataset และทำนายระดับฝีมือออกมาเป็น 5 ระดับ

### 🎯 ระดับฝีมือที่ระบบสามารถทำนายได้

| ระดับ               | ความหมาย       |
| ------------------- | -------------- |
| 🟢 **Beginner**     | ระดับเริ่มต้น  |
| 🔵 **Novice**       | ระดับพื้นฐาน   |
| 🟡 **Intermediate** | ระดับปานกลาง   |
| 🟠 **Advanced**     | ระดับสูง       |
| 🔴 **Elite**        | ระดับยอดเยี่ยม |

---

# 📌 1. แนวคิดของโครงงาน

ปัญหาที่ต้องการแก้คือ

> **"ถ้าเราทราบข้อมูลพื้นฐานและสถิติการยกน้ำหนักของนักกีฬาคนหนึ่ง เราสามารถใช้ Machine Learning ทำนายได้หรือไม่ว่านักกีฬาคนนั้นอยู่ในระดับใด?"**

ระบบจึงนำข้อมูลนักกีฬาจากการแข่งขันจริงมาใช้เป็น **Training Data**

เมื่อผู้ใช้กรอกข้อมูลนักกีฬาคนใหม่ ระบบจะ:

```text
ข้อมูลนักกีฬา
      ↓
ตรวจสอบข้อมูล
      ↓
Feature Scaling
      ↓
KNN Model
      ↓
ค้นหานักกีฬาที่ใกล้เคียงที่สุด k คน
      ↓
ดูระดับฝีมือของเพื่อนบ้าน
      ↓
เลือกกลุ่มที่พบมากที่สุด
      ↓
แสดงผลระดับฝีมือ
```

หลักการนี้เป็นแนวคิดพื้นฐานของ KNN ซึ่งจะจำแนกข้อมูลจากเพื่อนบ้านที่อยู่ใกล้กันมากที่สุด

---

# 🧠 2. K-Nearest Neighbors คืออะไร?

**K-Nearest Neighbors หรือ KNN** เป็น Machine Learning Algorithm สำหรับงาน Classification

หลักการง่าย ๆ คือ

> **"ข้อมูลใหม่มีแนวโน้มที่จะอยู่ในกลุ่มเดียวกับข้อมูลที่มีลักษณะใกล้เคียงกัน"**

ตัวอย่างเช่น ถ้าเราต้องการทำนายนักกีฬาคนใหม่ และกำหนด

```text
k = 5
```

ระบบจะหานักกีฬาที่มีข้อมูลใกล้เคียงกับนักกีฬาคนใหม่ที่สุด **5 คน**

สมมติพบว่า

```text
Beginner       1 คน
Novice         3 คน
Intermediate   1 คน
```

ระบบจะเลือก

```text
Novice
```

เพราะเป็นกลุ่มที่มีจำนวนมากที่สุด

### ทำไมต้องมีค่า k?

ค่า **k** คือจำนวนเพื่อนบ้านที่เราต้องการนำมาใช้ในการตัดสินใจ

ตัวอย่าง:

```text
k = 1   → ดูเพื่อนบ้านใกล้ที่สุด 1 คน
k = 3   → ดูเพื่อนบ้านใกล้ที่สุด 3 คน
k = 5   → ดูเพื่อนบ้านใกล้ที่สุด 5 คน
```

ในโปรแกรมนี้ผู้ใช้สามารถเลือกค่า **k ตั้งแต่ 1–25** ได้โดยตรงจาก GUI

---

# 📊 3. Dataset

ข้อมูลที่ใช้ในโครงงานมาจาก **OpenPowerlifting Dataset** ซึ่งเผยแพร่ผ่าน Kaggle

ข้อมูลต้นฉบับมีประมาณ

```text
386,414 records
```

หลังจากทำความสะอาดข้อมูลแล้ว เหลือประมาณ

```text
106,615 records
```

จากนั้นเลือกข้อมูลแบบสมดุลเพื่อใช้ในโปรแกรมจำนวน

```text
1,500 records
```

โดยแบ่งเป็น

```text
Beginner       300 records
Novice         300 records
Intermediate   300 records
Advanced       300 records
Elite          300 records
--------------------------------
รวม           1,500 records
```

การแบ่งให้แต่ละ Class มีจำนวนใกล้เคียงกันช่วยลดปัญหาที่โมเดลอาจเอนเอียงไปยัง Class ที่มีข้อมูลมากกว่า

---

# 🧹 4. การเตรียมข้อมูล

ไฟล์

```text
prepare_powerlifting_data.py
```

มีหน้าที่เตรียมข้อมูลก่อนนำไปใช้กับ KNN

ขั้นตอนหลักมีดังนี้

### Step 1 — อ่าน Dataset

อ่านเฉพาะข้อมูลที่จำเป็น ได้แก่

```text
Sex
Age
BodyweightKg
BestSquatKg
BestBenchKg
BestDeadliftKg
Wilks
```

### Step 2 — ลบข้อมูลที่ไม่สมบูรณ์

ข้อมูลที่มีค่า Missing จะถูกลบออก

```python
df = df.dropna()
```

### Step 3 — ตรวจสอบค่าการยกน้ำหนัก

ต้องมีค่า

```text
Squat > 0
Bench > 0
Deadlift > 0
```

เพราะค่าที่เป็น 0 หรือติดลบไม่เหมาะสำหรับใช้เป็นสถิติการยกที่สำเร็จ

### Step 4 — ตรวจสอบอายุและน้ำหนักตัว

กำหนดช่วงข้อมูลให้สมเหตุสมผล

```text
Age:          10–90 ปี
Bodyweight:   30–200 kg
```

### Step 5 — แปลงเพศเป็นตัวเลข

เนื่องจาก Machine Learning ต้องการข้อมูลในรูปแบบตัวเลข จึงกำหนด

```text
Male   = 1
Female = 0
```

### Step 6 — สร้าง Label

ระบบใช้ **Wilks Score** เพื่อแบ่งระดับฝีมือ

| Wilks Score | TIER         |
| ----------: | ------------ |
|       < 200 | Beginner     |
|     200–300 | Novice       |
|     300–400 | Intermediate |
|     400–500 | Advanced     |
|       ≥ 500 | Elite        |

การสร้าง Label ถูกทำในขั้นตอนเตรียม Dataset

---

# ⚠️ 5. ทำไมไม่ใช้ Wilks Score เป็น Feature?

นี่เป็นจุดสำคัญที่สามารถนำไปอธิบายตอนนำเสนอได้

เราใช้ **Wilks Score เพื่อสร้างคำตอบ (Label)**

แต่เรา **ไม่ได้เอา Wilks Score เข้าไปเป็น Feature**

เพราะถ้าทำแบบนี้

```text
Wilks Score → TIER
```

แล้วนำทั้ง Wilks Score และ TIER เข้า Model

โมเดลจะสามารถเห็นข้อมูลที่ใช้สร้างคำตอบอยู่แล้ว

เรียกว่า

> **Feature Leakage**

ดังนั้นโปรเจกต์จึงให้ KNN เรียนรู้จากข้อมูลจริง 6 Features ได้แก่

```text
SEX
AGE
BODYWEIGHT
SQUAT
BENCH
DEADLIFT
```

และให้

```text
TIER
```

เป็นสิ่งที่ต้องการให้โมเดลทำนาย

---

# 🔢 6. Features ที่ใช้ใน Model

โมเดลใช้ทั้งหมด **6 Features**

| Feature      | รายละเอียด                 | หน่วย |
| ------------ | -------------------------- | ----- |
| `SEX`        | เพศ                        | 0 / 1 |
| `AGE`        | อายุ                       | ปี    |
| `BODYWEIGHT` | น้ำหนักตัว                 | kg    |
| `SQUAT`      | น้ำหนัก Squat สูงสุด       | kg    |
| `BENCH`      | น้ำหนัก Bench Press สูงสุด | kg    |
| `DEADLIFT`   | น้ำหนัก Deadlift สูงสุด    | kg    |

ส่วนสิ่งที่ต้องการทำนายคือ

```text
TIER
```

ซึ่งมีทั้งหมด 5 Class

```text
Beginner
Novice
Intermediate
Advanced
Elite
```

---

# ⚖️ 7. Feature Scaling

ก่อนนำข้อมูลเข้า KNN โปรแกรมใช้

```python
StandardScaler()
```

เพื่อทำ **Feature Scaling**

เหตุผลคือแต่ละ Feature มีขนาดแตกต่างกันมาก

ตัวอย่างเช่น

```text
SEX          = 0 หรือ 1
AGE          = ประมาณ 20–50
BODYWEIGHT   = ประมาณ 50–150
SQUAT        = หลักร้อย
DEADLIFT     = หลักร้อย
```

ถ้าไม่ Scaling ตัวแปรที่มีค่ามากอาจมีอิทธิพลต่อการคำนวณระยะทางมากเกินไป

ดังนั้นโปรแกรมจึงแปลงข้อมูลให้อยู่ในมาตราส่วนที่เหมาะสมก่อนใช้ KNN

---

# 🤖 8. การทำงานของ Model

โปรแกรมใช้

```python
KNeighborsClassifier
```

จาก Library

```text
scikit-learn
```

ขั้นตอนการทำงานคือ

### 1. โหลด Dataset

```python
df = pd.read_csv("data.csv")
```

### 2. แยก Features และ Label

```python
X = df[FEATURE_COLS]
y = df["TIER"]
```

### 3. Scaling

```python
X_scaled = scaler.fit_transform(X)
```

### 4. สร้าง KNN

```python
model = KNeighborsClassifier(n_neighbors=k)
```

### 5. Train

```python
model.fit(X_scaled, y)
```

### 6. รับข้อมูลนักกีฬาคนใหม่

ตัวอย่าง

```text
SEX        = 1
AGE        = 25
BODYWEIGHT = 90
SQUAT      = 150
BENCH      = 100
DEADLIFT   = 180
```

### 7. Scaling ข้อมูลใหม่

ใช้ Scaler ตัวเดียวกับ Training Data

### 8. Predict

```python
model.predict(X_new_scaled)
```

### 9. แสดงผล

เช่น

```text
Intermediate
```

กระบวนการดังกล่าวอยู่ใน `knn_app.py` โดยตรง

---

# 🖥️ 9. ส่วนติดต่อผู้ใช้ (GUI)

โปรแกรมใช้

```text
Python + Tkinter
```

ในการสร้าง GUI

หน้าจอหลักแบ่งออกเป็น 3 ส่วน

## ① Training Data

ด้านซ้ายแสดง Dataset ที่ใช้ในการฝึกโมเดล

มีข้อมูล

```text
SEX
AGE
BODYWEIGHT
SQUAT
BENCH
DEADLIFT
TIER
```

และสามารถค้นหาข้อมูลตาม `TIER` ได้

---

## ② Input

ด้านขวาเป็นช่องสำหรับกรอกข้อมูลนักกีฬาคนใหม่

ผู้ใช้กรอก

```text
SEX
AGE
BODYWEIGHT
SQUAT
BENCH
DEADLIFT
```

และเลือกค่า

```text
k = 1–25
```

---

## ③ Predict

เมื่อกดปุ่ม

```text
Predict
```

โปรแกรมจะ

```text
ตรวจสอบ Input
      ↓
อ่านค่า k
      ↓
Train KNN ด้วย k ที่เลือก
      ↓
Scaling ข้อมูลใหม่
      ↓
Predict
      ↓
แสดง TIER
```

โค้ดจริงจะ Train Model ใหม่ทุกครั้งเมื่อผู้ใช้เปลี่ยนค่า `k` เพื่อให้ผลลัพธ์ตรงกับค่า k ที่เลือก

---

# 🛡️ 10. การตรวจสอบข้อมูล Input

โปรแกรมไม่ได้รับข้อมูลทุกอย่างโดยไม่ตรวจสอบ

ก่อน Predict ระบบจะตรวจสอบว่า

### SEX

ต้องเป็น

```text
0 หรือ 1
```

### AGE

ต้องอยู่ในช่วงที่กำหนด

```text
10–90 ปี
```

### BODYWEIGHT

ต้องอยู่ในช่วง

```text
30–200 kg
```

### Squat / Bench / Deadlift

ต้องอยู่ในช่วง

```text
0–500 kg
```

ถ้ากรอกข้อมูลไม่ถูกต้อง โปรแกรมจะแสดง Warning และไม่ทำการ Predict

---

# 🧪 11. การทดสอบ Model

โปรเจกต์มีไฟล์

```text
test_app.py
```

สำหรับตรวจสอบว่า Dataset และ Model สามารถทำงานได้ถูกต้อง

การทดสอบแบ่งเป็น 3 ส่วน

### Test 1 — ตรวจสอบ Dataset

ตรวจสอบว่า Dataset มีข้อมูลอย่างน้อย 500 แถว และมี Features ครบ

```text
SEX
AGE
BODYWEIGHT
SQUAT
BENCH
DEADLIFT
TIER
```

### Test 2 — ตรวจสอบ Accuracy

แบ่งข้อมูลเป็น

```text
80% Training
20% Testing
```

แล้วทดสอบค่า

```text
k = 1
k = 3
k = 5
k = 7
```

โดยใช้ `accuracy_score` วัดผลลัพธ์

ผลการทดสอบที่บันทึกไว้ในโครงงานคือ

|  k |  Accuracy |
| -: | --------: |
|  1 | **83.7%** |
|  3 | **86.0%** |
|  5 | **87.0%** |
|  7 | **85.0%** |

จากผลการทดลองนี้

> **k = 5 ให้ Accuracy สูงที่สุดในชุดการทดลองนี้ที่ 87.0%**

ดังนั้นถ้าพิจารณาเฉพาะค่า k ที่ทดสอบ `1, 3, 5, 7` ค่า **k = 5** ให้ผลดีที่สุด

---

# 🧪 12. ตัวอย่าง Unit Test

นอกจาก Accuracy แล้ว ยังมีการทดสอบกรณีตัวอย่าง

เช่นนักกีฬาที่

```text
น้ำหนักตัว = 90 kg
Squat      = 60 kg
Bench      = 40 kg
Deadlift   = 70 kg
```

ระบบคาดหวังว่าควรอยู่ในกลุ่ม

```text
Beginner หรือ Novice
```

จากนั้นตรวจสอบว่าผลที่ Model ทำนายอยู่ในสอง Class นี้หรือไม่

---

# 📁 13. โครงสร้างโปรเจกต์

```text
knn-app/
│
├── README.md
│
├── knn_app.py
│
├── prepare_powerlifting_data.py
│
├── test_app.py
│
├── data.csv
│
└── openpowerlifting.csv
```

### `knn_app.py`

โปรแกรมหลักของระบบ

หน้าที่:

```text
โหลด Dataset
สร้าง KNN Model
ทำ Feature Scaling
สร้าง GUI
รับ Input
Predict
แสดงผล
```

---

### `prepare_powerlifting_data.py`

ใช้สำหรับเตรียม Dataset

หน้าที่:

```text
อ่านข้อมูลดิบ
↓
Clean Data
↓
แปลงข้อมูล
↓
สร้าง TIER จาก Wilks
↓
เลือกข้อมูลแบบสมดุล
↓
สร้าง data.csv
```

---

### `data.csv`

Dataset ที่ผ่านการเตรียมข้อมูลแล้ว

ใช้เป็นข้อมูลสำหรับ KNN

```text
6 Features + 1 Label
```

---

### `test_app.py`

ใช้ทดสอบ

```text
Dataset
Model
Accuracy
Prediction
```

---

### `openpowerlifting.csv`

Dataset ต้นฉบับที่นำมาใช้เตรียมข้อมูล

---

# 🚀 14. วิธีติดตั้ง

ต้องติดตั้ง Python ก่อน

จากนั้นเปิด Terminal / Command Prompt ในโฟลเดอร์โปรเจกต์

ติดตั้ง Library ที่จำเป็น

```bash
pip install pandas scikit-learn
```

> Tkinter โดยทั่วไปมีมากับ Python บน Windows และ macOS หากใช้ Linux อาจต้องติดตั้ง `python3-tk` เพิ่ม

---

# ▶️ 15. วิธี Run โปรแกรม

ใช้คำสั่ง

```bash
python knn_app.py
```

จากนั้น GUI ของ KNN APP จะเปิดขึ้นมา

### ขั้นตอนการใช้งาน

**Step 1**

กรอกข้อมูล

```text
SEX
AGE
BODYWEIGHT
SQUAT
BENCH
DEADLIFT
```

**Step 2**

เลือกค่า

```text
k
```

เช่น

```text
k = 5
```

**Step 3**

กด

```text
Predict
```

**Step 4**

ระบบจะแสดงระดับฝีมือ เช่น

```text
Intermediate
```

---

# 🔄 16. ถ้าต้องการสร้าง Dataset ใหม่

สามารถแก้ไขข้อมูลหรือเกณฑ์ใน

```text
prepare_powerlifting_data.py
```

จากนั้นรัน

```bash
python prepare_powerlifting_data.py
```

โปรแกรมจะสร้าง

```text
data.csv
```

ใหม่จากข้อมูลต้นฉบับ

---

# 🧩 17. จุดเด่นของโปรเจกต์

### ✅ ใช้ข้อมูลการแข่งขันจริง

ไม่ได้สร้างข้อมูลขึ้นมาเอง แต่ใช้ Dataset จาก OpenPowerlifting

### ✅ มีการทำ Data Cleaning

มีการตรวจสอบ Missing Data, อายุ, น้ำหนักตัว และค่าสถิติการยก

### ✅ มี Feature Scaling

ช่วยให้ Features มีผลต่อระยะทางอย่างเหมาะสม

### ✅ Dataset มีความสมดุล

เลือกข้อมูลระดับละ 300 records รวม 1,500 records

### ✅ สามารถเปลี่ยนค่า k ได้

ผู้ใช้สามารถทดลองค่า k ต่าง ๆ ได้โดยตรงจาก GUI

### ✅ มี Input Validation

ลดปัญหาจากการกรอกข้อมูลผิดรูปแบบ

### ✅ มีการทดสอบ

มีทั้ง Accuracy Test และ Single Prediction Test

---

# ⚠️ 18. ข้อจำกัดของระบบ

แม้โมเดลจะมี Accuracy ค่อนข้างดี แต่ระบบยังมีข้อจำกัด

### 1. Dataset ที่ใช้ Train มีเพียง 1,500 records

แม้ Dataset ต้นฉบับจะมีข้อมูลจำนวนมาก แต่โปรเจกต์เลือกมาใช้เพียง 1,500 records เพื่อให้ Dataset สมดุลและเหมาะกับ GUI

### 2. KNN เป็น Distance-Based Algorithm

ผลลัพธ์ขึ้นอยู่กับระยะห่างของข้อมูล ดังนั้น Feature Scaling จึงมีความสำคัญ

### 3. ระดับ TIER อ้างอิงจาก Wilks Score

ดังนั้นการแบ่งระดับไม่ได้หมายความว่าเป็นมาตรฐานสากลสำหรับการจัดระดับนักกีฬาทุกประเภท แต่เป็นเกณฑ์ที่โปรเจกต์กำหนดจาก Wilks Score

### 4. ไม่มี Probability ของ Prediction

ระบบปัจจุบันแสดงเพียง Class ที่โมเดลทำนาย เช่น

```text
Advanced
```

ยังไม่ได้แสดงว่า

```text
Advanced = 80%
Intermediate = 15%
Elite = 5%
```

---

# 📈 19. ผลลัพธ์ของโครงงาน

จากการทดลองค่า k หลายค่า

```text
k = 1 → 83.7%
k = 3 → 86.0%
k = 5 → 87.0%
k = 7 → 85.0%
```

ค่า **k = 5** ให้ผลดีที่สุดในชุดการทดลองนี้

```text
Accuracy = 87.0%
```

จึงสามารถสรุปได้ว่า KNN สามารถนำข้อมูลสถิติของนักกีฬายกน้ำหนักมาใช้จำแนกระดับฝีมือได้ในระดับที่น่าพอใจสำหรับโครงงานนี้

---

# 🎤 20. สรุปสำหรับการนำเสนอ

สามารถอธิบายโปรเจกต์สั้น ๆ ได้ดังนี้:

> **"โปรเจกต์นี้เป็นโปรแกรม AI ที่ใช้ Algorithm K-Nearest Neighbors หรือ KNN เพื่อทำนายระดับฝีมือของนักกีฬายกน้ำหนัก โดยเราใช้ข้อมูลจาก OpenPowerlifting ซึ่งประกอบด้วยเพศ อายุ น้ำหนักตัว และสถิติ Squat, Bench Press และ Deadlift เป็น Features ส่วน Label คือระดับฝีมือ 5 ระดับ ได้แก่ Beginner, Novice, Intermediate, Advanced และ Elite ซึ่งสร้างจาก Wilks Score"**

> **"ก่อนนำข้อมูลเข้า Model เราทำ Data Cleaning และ Feature Scaling ด้วย StandardScaler เพื่อให้ Features ทุกตัวมีมาตราส่วนที่เหมาะสม จากนั้นใช้ KNN เปรียบเทียบข้อมูลนักกีฬาคนใหม่กับข้อมูลที่ใกล้เคียงที่สุดตามค่า k ที่ผู้ใช้เลือก"**

> **"โปรแกรมมี GUI ที่พัฒนาด้วย Tkinter สามารถกรอกข้อมูลนักกีฬา เลือกค่า k และกด Predict เพื่อดูระดับฝีมือได้ทันที นอกจากนี้ยังมีการทดสอบ Model โดยแบ่งข้อมูล 80% สำหรับ Train และ 20% สำหรับ Test โดยค่า k ที่ให้ผลดีที่สุดจากชุดที่ทดลองคือ k=5 มี Accuracy 87.0%"**

---

# 🎯 21. Flowchart ของระบบ

```text
                 ┌──────────────────────┐
                 │  OpenPowerlifting    │
                 │       Dataset        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Data Cleaning    │
                 │ Missing / Outlier    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Create TIER Label  │
                 │    from Wilks Score  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Balanced Dataset   │
                 │      1,500 rows      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    StandardScaler    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    KNN Classifier    │
                 │       k = 1–25       │
                 └──────────┬───────────┘
                            │
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
   ┌───────────────────┐        ┌───────────────────┐
   │ Training Dataset  │        │   New Athlete     │
   └───────────────────┘        └─────────┬─────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │   Input Validation│
                               └─────────┬─────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │      Predict      │
                               └─────────┬─────────┘
                                         │
                                         ▼
                              ┌────────────────────┐
                              │      TIER          │
                              │ Beginner → Elite   │
                              └────────────────────┘
```

---

# 📚 22. Technologies

| Technology               | ใช้สำหรับ           |
| ------------------------ | ------------------- |
| **Python**               | ภาษาหลัก            |
| **Pandas**               | จัดการ Dataset      |
| **Scikit-learn**         | Machine Learning    |
| **StandardScaler**       | Feature Scaling     |
| **KNeighborsClassifier** | KNN Model           |
| **Tkinter**              | สร้าง GUI           |
| **GitHub**               | จัดเก็บ Source Code |

---

# 👥 23. สมาชิกกลุ่ม

| ลำดับ | ชื่อ                   | รหัสนักศึกษา |
| ----: | ---------------------- | ------------ |
|     1 | นายธนกร ทิพเนตร        | 673450191-2  |
|     2 | นายกิตติศักดิ์ ขันแข็ง | 673450031-4  |
|     3 | นางสาวพัชราภา สุขเดช   | 673450398-0  |

---

# 🔗 24. Repository

Source Code ของโครงงาน:

[GitHub — tthp143/knn-app](https://github.com/tthp143/knn-app?utm_source=chatgpt.com)

Dataset อ้างอิงมาจาก OpenPowerlifting ผ่าน Kaggle ตามที่ระบุไว้ใน Repository

---

# 📝 25. สรุปภาพรวม

โปรเจกต์นี้แสดงให้เห็นการนำ **Machine Learning + Real-world Dataset + GUI Application** มารวมกันเป็นโปรแกรมที่สามารถใช้งานได้จริง

ตั้งแต่

```text
Raw Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering / Labeling
      ↓
Balanced Dataset
      ↓
Feature Scaling
      ↓
KNN Training
      ↓
Model Testing
      ↓
GUI Application
      ↓
Prediction
```

โดยผลการทดลองในชุดค่า k ที่ทดสอบพบว่า **k = 5 ให้ Accuracy สูงสุด 87.0%**

โปรเจกต์จึงเป็นตัวอย่างการประยุกต์ใช้ KNN สำหรับ **Classification Problem** โดยใช้ข้อมูลกีฬายกน้ำหนักเป็นกรณีศึกษา
