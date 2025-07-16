
### `README.md`

#  Medical Insurance Cost Prediction – Polynomial & Ridge Regression Model

This project implements a robust machine learning pipeline to predict **medical insurance costs** using **Polynomial Regression** with **Ridge Regularization**, enhanced by **Cross-Validation** and **GridSearchCV** for hyperparameter tuning.

---

##  Dataset Overview

- Source: `insurance.csv`
- Rows: 1338 samples
- Features:
  - `age` — Age of the insured person
  - `sex` — Gender (`male`, `female`)
  - `bmi` — Body Mass Index
  - `children` — Number of children/dependents
  - `smoker` — Smoking status (`yes`, `no`)
  - `region` — Residential region
  - `charges` — **Target** variable (Medical insurance charges)

---

##  Exploratory Data Analysis (EDA)

- Dataset contains **no missing values**
- Used `.info()` and `.describe()` to inspect data types and summary statistics
- Checked balance across categorical variables (sex, region, smoker)

---

##  Data Preprocessing

### 🔹 Feature Categorization
- **Numerical**: `age`, `bmi`, `children` → Scaled using `StandardScaler`
- **Categorical**: `sex`, `smoker`, `region` → Encoded using `OneHotEncoder` (drop='first' to avoid multicollinearity)

### 🔹 Pipeline Design
Used `ColumnTransformer` to apply transformations in a unified pipeline. This ensures clean preprocessing integration.

---

##  Modeling Strategy

### Step 1: Polynomial Regression
- Created polynomial features (degrees 1 to 5)
- Evaluated `Train MSE` and `Test MSE` for each degree
- **Overfitting** observed with higher degrees; **underfitting** with degree 1
- Degree **2** selected as optimal based on MSE curves

### Step 2: Cross-Validation
- Applied `cross_val_score` with 5-fold CV
- Obtained average R² performance:
  ```python
  Average R2 ≈ 0.86 (varies slightly depending on random seed)
````

### Step 3: Ridge Regression

* Replaced `LinearRegression` with `Ridge(alpha=1.0)`
* Reduced variance and improved generalization

### Step 4: GridSearchCV

* Tuned `alpha` in Ridge using:

  ```python
  param_grid = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}
  ```
* Selected **best alpha** and validated performance

---

##  Model Evaluation

### Final Metrics on Test Set

* **R² Score**: \~0.86 (after Ridge + best alpha)
* **Mean Squared Error (MSE)**: Reported on both train/test
* Predictions closely aligned with actual values

---

##  Prediction Example

New user data:

```json
{
  "age": 30,
  "sex": "male",
  "bmi": 29.5,
  "children": 2,
  "smoker": "yes",
  "region": "southeast"
}
```

### Output:

*  Predicted Charges (USD): e.g. `$32,800.00`
*  Approx. in Yemeni Rial: `19,680,000 ريال` (using 1 USD = 600 YER)

---

##  Visualizations

* ✅ **Polynomial Degree vs Error Curve**
* ✅ **Actual vs Predicted** for Train/Test
* ✅ **Residual Histograms**
* ✅ **Distribution KDE** of Actual vs Predicted

Each visualization helps assess:

* Model fit quality
* Overfitting patterns
* Bias/variance trade-off

---

##  Key Learnings

* **Polynomial regression** improves accuracy by modeling non-linearity.
* **Ridge regularization** reduces overfitting in high-degree polynomials.
* **GridSearchCV** automates hyperparameter tuning (like alpha).

---

##  Tools Used

* Python 3
* Pandas, NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook

---

##  Future Improvements

* Try **XGBoost** or **Random Forests** for feature interaction modeling
* Apply **log transformation** to skewed variables (e.g. `charges`)
* Tune additional hyperparameters like `degree`, `cv`, `features`



---

### `README.md`

# توقع تكلفة التأمين الطبي – نموذج الانحدار المتعدد والريج (Polynomial & Ridge Regression)

ينفذ هذا المشروع سلسلة متكاملة من خطوات تعلم الآلة للتنبؤ بـ **تكاليف التأمين الطبي** باستخدام **الانحدار المتعدد (Polynomial Regression)** مع **تنظيم Ridge**، ويعزز ذلك باستخدام **التحقق المتقاطع (Cross-Validation)** و **GridSearchCV** لضبط المعاملات الفائقة (Hyperparameters).

---

## نظرة عامة على البيانات

* المصدر: `insurance.csv`
* عدد العينات: 1338 صفًا
* الخصائص:

  * `age` — عمر الشخص المؤمن عليه
  * `sex` — الجنس (`male`, `female`)
  * `bmi` — مؤشر كتلة الجسم
  * `children` — عدد الأطفال أو المعالين
  * `smoker` — حالة التدخين (`yes`, `no`)
  * `region` — المنطقة السكنية
  * `charges` — **المتغير الهدف** (تكاليف التأمين الطبي)

---

## تحليل البيانات الاستكشافي (EDA)

* لا تحتوي البيانات على **قيم مفقودة**
* تم استخدام `.info()` و `.describe()` لفحص أنواع البيانات والإحصائيات العامة
* تم تحليل التوزيع المتوازن للمتغيرات التصنيفية مثل الجنس، المنطقة، وحالة التدخين

---

## المعالجة المسبقة للبيانات

### 🔹 تصنيف الخصائص

* **رقمية**: `age`, `bmi`, `children` → تمت معالجتها باستخدام `StandardScaler`
* **تصنيفية**: `sex`, `smoker`, `region` → تم ترميزها باستخدام `OneHotEncoder` مع `drop='first'` لتجنب التعدد الخطي

### 🔹 تصميم خطوط المعالجة (Pipeline)

تم استخدام `ColumnTransformer` لتطبيق جميع التحويلات ضمن خط معالجة موحد لضمان تكامل عمليات المعالجة المسبقة بشكل نظيف وفعّال.

---

## استراتيجية النمذجة

### الخطوة 1: الانحدار المتعدد (Polynomial Regression)

* إنشاء خصائص متعددة الحدود بدرجات من 1 إلى 5
* حساب متوسط الخطأ التربيعي للمربعات (MSE) للتدريب والاختبار
* تم ملاحظة **الإفراط في التعلّم (Overfitting)** في الدرجات العالية، و**نقص التعلّم (Underfitting)** في الدرجة 1
* تم اختيار الدرجة **2** كأفضل درجة بناءً على منحنيات MSE

### الخطوة 2: التحقق المتقاطع (Cross-Validation)

* استخدام `cross_val_score` مع التحقق المتقاطع بخمس طيات (5-fold CV)
* الأداء المتوسط لمعامل R²:

  ```python
  متوسط R2 ≈ 0.86 (قد يختلف قليلًا حسب العشوائية)
  ```

### الخطوة 3: استخدام Ridge Regression

* تم استبدال `LinearRegression` بـ `Ridge(alpha=1.0)`
* قلل من التباين وحسن من التعميم

### الخطوة 4: ضبط المعامل باستخدام GridSearchCV

* تم ضبط معامل `alpha` في Ridge باستخدام:

  ```python
  param_grid = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}
  ```
* تم اختيار **أفضل قيمة لـ alpha** وتأكيد الأداء

---

## تقييم النموذج

### القياسات النهائية على مجموعة الاختبار:

* **معامل R²**: \~0.86 (بعد استخدام Ridge + أفضل alpha)
* **متوسط الخطأ التربيعي (MSE)**: تم الإبلاغ عنه لكل من مجموعة التدريب والاختبار
* التوقعات كانت متقاربة جدًا من القيم الحقيقية

---

## مثال على التنبؤ

بيانات مستخدم جديد:

```json
{
  "age": 30,
  "sex": "male",
  "bmi": 29.5,
  "children": 2,
  "smoker": "yes",
  "region": "southeast"
}
```

### النتيجة:

* التكاليف المتوقعة (بالدولار الأمريكي): مثلًا `$32,800.00`
* ما يعادل بالريال اليمني تقريبًا: `19,680,000 ريال` (بسعر صرف 1 دولار = 600 ريال)

---

## التصورات البيانية

* ✅ **منحنى الخطأ مقابل درجة الانحدار**
* ✅ **المقارنة بين القيم الحقيقية والمتوقعة** للتدريب والاختبار
* ✅ **رسوم بيانية لبواقي الأخطاء (Residuals)**
* ✅ **مقارنة التوزيع الاحتمالي (KDE)** للقيم الحقيقية والمتوقعة

كل تصور يساعد في تقييم:

* جودة ملاءمة النموذج
* أنماط الإفراط أو النقص في التعلّم
* المفاضلة بين التحيّز والتباين

---

## أهم الدروس المستفادة

* **الانحدار المتعدد** يساعد على تحسين الدقة من خلال تمثيل العلاقات غير الخطية
* **تنظيم Ridge** يقلل من الإفراط في التعلّم خصوصًا في النماذج عالية التعقيد
* **GridSearchCV** يوفر وسيلة تلقائية لضبط المعاملات الفائقة مثل `alpha`

---

## الأدوات المستخدمة

* Python 3
* Pandas, NumPy
* Scikit-learn
* Matplotlib و Seaborn
* Jupyter Notebook

---

## تحسينات مستقبلية مقترحة

* تجربة **XGBoost** أو **Random Forests** لنمذجة التفاعلات بين الخصائص
* تطبيق **تحويل لوغاريتمي** للمتغيرات المنحرفة مثل `charges`
* ضبط معاملات إضافية مثل `degree`, `cv`, وخصائص الإدخال المختارة

---


