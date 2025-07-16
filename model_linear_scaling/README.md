#  Medical Insurance Cost Prediction – Linear Regression Approach

This project aims to predict **medical insurance charges** using a linear regression model on structured demographic and lifestyle data. The implementation includes data preprocessing, one-hot encoding, scaling, evaluation, and visualization.

---

##  Dataset

The dataset used: `insurance.csv`  
It includes the following columns:

- `age` (int): Age of the individual
- `sex` (object): Gender (male/female)
- `bmi` (float): Body mass index
- `children` (int): Number of dependents
- `smoker` (object): Smoking status (yes/no)
- `region` (object): Residential region
- `charges` (float): Insurance cost (target variable)

---

##  Exploratory Data Analysis (EDA)

- No missing values found.
- Pearson correlation:
  - `age`: 0.299 → moderate positive correlation
  - `bmi`: 0.198 → weak correlation
  - `children`: 0.068 → very weak correlation
- Visualizations:
  - KDE plots for `age`, `bmi`, and `charges`
  - Count plots for `sex`, `children`, `smoker`, `region`
  - Histograms and residual plots to understand the distribution

---

##  Data Preprocessing

### 🔹 Categorical Encoding
- Used `OneHotEncoder` manually (instead of pipeline)
- Applied to: `sex`, `smoker`, `region`

### 🔹 Scaling
- Compared `StandardScaler` and `MinMaxScaler`
- Final model used `StandardScaler`

---

##  Modeling

- Applied `LinearRegression` on scaled features
- Compared results using both scalers (same accuracy in this case)
- Model trained using `train_test_split (test_size=0.2)`

---

##  Evaluation

- **R² Score**: `~0.78` on test set
- **MSE**: Reported for both train and test sets
- Evaluation included:
  - Predicted vs Actual plots
  - Residual distribution
  - Density comparison of true vs predicted charges

---

##  Prediction Example

Given a new input:
```json
{
  "age": 30,
  "sex": "male",
  "bmi": 29.5,
  "children": 2,
  "smoker": "yes",
  "region": "southeast"
}

---

# توقع تكلفة التأمين الطبي – باستخدام الانحدار الخطي (Linear Regression)

يهدف هذا المشروع إلى التنبؤ بـ **تكاليف التأمين الطبي** باستخدام نموذج الانحدار الخطي، بناءً على بيانات منظمة تشمل معلومات ديموغرافية ونمط الحياة. يشمل التنفيذ مراحل المعالجة المسبقة للبيانات، الترميز، التحجيم، التقييم، والتصور البياني.

---

## مجموعة البيانات

البيانات المستخدمة من الملف: `insurance.csv`
وتتضمن الأعمدة التالية:

* `age` (عدد صحيح): عمر الشخص
* `sex` (كائن نصي): الجنس (ذكر/أنثى)
* `bmi` (عدد عشري): مؤشر كتلة الجسم
* `children` (عدد صحيح): عدد المعالين
* `smoker` (نصي): حالة التدخين (نعم/لا)
* `region` (نصي): المنطقة السكنية
* `charges` (عدد عشري): تكلفة التأمين (المتغير الهدف)

---

## تحليل البيانات الاستكشافي (EDA)

* لم يتم العثور على أي **قيم مفقودة**
* الارتباط (Pearson Correlation):

  * `age`: 0.299 → ارتباط موجب متوسط
  * `bmi`: 0.198 → ارتباط ضعيف
  * `children`: 0.068 → ارتباط ضعيف جدًا
* التصورات البيانية:

  * منحنيات الكثافة الاحتمالية (KDE) للخصائص: `age`, `bmi`, `charges`
  * رسوم عدّ (Count Plots) لـ: `sex`, `children`, `smoker`, `region`
  * هيستوغرامات ورسوم البواقي لفهم التوزيع

---

## المعالجة المسبقة للبيانات

### 🔹 ترميز الخصائص التصنيفية

* تم استخدام `OneHotEncoder` يدويًا (دون استخدام Pipeline)
* تم تطبيقه على الخصائص: `sex`, `smoker`, `region`

### 🔹 التحجيم (Scaling)

* تمت مقارنة طريقتين: `StandardScaler` و `MinMaxScaler`
* تم اعتماد `StandardScaler` في النموذج النهائي

---

## النمذجة

* تم تطبيق نموذج `LinearRegression` على الخصائص بعد التحجيم
* تمت مقارنة النتائج باستخدام كلا المحولين (وكانت النتائج متقاربة في الدقة)
* تم تدريب النموذج باستخدام `train_test_split` بنسبة اختبار 20% (`test_size=0.2`)

---

## التقييم

* **معامل R²**: حوالي `0.78` على مجموعة الاختبار
* **متوسط الخطأ التربيعي (MSE)**: تم حسابه لكل من التدريب والاختبار
* شمل التقييم:

  * رسم مقارنة بين القيم المتوقعة والحقيقية
  * توزيع البواقي
  * مقارنة الكثافة الاحتمالية بين القيم الحقيقية والمتوقعة لتكاليف التأمين

---

## مثال على التنبؤ

عند إدخال البيانات التالية:

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

---


