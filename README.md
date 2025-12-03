# mlops-ml-pipeline-lab


## **MLOps – Practical Exam**

Follow the steps below to complete this exam.

**Best practice:** Commit and push **your code** to your repository **after completing each step** to ensure proper version control and avoid losing work.


---

## **1️⃣ Fork the repository**

Fork the original repository into your own GitHub account.

---

## **2️⃣ Clone your fork**

Clone **your** repository:

```bash
git clone <URL_OF_YOUR_FORK>
```

---

## **3️⃣ Navigate into the project**

```bash
cd mlops-ml-pipeline-lab
```

---

## **4️⃣ Create and activate the virtual environment**

Create the environment:

```bash
make env_update
```

Activate it:

```bash
conda activate ml_env
```

---

## **5️⃣ Complete the notebook**

Open and complete:

```
notebook/ml_houseprice_prediction.ipynb
```

Fill in all missing code sections.

---

## **6️⃣ Complete and run the three scripts**

### **a) Data Preprocessing**

Path:

```
ml_houseprice_prediction/src/ml_houseprice_prediction/data_preprocessing/preprocessing.py
```

Run the script with the appropriate arguments.

Example:

```bash
python preprocessing.py --input_data_path ../../../../datastores/raw_data/housing.csv --output_data_filename clean_housing.csv
```


---

### **b) Data Splits**

Path:

```
ml_houseprice_prediction/src/ml_houseprice_prediction/data_splits/splits.py
```

Run the script with the appropriate arguments.

Example:

```bash
python splits.py --input_data_path ../../../../datastores/clean_data/clean_housing.csv 
```


---

### **c) Model Training**

Path:

```
ml_houseprice_prediction/src/ml_houseprice_prediction/train_model/train.py
```

Run the script with the appropriate arguments.

Example:

```bash
python train.py \
  --input_train_data ../../../../datastores/splits_data/train_data.csv \
  --input_test_data ../../../../datastores/splits_data/test_data.csv \
  --model_filename LinearRegression.joblib
```

---

## **7️⃣ Complete the Makefile**

Add all required commands so the project can be automated.

---

## **8️⃣ Run the full pipeline**

Execute all steps with one command:

```bash
make pipeline
```

---

## **Final Step**

Commit and push **all your changes (your code)** to your repository.

Then email me the link to your GitHub repository.

