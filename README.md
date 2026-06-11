# SVM Classification Application - Iris Flower Classification

This project demonstrates how to build a **Support Vector Machine (SVM) classification model** using the classic Iris flower dataset. The goal is to classify Iris flowers into one of three species based on four numeric flower measurements.

![Iris Flower Types](iris_types.jpg)

## Project Overview

The notebook trains and evaluates multiple SVM classifiers using different kernel functions. It compares how well each kernel separates the Iris species using sepal and petal measurements.

The three target classes are:

- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

The project is useful for learning the basics of:

- Supervised machine learning
- Multi-class classification
- Support Vector Machines
- Feature scaling
- Train/test splitting
- Confusion matrix evaluation
- Accuracy comparison across SVM kernels

## Dataset

The project uses the `iris.csv` dataset.

### Dataset Columns

| Column | Description |
|---|---|
| `sepal_length` | Sepal length measurement |
| `sepal_width` | Sepal width measurement |
| `petal_length` | Petal length measurement |
| `petal_width` | Petal width measurement |
| `species` | Target flower class |

### Sample Records

| sepal_length | sepal_width | petal_length | petal_width | species |
|---:|---:|---:|---:|---|
| 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 4.9 | 3.0 | 1.4 | 0.2 | setosa |
| 4.7 | 3.2 | 1.3 | 0.2 | setosa |
| 4.6 | 3.1 | 1.5 | 0.2 | setosa |
| 5.0 | 3.6 | 1.4 | 0.2 | setosa |

## Project Files

```text
.
├── SVM classification application.ipynb   # Main Jupyter Notebook
├── iris.csv                              # Iris dataset
├── iris_types.jpg                        # Image showing Iris flower classes
└── README.md                             # Project documentation
```

## Machine Learning Workflow

The notebook follows this workflow:

1. **Import required libraries**
   - NumPy
   - Pandas
   - Matplotlib
   - Scikit-learn

2. **Load the dataset**
   - Reads the `iris.csv` file using Pandas.

3. **Inspect the dataset**
   - Displays the first few records using `dataset.head()`.

4. **Display flower image**
   - Loads and displays `iris_types.jpg` to show the three Iris species.

5. **Split features and target**
   - Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
   - Target: `species`

6. **Split data into training and testing sets**
   - 80% training data
   - 20% testing data
   - Random state: `82`

7. **Apply feature scaling**
   - Uses `StandardScaler` to normalize feature values.
   - Scaling is important for SVM because distance-based calculations are affected by feature magnitude.

8. **Train SVM models**
   - Linear kernel
   - Polynomial kernel
   - RBF kernel
   - Sigmoid kernel

9. **Evaluate model performance**
   - Generates predictions on the test set.
   - Compares actual vs predicted values.
   - Builds confusion matrices.
   - Calculates accuracy for each kernel.

## SVM Kernels Used

### 1. Linear Kernel

The linear kernel is used when the data can be separated using a straight line or hyperplane.

```python
SVC(kernel='linear', random_state=0)
```

### 2. Polynomial Kernel

The polynomial kernel is useful when the relationship between features and classes is non-linear.

```python
SVC(kernel='poly', random_state=0)
```

### 3. RBF Kernel

The Radial Basis Function kernel is commonly used for non-linear classification problems. It can create complex decision boundaries.

```python
SVC(kernel='rbf', random_state=0)
```

### 4. Sigmoid Kernel

The sigmoid kernel behaves similarly to a neural network activation function in some scenarios.

```python
SVC(kernel='sigmoid', random_state=0)
```

## Model Performance

Using the provided dataset and the notebook's train/test split, the following results are produced in a clean run:

| Kernel | Correct Predictions | False Predictions | Accuracy |
|---|---:|---:|---:|
| Linear | 29 | 1 | 96.67% |
| Polynomial | 28 | 2 | 93.33% |
| RBF | 28 | 2 | 93.33% |
| Sigmoid | 25 | 5 | 83.33% |

### Best Performing Model

The **Linear SVM kernel** gives the highest accuracy in this project:

```text
Accuracy: 96.67%
```

This suggests that the Iris dataset is mostly linearly separable using the selected features.

## Confusion Matrix Example

For the linear kernel, the confusion matrix is:

```text
[[11  0  0]
 [ 0  8  1]
 [ 0  0 10]]
```

This means:

- 11 Setosa samples were correctly classified.
- 8 Versicolor samples were correctly classified.
- 1 Versicolor sample was incorrectly classified as Virginica.
- 10 Virginica samples were correctly classified.

## Requirements

Install the following Python packages before running the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Or create a `requirements.txt` file with:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Then install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run the Project

### Option 1: Run with Jupyter Notebook

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
```

2. Move into the project folder:

```bash
cd your-repository-name
```

3. Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open the notebook:

```text
SVM classification application.ipynb
```

6. Run all cells from top to bottom.

### Option 2: Run with JupyterLab

```bash
jupyter lab
```

Then open the notebook file and execute all cells.

## Important Notes

- Keep `iris.csv` in the same folder as the notebook.
- Keep `iris_types.jpg` in the same folder as the notebook if you want the image cell to display correctly.
- The notebook uses `random_state=82` for train/test splitting, so the output should be reproducible.
- Results may vary slightly if package versions or model parameters are changed.

## Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Support Vector Machine Classification**

## Key Learning Outcomes

After completing this project, you should understand how to:

- Load a dataset using Pandas.
- Separate independent variables and target labels.
- Split data into training and testing sets.
- Apply feature scaling using `StandardScaler`.
- Train SVM classification models using different kernels.
- Evaluate models using confusion matrices.
- Compare model accuracy across multiple kernels.

## Possible Improvements

This project can be improved by adding:

- Exploratory Data Analysis graphs
- Pair plots for feature relationships
- Classification report with precision, recall, and F1-score
- Cross-validation
- Hyperparameter tuning using `GridSearchCV`
- Decision boundary visualization
- Model export using `joblib` or `pickle`
- A simple web app using Flask, FastAPI, or Streamlit

## Example Future Enhancement

A future version could include a simple prediction function:

```python
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    sample_scaled = sc.transform(sample)
    prediction = svcclassifier.predict(sample_scaled)
    return prediction[0]
```

Example usage:

```python
predict_iris_species(5.1, 3.5, 1.4, 0.2)
```

Expected output:

```text
setosa
```

## Repository Purpose

This repository is intended for educational and portfolio purposes. It shows a clear machine learning workflow from dataset loading to model evaluation using Support Vector Machine classification.

## Author

**Anas Mohammed**

## License

This project can be used for learning and educational purposes. Add a license file such as `MIT License` if you want to make the repository publicly reusable.
