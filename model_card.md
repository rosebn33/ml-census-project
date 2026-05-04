# Model Card

For more background on model cards, see:
https://arxiv.org/pdf/1810.03993.pdf

---

## Model Details

This project uses a **Logistic Regression** model to predict whether a person earns **more than $50,000 per year** based on U.S. Census data.  
The model works with structured, tabular data and was built using scikit‑learn.

Before training, the data was cleaned and prepared by:
- One‑hot encoding categorical features
- Imputing missing numerical values
- Splitting the data into training and test sets

---

## Intended Use

This model is intended **only for learning and demonstration purposes**.  
It shows how to build a complete machine‑learning pipeline, including preprocessing, training, evaluation, testing, deployment with an API, and CI/CD.

The model **should not be used in real‑world decision‑making**, such as hiring, lending, or any situation where predictions could impact people’s lives.

---

## Training Data

The model was trained on the **UCI Census Income (Adult) dataset**, which contains demographic and employment‑related information such as age, education level, occupation, workclass, and hours worked per week.

The dataset reflects historical data and may contain social and economic biases that are present in real‑world data.

---

## Evaluation Data

To evaluate performance, the dataset was split into:
- **80% training data**
- **20% test data**

The test data was not used during training and was only used to measure how well the model performs on unseen examples.

---

## Metrics

The model was evaluated using the following metrics:

- **Precision**
- **Recall**
- **F1 Score**

On the test dataset, the model achieved approximately:

- **Precision:** 0.73  
- **Recall:** 0.56  
- **F1 Score:** 0.64  

In addition to overall performance, the model was evaluated on **slices of the data based on categorical features** (such as workclass and sex).  
These slice‑level results are saved in the file `slice_output.txt`.

---

## Ethical Considerations

This model was trained using U.S. Census data that includes sensitive demographic information such as sex, race, marital status, and occupation. Because income is closely connected to social and economic inequalities, a model trained on this data may reflect existing biases present in the historical data.

There is a risk that predictions made by this model could perform differently for certain demographic groups. For example, factors like education level or occupation may be influenced by unequal access to opportunities, which the model does not account for. This model has not been specifically tested or adjusted to ensure fairness across all groups.

For these reasons, this model should not be used to make real‑world decisions that could impact individuals, such as hiring, lending, or eligibility determinations. Its purpose is strictly educational, to demonstrate machine learning concepts rather than to support decision‑making about people.
