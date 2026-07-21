1. Problem Statement
   Universities across Canada face an increasingly competitive global market when attracting high-quality graduate business students. Relying on intuition or outdated pricing models leaves institutions vulnerable to misaligned tuition fees, reduced student intake, and uncompetitive program offerings. Specifically, the University of Windsor needs a clear, data-driven framework to understand what truly makes graduate business programs attractive to prospective students and how to price and position its offerings against rival Canadian universities.

2. Business Value of the Analysis
   This predictive machine learning pipeline provides leadership at the University of Windsor with an objective benchmark of its graduate business programs. By quantifying how program features (such as reputation, salary outcomes, and program length) influence tuition and overall program standing, decision-makers can move beyond assumptions. The model pinpoint where Windsor excels, where performance gaps exist, and how to make targeted structural improvements to maximize ROI and student enrollment.

3. ETL Pipeline & Data Preprocessing
	Automated Extraction: Ingests program-level data from Canadian business schools, standardizing metadata, tuition rates, rankings, and employment outcomes.
	Data Cleaning & Sanitization: Automatically cleans currency symbols, commas, and string artifacts. Missing numerical data is imputed using median values, missing categories are filled, and extreme outliers are clipped (1st to 99th percentiles).
	Leakage Prevention: Dynamically strips specific identifier columns (e.g., URLs, program names, report labels) and cost-derived scores from training features (X) to prevent data leakage.
	Pipeline Processing: Features pass through standard scaling (for numeric variables) and dynamic one-hot encoding (for categorical variables) within a unified scikit-learn ColumnTransformer.

4. Model Development & Evaluation
	XGBoost Regression: An XGBRegressor model was trained to predict international tuition levels and evaluate overall program positioning.
	Aggregated Feature Importance: Individual categorical dummies (e.g., specific provinces) are aggregated back into parent columns so key drivers—such as QS Rank Standing, Average Salary, Program Duration, and Employment Rate are represented clearly.
	Robust Validation: Validated using an 80/20 train-test split alongside 5-fold cross-validation to prevent overfitting and guarantee reliable predictions.

5. Model Evaluation
  Regression Performance
  Metric	Result	Interpretation
  R² Score	0.334	The model explains approximately 33.4% of the variation in tuition fees, indicating moderate predictive performance.
  MAE	CAD 7,197.75	On average, tuition predictions differ from actual tuition by approximately CAD 7.2K.
  RMSE	CAD 12,194.21	Larger prediction errors exist for some programs, suggesting the presence of pricing variability and outliers.

6. Recommendations
   a. Business Recommendations
      Strengthen program reputation through faculty achievements, research, and industry partnerships to improve perceived quality.
      Promote Windsor's competitive tuition as a high-value, affordable option with strong graduate outcomes.
      Expand co-op, internships, and industry projects to enhance career readiness and student attraction.
      Introduce industry-recognized certifications (e.g., PMI, CHRP) and flexible program pathways to increase program competitiveness.
   b. Model Recommendations
      Expand the dataset by including more universities, programs, and historical tuition data.
      Incorporate additional features such as student demand, scholarships, cost of living, visa policies, and student satisfaction.
      Improve model performance through hyperparameter tuning and comparison with models such as Random Forest, LightGBM, and CatBoost.
      Retrain and validate the model regularly using updated data to improve prediction accuracy and generalizability.

7. Limitations
  The model was trained on a relatively small dataset (105 graduate business programs), which may limit its generalizability.
  Program quality was primarily represented using QS Rankings, which may not capture all dimensions of institutional reputation.
  Important external factors such as student preferences, geographic location, visa policies, scholarships, immigration opportunities, and economic conditions were not included in the analysis.
  The model predicts tuition using observable program characteristics but cannot account for strategic institutional pricing decisions or rapidly changing market conditions.
