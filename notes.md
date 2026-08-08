# Module 1: Exploratory Data Analysis (EDA) Notes

## Data Overview
- **Dataset Shape:** 40 rows, 7 columns
- **Columns:** `title`, `price`, `availability`, `raiting`, `description`, `category`, `url`

## Answers to EDA Questions
1. **Target Column:** `price` (Needs cleaning as it contains currency symbols like `Â£`).
2. **Data Types:**
   - Categorical/Text: `title`, `availability`, `raiting`, `description`, `category`, `url`
   - Numeric: None in raw form (Price is currently object/string type).
3. **Missing Data:** 0 missing values across all columns.
4. **Junk/Data Quality Issues:**
   - `price` has encoding issue with currency character (`Â£51.77`).
   - `raiting` column name has a typo and uses word ratings (`One`, `Two`, `Three`, `Four`, `Five`) instead of numbers.
   - `availability` contains text with embedded numbers (e.g., `In stock (22 available)`).
   ## Module 4: Baseline Model Performance
- **Model:** Linear Regression
- **Baseline MAE:** 15.01
- **Baseline R²:** 0.05
## Module 5: Advanced Model Performance
- **Model:** Random Forest Regressor
- **Random Forest MAE:** 10.51
- **Random Forest R²:** 0.19
- **Conclusion:** Random Forest performed better than Linear Regression baseline.