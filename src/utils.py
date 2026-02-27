# ============================================================================= #
# >>> Module of functions and classes for creating testing data.                #                                        
# ============================================================================= #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
# Data manipulation and testing:
# Pandas
import pandas as pd
# Scipy
from scipy.stats import mannwhitneyu, pointbiserialr, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# Numpy
import numpy as np

# ======================================================== #
# Optimize Dtypes - Functions                              #
# ======================================================== #
def optimize_dtypes(
    df: pd.DataFrame,
):
    """
    Adjusting the variable types with their respective characteristics. In this data, there are both binary and ordinal variables; I will be adjusting them so that there is no invalid statistical aggregation in the analyses.
    """
    # Copy DF
    df_clean = df.copy()

    # Categorical Columns
    categorical_cols = ['custcat',]

    # Interger Columns
    integer_cols =  ['tenure', 'age', 'address', 'employ', 'ed',]

    # Binary Columns
    binary_cols = [
        'equip','callcard', 'wireless','ebill', 'voice', 'pager',
        'internet', 'callwait', 'confer','churn'
    ]

    # Continus Columns
    continuos_cols = [
        'income',  'longmon', 'tollmon', 'equipmon', 'cardmon',
        'wiremon',
    ]

    # Categorical Columns
    df_clean[categorical_cols] = df_clean[categorical_cols].astype('int32')
    df_clean[categorical_cols] = df_clean[categorical_cols].astype('category')

    # Interger Columns
    df_clean[integer_cols] = df_clean[integer_cols].astype('Int32')

    # Binary Columns
    df_clean[binary_cols] = df_clean[binary_cols].astype('int8')

    # Continus Columns
    df_clean[continuos_cols] = df_clean[continuos_cols].astype('float32')

    return df_clean


# ======================================================== #
# EDATest - Class                                          #
# ======================================================== #
class EDATest:

# Initialize Class
    def __init__(
        self, 
        data: pd.DataFrame,
    ):
        try:
            # Entry checks
            if data.empty:
                raise ValueError('The provided DataFrame is empty.')

            if data.isnull().any().any():
                raise ValueError('The DataFrame contains null data.')

            self.data = data

        except Exception  as e:
                print(f'[Error] Failed to load Dataframe : {str(e)}')

    # ======================================================== #
    # Mannwhitney U Test - Function                            #
    # ======================================================== #
    def mannwhitney_u_test(
        self,
        audit_vars: list,
        target: str,
    ):
        
        """
        Performs a complete statistical audit to assess the predictive power of numerical variables in relation to a binary target variable.

        This method performs three main analyses for each variable:

        1. Mann-Whitney U Test (Non-parametric): Checks if the distributions of the two groups (e.g., Churn vs. Non-Churn) are statistically different. It is robust against outliers and does not assume a normal distribution.

        2. Point-Biserial Correlation: Measures the strength and direction of the linear association between the numerical variable and the binary target.

        3. Median Lift (Impact): Calculates the percentage change (%) of the median between the majority group (0/No) and the minority group (1/Yes) to measure the practical "effect size" in the business.

        The result is automatically classified based on the p-value:

        - ⭐⭐⭐ Strong (P < 0.001): Very high difference, variable Strong predictor.

        - ⭐⭐ Medium (P < 0.05): Significant difference.

        - ❌ Noise (P >= 0.05): Probably noise, without statistical separation power.

        Args:

        audit_vars (list): List of strings with the names of the numeric columns to be tested.

        target (str): Name of the target column (must be binary: 0/1, True/False, Yes/No).

        Returns:

        None: The function directly displays a DataFrame (display) ordered by the magnitude of the absolute correlation.
        """
        try:
            results_audit = []
            
            # Binary Target Validation
            unique_targets = self.data[target].dropna().unique()
            if len(unique_targets) != 2:
                print(f"[Warning] Target '{target}' is not binary (found {len(unique_targets)} classes). Skipping test.")
                return

            # Assumes that the "Positive" (Churn) class is the largest (1) or the second class found
            # If numeric, takes the max. If string, tries to infer or uses alphabetical order.
            try:
                positive_class = 1 if 1 in unique_targets else unique_targets[1]
                negative_class = 0 if 0 in unique_targets else unique_targets[0]
            except:
                positive_class = unique_targets[1]
                negative_class = unique_targets[0]

            for var in audit_vars:
                if var not in self.data.columns:
                    continue
                
                # Check if the variable is numeric
                if not pd.api.types.is_numeric_dtype(self.data[var]):
                    continue

                # Separating the groups
                group_churn = self.data[self.data[target] == positive_class][var].dropna()
                group_no_churn = self.data[self.data[target] == negative_class][var].dropna()
                
                # Check if the variable is numeric
                if len(group_churn) == 0 or len(group_no_churn) == 0:
                    continue

                # --- STATISTICAL TESTS ---
                
                # 1. Mann-Whitney U Test (Non-parametric)
                try:
                    stat, p_val = mannwhitneyu(group_churn, group_no_churn, alternative='two-sided')
                except Exception:
                    p_val = 1.0
                
                # 2. Point Biserial Correlation (Linear)
                # Requires a numeric target. If it's text, we attempt a temporary conversion.
                try:
                    if pd.api.types.is_numeric_dtype(self.data[target]):
                        corr, _ = pointbiserialr(self.data[target], self.data[var])
                    else:
                        # Temporary encoding for correlation (0/1)
                        temp_target = self.data[target].map({positive_class: 1, negative_class: 0})
                        corr, _ = pointbiserialr(temp_target, self.data[var])
                except Exception:
                    corr = 0.0

                # --- EFFECT SIZE (LIFT) ---
                med_churn = group_churn.median()
                med_loyal = group_no_churn.median()
                
                # Lift Calculation
                if med_loyal == 0:
                    if med_churn == 0:
                        lift_median = 0.0 
                    else:
                        lift_median = 100.0 # 0 
                else:
                    lift_median = ((med_churn - med_loyal) / abs(med_loyal)) * 100
                
                # Diagnosis
                if p_val < 0.001:
                    strength = '⭐⭐⭐ Strong'
                elif p_val < 0.05:
                    strength = '⭐⭐ Medium'
                else:
                    strength = '❌ Noise'
                
                # Results
                results_audit.append({
                    'Feature': var,
                    'Strength': strength,
                    'P-Value': round(p_val, 5),
                    'Pearson Corr': round(corr, 3),
                    'Median Churn': round(med_churn, 2),
                    'Median Loyal': round(med_loyal, 2),
                    'Impact (%)': round(lift_median, 1), 
                })
            
            # Finalization
            if not results_audit:
                print("No numerical variables analyzed.")
                return

            # Sort by Absolute Correlation
            df_audit = pd.DataFrame(results_audit)
            df_audit['Abs_Corr'] = df_audit['Pearson Corr'].abs()
            df_audit = df_audit.sort_values(by='Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])
        
            # Display
            print('Table of Scientific Evidence (Numerical):')
            display(df_audit)

        except Exception as e:
            print(f'[Error] Failure to generate statistical tests: {str(e)}')
    
    # ======================================================== #
    # Vif Test - Function                                      #
    # ======================================================== #
    def vif_test(
        self,
        audit_vars: list,
    ):
        """
        Calculates the Variance Inflation Factor (VIF) to detect severe multicollinearity among the numerical independent variables.

        Multicollinearity occurs when two or more variables are highly correlated, making it difficult for the model to distinguish the individual effect of each. This inflates the variance of the coefficients and makes the model unstable.

        Methodology:

        1. Cleaning: Converts booleans to int(0/1), numerical strength, and removes NaNs/Infinities.

        2. Calculation: Performs a linear regression of each variable against all others.

        VIF = 1 / (1 - R²).

        3. Constant: Adds a constant (intercept) to ensure that the calculation is not biased by the mean of the data.

        Interpretation of Results (Rule of thumb):

        - VIF = 1: No correlation (Orthogonal variable).

        - 1 < VIF < 5: Moderate correlation (Acceptable).

        - VIF >= 5: High multicollinearity (Warning sign).

        - VIF >= 10: Severe multicollinearity (The variable should be removed).

        Args:

        audit_vars(list): List with the names of the numeric columns to be audited.

        Returns:

        None: Prints an ordered DataFrame with the VIF score of each feature, including the constant (const).
        """
        try:
            
            # Selection features for test
            data = self.data[audit_vars].copy()

            for col in data.columns:
                # Ensures that booleans (True/False) reach 1/0
                if data[col].dtype == 'bool':
                    data[col] =  data[col].astype(int)
                
                # Force the numerical transformation to avoid errors in the test
                data[col] = pd.to_numeric(data[col], errors = 'coerce')
            # Cleaning data
            data.replace([np.inf, -np.inf], np.nan, inplace = True)
            data.dropna(inplace = True)
            print(f'Rows remaining after cleaning: {len(data)}')

            # Calculating the VIF
            data = data.astype(float)
            x_vif = add_constant(data)

            vif_data = pd.DataFrame()
            vif_data['Feature'] = x_vif.columns
            vif_data['VIF'] = [variance_inflation_factor(x_vif.values, i) for i in range(len(x_vif.columns))]   

            print('\n----- VIF Result -----') 
            display(vif_data.sort_values('VIF', ascending = False))
        
        except Exception as e:
            print(f'[Error] Failure to generate VIF test: {str(e)}')


    # ======================================================== #
    # Cramers V - Function                                     #
    # ======================================================== #
    def _cramers_v(
        self,
        x,
        y
    ):
        """
        Calculates Cramer's V statistic (with bias correction) to measure the strength of the association between two categorical (nominal) variables.

        Unlike the Chi-Square test (which only indicates whether there is significance), Cramer's V normalizes the result between 0 and 1, functioning as a "correlation" for categorical data.

        Technical Details:

        - This implementation applies the Bergsma-Wicherink correction to mitigate the positive bias (overestimation) common in the standard calculation of Cramer's V, especially in non-square contingency tables or finite samples.

        Interpretation (Rule of thumb):

        - 0.0: No association (Independent variables).

        - 0.1 to 0.3: Weak association.

        - 0.3 to 0.5: Moderate association.

        - > 0.5: Strong association.

        Args:

        x (pd.Series): First variable Categorical (Predictor).

        y (pd.Series): Second categorical variable (Target).

        Returns:

        float: A value between 0.0 and 1.0 representing the magnitude of the association.

        Returns 0 if there is no association or if a mathematical error occurs.
        """
        try:

            confusion_matrix = pd.crosstab(x, y)
            if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
                return 0 
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                rcorr = r - ((r - 1) **2) / (n - 1)
                kcorr = k - ((k - 1) **2) / (n - 1)
                if min((kcorr - 1), (rcorr - 1)) == 0:
                    return 0
                return np.sqrt(phi2corr / min((kcorr -1), (rcorr -1)))
            
        except Exception as e:
            print(f'[Error] Failed to execute the function _cramers_v: {str(e)}')
    
    # ======================================================== #
    # Chi Square - Function                                    #
    # ======================================================== #
    def chi_square_test(
        self, 
        audit_vars: list,
        target: str
    ):
        """
        Performs a complete statistical audit for categorical variables in relation to a target variable, focusing on dependency and risk analysis.

        This function answers three business questions:

        1. Is there a relationship? (Chi-Square Test of Independence).

        2. How strong is the relationship? (Cramér's V).

        3. Where is the risk? (Risk Profiling: identifies which specific category maximizes or minimizes the rate of the target event).

        Automatic Diagnosis (Strength):

        - ⚠️ Low sample: Expected frequency < 5 (The Chi-Square test loses validity).

        - ⭐⭐⭐ Strong: P-value < 0.001 (Very high certainty of dependency).

        - ⭐⭐ Median: P-value < 0.05 (Statistically significant dependency).

        - ❌ Noise: P-value >= 0.05 (Variables Independent/Noise).

        Args:

        audit_vars (list): List of categorical columns (nominal or ordinal) to be audited.

        target (str): Name of the binary target column (e.g., 'churn', 'fraud').

        The function assumes that the value '1' or the second column of the crosstab
        represents the event of interest.

        Returns:

        None: Displays a DataFrame (display) ordered by association strength (Cramér's V),
        detailing the 'Worst Category' (highest target rate) and the 'Best Category'.
        """
        try:

            # List of result
            results_audit = []

            for var in audit_vars:
                # Check for existence
                if var not in self.data.columns:
                    continue
                
                # Contingency Table (Crossover)
                crosstab = pd.crosstab(self.data[var], self.data[target])

                # Statistical Test (Chi-Square)
                chi2, p_val, dof, expected = chi2_contingency(crosstab)

                # Strength of Association (Cramér's V replaces Pearson)
                assoc = self._cramers_v(self.data[var], self.data[target])
                
                # Calculates the churn rate for each category
                # Axis 1 sums (Churn 0 + Churn 1) to get the category total
                churn_rates = crosstab[1] / crosstab.sum(axis = 1)

                # Indetify the extremes
                # Category with the most churn
                risky_cat = churn_rates.idxmax()                
                # Rate %
                risky_rate = churn_rates.max() * 100
                
                # Category with the least churn
                safe_cat = churn_rates.idxmin()                
                # Rate %
                safe_rate = churn_rates.min() * 100

                # Impact: The difference in risk between the worst and best-case scenarios
                risk_gap = risky_rate - safe_rate
                
                # Diagnosis/ Classification
                if expected.min() < 5:
                    strength = '⚠️ Low sample'
                elif p_val < 0.001:
                    strength = '⭐⭐⭐ Strong'
                elif p_val < 0.05:
                    strength = '⭐⭐ Median'
                else:
                    strength = '❌ Noise'

                # Results
                results_audit.append({
                    'Feature': var,
                    'Strength': strength,
                    'P-Value': round(p_val, 5),
                    'Assoc. (Cramer V)': round(assoc, 3),
                    'Worst Category': f'{risky_cat} ({risky_rate:.0f}%)',
                    'Best Category': f'{safe_cat} ({safe_rate:.0f}%)',
                    'Impact (Gap %)': round(risk_gap, 1)
                })

            # Ordering by Absolute Association (Cramér's V)
            df_audit_cat = pd.DataFrame(results_audit)
            df_audit_cat = df_audit_cat.sort_values(by = 'Assoc. (Cramer V)', ascending = False)

            # Display
            print('Table of Scientific Evidence (Categorical):')
            display(df_audit_cat)

        except Exception as e:
            print(f'[Error] Failure to generate statistical tests: {str(e)}')   

    # ======================================================== #
    # Correlation Ratio - Function                             #
    # ======================================================== #
    def _correlation_ratio(
        self,
        categories, 
        measurements
    ):
        """
        Calculates the Correlation Ratio (Eta / η), a measure of association between a categorical variable and a numerical variable.

        Mathematically, it answers: "How much of the variance of the numerical variable is explained by the grouping of the categorical variable?".

        It is calculated as the square root of the ratio between the sum of squares between groups (SS_Between) and the total sum of squares (SS_Total).

        Interpretation:

        - 1.0: The category predicts the number perfectly (Deterministic).

        - 0.0: The numerical mean is identical across all categories (Independence).

        - Useful for detecting redundancy: If a categorical variable has an Eta close to 1 with a numerical variable, they bring the same information to the model.

        Args:
        categories (array-like): Data from the categorical variable (X-axis).
        measurements (array-like): Data from the numerical variable (Y-axis).

        Returns:
        float: The value of Eta (0 to 1). Returns 0.0 in case of error or empty vectors.
        """
        
        try:

            # Forcing and ensuring correct typing
            cat = pd.Series(categories).astype(str)
            meas = pd.to_numeric(measurements, errors = 'coerce')

            # Security Cleanup (Removes NaNs generated during conversion)
            # Creates a mask where both data sets are valid
            valid_mask = (~cat.isna()) & (~meas.isna())
            cat = cat[valid_mask]
            meas = meas[valid_mask]

            if len(meas) == 0: return 0.0

            # Calcule of means
            y_total_avg = meas.mean()
            y_avg_per_cat = meas.groupby(cat, observed = True).mean()

            # Maps the category average back to each observation.
            y_avg_array = cat.map(y_avg_per_cat)

            # Calcule of Variance (Eta)
            numerator = np.sum((y_avg_array - y_total_avg) ** 2)
            denominator = np.sum((meas - y_total_avg) ** 2)

            if denominator == 0: return 0.0
            return np.sqrt(numerator / denominator)
        
        except Exception as e:
            print(f'[Error] Failed to execute the _correlation_ratio: {str(e)}')


    # ======================================================== #
    # Mixed Redundancy Test - Function                         #
    # ======================================================== #
    def mixed_redundancy_test(
        self,
        audit_pairs: list,
    ):
        """
        Performs a mixed redundancy test (Categorical vs. Numerical) to detect variables that provide the same information in different formats.

        Uses the Correlation Ratio (Eta / η) to measure how well the categorical variable can "explain" or predict the numerical variable.

        Common Scenario:

        - Numerical Variable: 'Age' (25, 30, 45...)

        - Categorical Variable: 'Age_Range' (Young, Adult...)

        - Result: Eta close to 1.0 (Critical Redundancy). The model does not need both.

        Decision Rules:

        - Eta > 0.95 (🔴 CRITIC): The variables are functionally identical. Removing one is mandatory.

        - Eta > 0.80 (⚠️ Alert): Strong overlap. Evaluate if the categorical variable provides any extra value.

        - Eta < 0.80 (✅ Keep): Distinct or complementary information.

        Args:

        audit_pairs (list of tuples): List of tuples containing the pairs to be tested.

        Format: [('categorical_column', 'numeric_column'), ...]

        Returns:

        None: Displays a DataFrame with the diagnosis and recommended action for each pair, ordered by redundancy severity (Eta Score).
        """
        try: 

            audit_results = []

            for cat_col, num_col in audit_pairs:
                if cat_col in self.data.columns and num_col in self.data.columns:

                    # Executes the mixed redundancy test
                    eta_score = self._correlation_ratio(self.data[cat_col], self.data[num_col])

                    # Rules of decison
                    decision = '✅ Keep'
                    action = '-'

                    if eta_score > 0.95:
                        decision = '🔴 CRITIC (Redundant)'
                        action = f'Dropt to `{cat_col}` or `{num_col}`'
                    
                    elif eta_score > 0.80:
                        decision = '⚠️ Alert (Strong)'
                        action = f'Evaluate removal'

                    # Results
                    audit_results.append({
                        'Categorical (X)': cat_col,
                        'Numerical (Y)': num_col,
                        'Eta Score':  round(eta_score, 4),
                        'Diagnosis' : decision,
                        'Action': action
                    })
            
            # Display
            df_audit = pd.DataFrame(audit_results).sort_values(by = 'Eta Score', ascending = False)
            display(df_audit)

        except Exception as e:
            print(f'[Error] Failure to generate statistical tests: {str(e)}')  
