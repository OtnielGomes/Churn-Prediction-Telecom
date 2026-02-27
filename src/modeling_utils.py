# ================================================================================================= #
# >>> A module of functions and classes to facilitate the flow and creation of pipelines in sklearn #                                 
# ================================================================================================= #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================== #
# Class FeatureEngineer                                     #
# ======================================================== #
class FeatureEngineer(
    BaseEstimator, 
    TransformerMixin
):
    """

    Custom transformer (scikit-learn) for feature engineering on the Telecom Churn dataset.

    This transformer creates features derived from:

    - Monthly spending aggregation (total_spend).

    - Affordability/income affinity indices (affordability_idx and *_inc).

    - "Toxicity" score (toxic_score) and interaction with education level (toxic_ed).

    - Behavior and stability interactions (tenure_years and interactions with spending/attributes).

    - "good_score" (sum of usage/service indicators).

    Notes:

    - Expects a pandas.DataFrame with column names; if it receives another format, it tries to convert

    to DataFrame (but this may lose column names). [file:1]

    - `fit()` does not learn parameters; it exists only for Pipeline compatibility. [web:2]

    Created features (if the base columns exist):

    - total_spend

    - affordability_idx

    - longmon_inc, equipmon_inc, cardmon_inc, wiremon_inc 
    - toxic_score, toxic_ed 
    - tenure_years 
    - tenure_longmon, tenure_cardmon 
    - age_longmon, age_cardmon 
    - address_longmon, address_cardmon 
    - employ_longmon, employ_cardmon 
    - stability_age, stability_address, stability_employ 
    - good_score 
    """


    # ======================================================== #
    # __Init__ - Function                                      #
    # ======================================================== #
    def __init__(
        self,
    ):
        """Initializes the transformer.

        Parameters

        ----------

        None
        This transformer does not have any hyperparameters at this time.

        """
        pass


    # ======================================================== #
    # Fit - Function                                           #
    # ======================================================== #
    def fit(
        self, 
        X, 
        y = None
    ):
        """ 
        Fit the transformer.

        This method estimates nothing; it only returns `self` to fulfill the scikit-learn contract

        and allow `fit_transform()`. [web:2]

        Parameters

        ---------
        X : pandas.DataFrame
        Set of features.

        y : array-like, default=None
        Target (ignored).

        Returns

        -------
        self : FeatureEngineer
        Fitted transformer instance (stateless).

        """

        return self


    # ======================================================== #
    # Transform - Function                                     #
    # ======================================================== #
    def transform(
        self, 
        X
    ):
        """
        Applies feature engineering.

        Parameters

        ----------
        X : pandas.DataFrame
        Input data with columns from the telecom dataset (e.g., income, tenure, longmon, etc.).

        Returns

        -------
        pandas.DataFrame
        DataFrame with the new features added.

        Raises

        ------
        Exception
        Forwards any errors that occurred during feature creation (safer in a pipeline). [web:2]

        """
        
        try:
            # Check Data set 
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            X = X.copy()

            # Total wallet share --------
            mon_cols = ['longmon', 'tollmon', 'equipmon', 'cardmon','wiremon']
            existing_mon = [c for c in  mon_cols if c in X.columns]
            X['total_spend'] = X[existing_mon].sum(axis = 1).astype('float32') if existing_mon else np.float32(0.0)

            # Affordability and expenses/income --------

            if 'income' in X.columns:
                income_scale = 1000.0
                denom_income = X['income'].astype('float64') * income_scale + 1.0
            else: 
                denom_income = 1.0

            X['affordability_idx'] = (
                X['total_spend'].astype('float64') / denom_income
            ).astype('float32')

            for col in ['longmon', 'equipmon', 'cardmon', 'wiremon']:
                if col in X.columns:
                    X[f'{col}_inc'] = (
                        X[col].astype('float64') / denom_income
                    ).astype('float32')
                
                else:
                    X[f'{col}_inc'] = np.float32(0.0)
            
            # Risk (toxicity + education) --------
            toxic_list = ['internet', 'wireless', 'equip', 'voice', 'pager']
            existing_toxic = [c for c in toxic_list if c in X.columns]
            X['toxic_score'] = X[existing_toxic].sum(axis = 1).astype('float32') if existing_toxic else np.float32(0.0)

            if 'ed' in X.columns:
                X['toxic_ed'] = (
                    X['toxic_score'].astype('float64') * X['ed'].astype('int64')
                ).astype('float32')
            else:
                X['toxic_ed'] = np.float32(0.0)

            # Tenure in years (tenure is in months) --------
            if 'tenure' in X.columns:
                X['tenure_years'] = (
                    X['tenure'].astype('float64') / 12.0
                ).astype('float32')
            else:
                X['tenure_years'] = np.float32(0.0)

            # Usage Behavior --------
            # tenure_longmon / tenure_cardmon
            if 'longmon' in X.columns:
                X['tenure_longmon'] = (
                    X['tenure_years'].astype('float64') * X['longmon'].astype('float64')
                ).astype('float32')
            else:
                X['tenure_longmon'] = np.float32(0.0)
            
            if 'cardmon' in X.columns:
                X['tenure_cardmon'] = (
                    X['tenure_years'].astype('float64') * X['cardmon'].astype('float64')
                ).astype('float32')
            else:
                X['tenure_cardmon'] = np.float32(0.0)

            # age_longmon / age_cardmon
            if 'age' in X.columns and 'longmon' in X.columns:
                X['age_longmon'] = (
                    (X['age'].astype('float64') -18.0) * X['longmon'].astype('float64')
                ).astype('float32')
            else:
                X['age_longmon'] = np.float32(0.0)

            if 'age' in X.columns and 'cardmon' in X.columns:
                X['age_cardmon'] = (
                    (X['age'].astype('float64') -18.0) * X['cardmon'].astype('float64')
                ).astype('float32')
            else:
                X['age_cardmon'] = np.float32(0.0)

            # address_longmon / address_cardmon
            if 'address' in X.columns and 'longmon' in X.columns:
                X['address_longmon'] = (
                    X['address'].astype('float64') * X['longmon'].astype('float64')
                ).astype('float32')
            else:
                X['address_longmon'] = np.float32(0.0)

            if 'address' in X.columns and 'cardmon' in X.columns:
                X['address_cardmon'] = (
                    X['address'].astype('float64') * X['cardmon'].astype('float64')
                ).astype('float32')
            else:
                X['address_cardmon'] = np.float32(0.0)

            # employ_longmon / employ_cardmon
            if 'employ' in X.columns and 'longmon' in X.columns:
                X['employ_longmon'] = (
                    X['employ'].astype('float64') * X['longmon'].astype('float64')
                ).astype('float32')
            else:
                X['employ_longmon'] = np.float32(0.0)

            if 'employ' in X.columns and 'cardmon' in X.columns:
                X['employ_cardmon'] = (
                    X['employ'].astype('float64') * X['cardmon'].astype('float64')
                ).astype('float32')
            else:
                X['employ_cardmon'] = np.float32(0.0)

            # Stability --------
            if 'age' in X.columns:
                X['stability_age'] = (
                    X['tenure_years'].astype('float64') * (X['age'].astype('float64') -18.0)
                ).astype('float32')
            else:
                X['stability_age'] = np.float32(0.0)

            if 'address' in X.columns:
                X['stability_address'] = (
                    X['tenure_years'].astype('float64') * X['address'].astype('float64')
                ).astype('float32')
            else:
                X['stability_address'] = np.float32(0.0)

            if 'employ' in X.columns:
                X['stability_employ'] = (
                    X['tenure_years'].astype('float64') * X['employ'].astype('float64')
                ).astype('float32')
            else:
                X['stability_employ'] = np.float32(0.0)
            
            # Good score --------
            good_cols =  ['callcard', 'confer', 'callwait']
            existing_good = [c for c in good_cols if c in X.columns]
            X['good_score'] = X[existing_good].sum(axis = 1).astype('int32') if existing_good else np.int32(0)

            # Categorical column 'ed'

            if 'ed' in X.columns:
                X['ed'] = X['ed'].astype('int32').replace({5: 4})
            else:
                X['ed'] = np.int32(0)

            # Final cleanup (inf / NaN -> 0), same as project style --------

            num_cols = X.select_dtypes(include = [np.number]).columns
            X[num_cols] = (
                X[num_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            return X
        
        except Exception  as e:
            print(f'[Error] Failure in the creation of new features : {str(e)}.')

# ======================================================== #
# Class DtypeOptimizer                                     #
# ======================================================== #

class DtypeOptimizer(
    BaseEstimator,
    TransformerMixin
):
    """
    Transformer (scikit-learn) to standardize the column types (dtypes) of the Telecom dataset.

    Objective

    --------

    - Reduce memory usage (e.g., float32/int8).

    - Ensure consistency between training and testing (prevent columns from becoming objects).

    - Facilitate use in Pipeline/ColumnTransformer with models like LightGBM/XGBoost.

    Strategy

    ----------

    - continuous_cols -> float32

    - binary_cols -> int8 (0/1)

    - integer_cols -> int32 (counts/years/months; includes 'ed' if you want to treat it as a numeric ordinal)

    - categorical_cols -> category (nominals, e.g., 'custcat')

    Important Note

    ---------------------

    The `transform()` method expects a pandas.DataFrame with column names. If it receives another type,

    it attempts to convert it to DataFrame; this can create columns 0..N and you lose the original names,

    so it's best to use this step before any step that generates numpy arrays. [web:2]

    """


    # ======================================================== #
    # __Init__ - Function                                      #
    # ======================================================== #
    def __init__(
        self,
        categorical_cols:list = ['custcat'],
        integer_cols:list = ['tenure', 'age', 'address', 'employ', 'ed'],
        binary_cols:list = [
            'equip','callcard', 'wireless','ebill', 'voice', 'pager',
            'internet', 'callwait', 'confer','churn'
        ],
        continuous_cols:list = [
            'income',  'longmon', 'tollmon', 'equipmon', 'cardmon','wiremon',
        ],
    ):
        
        """
        Parameters
        ----------
        categorical_cols : list of str
        Columns to be converted to dtype 'category' (nominal variables).

        integer_cols : list of str
        Columns to be converted to dtype 'int32' (counts/ages/time; numeric ordinals).

        binary_cols : list of str
        Binary columns (0/1) to be converted to dtype 'int8'.

        continuous_cols : list of str
        Continuous columns (e.g., monetary values) to be converted to dtype 'float32'.

        """

        self.categorical_cols = categorical_cols
        self.integer_cols = integer_cols
        self.binary_cols = binary_cols
        self.continuous_cols = continuous_cols


    # ======================================================== #
    # Fit - Function                                           #
    # ======================================================== #
    def fit(
        self,
        X, 
        y = None
    ):
        """
        Fit the transformer.

        Does not learn parameters; returns `self` to fulfill the scikit-learn contract

        and allow `fit_transform()`. [web:2]

        Parameters

        ---------
        X : pandas.DataFrame
        Input data.

        y : array-like, default=None
        Target (ignored).

        Returns

        -------
        self : DtypeOptimizer
        """
        return self
    

    # ======================================================== #
    # Transform - Function                                     #
    # ======================================================== #
    def transform(
        self, 
        X
    ):
        """
        Converts the dtypes of the columns defined in __init__.

        Parameters
        ----------
        X : pandas.DataFrame
        DataFrame with the dataset columns.

        Returns

        -------
        pandas.DataFrame
        Copy of the DataFrame with adjusted dtypes.
        """
        try:
            # Check Dataset
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFramme(X)
            X = X.copy()

            # Continuos
            continuous = [c for c in self.continuous_cols if c in X.columns]
            if continuous:
                X[continuous] = X[continuous].astype('float32')

            # Binary
            binary = [c for c in self.binary_cols if c in X.columns]
            if binary:
                X[binary] = X[binary].astype('int8')
            
            # Integer
            integer = [c for c in self.integer_cols if c in X.columns]
            if integer:
                X[integer] = X[integer].astype('int32')

            # Categorical
            categorical = [c for c in self.categorical_cols if c in X.columns]
            if categorical:
                X[categorical] = X[categorical].astype('category')
            
            return X

        except Exception as e:
            print(f'[Error] Failure to add data types to dataset variables: {str(e)}.')