from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class StringDictionaryMapper(BaseEstimator, TransformerMixin):
    """
    Este estimador personalizado puede mapear valores de cadena y revertir el mapeo cuando sea necesario.
    Crea automáticamente un diccionario de mapeo inverso para operaciones reversibles.

    Args:
    mapping_dict : dict
        Diccionario que contiene el mapeo de valores originales a nuevos valores.

    columns : list o str, por defecto=None
        Nombre(s) de columna(s) a los que aplicar el mapeo.

    handle_missing : str, por defecto='ignore'
        Cómo manejar los valores que no se encuentran en el diccionario de mapeo.

    default_value : cualquier tipo, por defecto=None
        Valor por defecto a usar cuando handle_missing='default'

    reversible : bool, por defecto=True
        Si habilitar la capacidad de transformación inversa.
        Si es True, crea y almacena el mapeo inverso.
    """
    
    def __init__(self, mapping_dict=None, columns=None, handle_missing='ignore', 
                 default_value=None, reversible=True):
        self.mapping_dict = mapping_dict
        self.columns = columns
        self.handle_missing = handle_missing
        self.default_value = default_value
        self.reversible = reversible
    
    def fit(self, X, y=None):
        """Fit the estimator and create inverse mapping if reversible=True."""
        # Validate inputs
        if self.mapping_dict is None:
            raise ValueError("mapping_dict cannot be None")
        
        if not isinstance(self.mapping_dict, dict):
            raise ValueError("mapping_dict must be a dictionary")
        
        # Determine columns to transform
        if self.columns is None:
            self.columns_ = X.select_dtypes(include=['object']).columns.tolist()
        elif isinstance(self.columns, str):
            self.columns_ = [self.columns]
        else:
            self.columns_ = list(self.columns)
        
        # Validate columns exist
        missing_cols = set(self.columns_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in input data")
        
        # Store mapping
        self.mapping_dict_ = self.mapping_dict.copy()
        
        # Create inverse mapping if reversible
        if self.reversible:
            self.inverse_mapping_dict_ = self._create_inverse_mapping()
            
        # Store original values for each column (for non-mapped values)
        self.original_values_ = {}
        for col in self.columns_:
            if col in X.columns:
                unique_vals = X[col].dropna().unique()
                # Store values that are NOT in the mapping dict
                unmapped_vals = set(unique_vals) - set(self.mapping_dict_.keys())
                self.original_values_[col] = list(unmapped_vals)
        
        self.is_fitted_ = True
        return self
    
    def _create_inverse_mapping(self):
        """Create inverse mapping dictionary for reverse transformation."""
        inverse_dict = {}
        
        # Check for duplicate values in mapping (would make inverse impossible)
        values = list(self.mapping_dict_.values())
        if len(values) != len(set(values)):
            duplicates = [v for v in values if values.count(v) > 1]
            raise ValueError(
                f"Cannot create inverse mapping: duplicate values found {set(duplicates)}. "
                "Inverse transformation requires one-to-one mapping."
            )
        
        # Create inverse mapping
        for original, mapped in self.mapping_dict_.items():
            inverse_dict[mapped] = original
            
        return inverse_dict
    
    def transform(self, X):
        """Transform data using the mapping dictionary."""
        check_is_fitted(self, 'is_fitted_')
        
        X_transformed = X.copy()
        
        for col in self.columns_:
            if col in X_transformed.columns:
                X_transformed[col] = self._apply_mapping(X_transformed[col])
        
        return X_transformed
    
    def inverse_transform(self, X):
        """
        Reverse the transformation using the inverse mapping.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Transformed data to reverse
            
        Returns
        -------
        X_original : pandas.DataFrame
            Data with original string values restored
        """
        check_is_fitted(self, 'is_fitted_')
        
        if not self.reversible:
            raise ValueError(
                "Inverse transform not available. Set reversible=True when creating the estimator."
            )
        
        X_original = X.copy()
        
        for col in self.columns_:
            if col in X_original.columns:
                X_original[col] = self._apply_inverse_mapping(X_original[col])
        
        return X_original
    
    def _apply_mapping(self, series):
        """Apply forward mapping to a series."""
        if self.handle_missing == 'ignore':
            return series.map(self.mapping_dict_).fillna(series)
        elif self.handle_missing == 'error':
            unmapped_values = set(series.dropna().unique()) - set(self.mapping_dict_.keys())
            if unmapped_values:
                raise ValueError(f"Values {unmapped_values} not found in mapping dictionary")
            return series.map(self.mapping_dict_)
        elif self.handle_missing == 'nan':
            return series.map(self.mapping_dict_)
        elif self.handle_missing == 'default':
            mapped = series.map(self.mapping_dict_)
            return mapped.fillna(self.default_value)
        else:
            raise ValueError(f"handle_missing must be one of ['ignore', 'error', 'nan', 'default']")
    
    def _apply_inverse_mapping(self, series):
        """Apply inverse mapping to a series."""
        # For inverse mapping, we need to handle values that weren't originally mapped
        if self.handle_missing == 'ignore':
            # Use inverse mapping, but keep values that weren't in the original mapping
            inversely_mapped = series.map(self.inverse_mapping_dict_)
            # Fill NaN with original series values (for values that weren't mapped originally)
            return inversely_mapped.fillna(series)
        else:
            # For other modes, apply inverse mapping directly
            return series.map(self.inverse_mapping_dict_)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for output."""
        check_is_fitted(self, 'is_fitted_')
        if input_features is None:
            return self.columns_
        else:
            return input_features