"""
Spotify Data Cleaning and Preprocessing Script

This module provides a comprehensive, object-oriented data cleaning solution for the Spotify dataset.
It loads configuration from config.yaml, sets up logging, and defines a DataCleaner class that handles
all cleaning and preprocessing steps as private methods. The pipeline includes missing value handling,
duplicate removal, type conversions, feature engineering, and memory optimization, with robust exception handling.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import yaml
from src.exception import CustomException
from pathlib import WindowsPath
import re

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load config.yaml
CONFIG_PATH = WindowsPath('D:\projects\spotify_emotion_prediction\config\config.yaml')
try:
    with open(CONFIG_PATH, 'r') as config_file:
        CONFIG = yaml.safe_load(config_file)
    logger.info(f"Configuration loaded successfully from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    CONFIG = {}


    # Define project root
    PROJECT_ROOT = WindowsPath('D:\projects\spotify_emotion_prediction')
    sys.path.append(str(PROJECT_ROOT))

class DataCleaner:
    """
    Object-Oriented Data Cleaner for Spotify Dataset.
    Handles all cleaning and preprocessing steps as per config and roadmap.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or CONFIG
        data_cfg = self.config.get('data', {})
        pre_cfg = self.config.get('preprocessing', {})
        self.raw_data_path = Path(data_cfg.get('raw_data_path', 'src/data/raw/spotify_dataset.csv'))
        self.processed_data_dir = Path(data_cfg.get('processed_data_dir', 'src/data/processed/'))
        self.target_column = data_cfg.get('target_column', 'emotion')
        self.text_column = data_cfg.get('text_column', 'text')
        self.missing_value_strategy = pre_cfg.get('missing_value_strategy', 'drop')
        self.time_signature_imputation = pre_cfg.get('time_signature_imputation', 'mode')
        self.length_conversion_format = pre_cfg.get('length_conversion_format', 'MM:SS')
        self.loudness_suffix_to_remove = pre_cfg.get('loudness_suffix_to_remove', 'db')
        self.release_date_feature_type = pre_cfg.get('release_date_feature_type', 'year')
        self.date_ordinal_suffixes = pre_cfg.get('date_ordinal_suffixes', ["st", "nd", "rd", "th"])
        self.key_encoding = pre_cfg.get('key_encoding', 'label')
        self.explicit_mapping = pre_cfg.get('explicit_mapping', {"Yes": 1, "No": 0})
        self.emotion_cleaning_map = pre_cfg.get('emotion_cleaning_map', {})
        self.rare_emotion_threshold = pre_cfg.get('rare_emotion_threshold', 10)
        # Duplicate handling
        duplicate_cfg = pre_cfg.get('duplicate_handling', {})
        self.duplicate_strategy = duplicate_cfg.get('strategy', 'keep_first')
        self.duplicate_subset = duplicate_cfg.get('subset', None)
        self.outlier_handling_method = pre_cfg.get('outlier_handling_method', 'capping_iqr')
        self.numerical_scaling_method = pre_cfg.get('numerical_scaling_method', 'standard_scaler')
        self.text_vectorization_method = pre_cfg.get('text_vectorization_method', 'transformer_embeddings')
        self.genre_split_delimiter = pre_cfg.get('genre_split_delimiter', ',')
        self.genre_wrap_as_list = pre_cfg.get('genre_wrap_as_list', True)
        self.fill_na_string = pre_cfg.get('fill_na_string', '')
        self.test_size = pre_cfg.get('test_size', 0.1)
        self.validation_size = pre_cfg.get('validation_size', 0.1)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from: {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise CustomException(e, sys)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            strategy = self.missing_value_strategy
            fill_na_string = self.fill_na_string
            logger.info(f"Handling missing values using strategy: {strategy}")
            if strategy == 'drop':
                df = df.dropna()
            elif strategy == 'impute_mode':
                for col in df.columns:
                    if df[col].dtype == 'object':
                        mode = df[col].mode()
                        fill_val = mode[0] if not mode.empty else fill_na_string
                        df[col] = df[col].fillna(fill_val)
                    else:
                        median = df[col].median()
                        df[col] = df[col].fillna(median)
            elif strategy == 'impute_constant':
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna(fill_na_string)
                    else:
                        df[col] = df[col].fillna(0)
            else:
                logger.warning(f"Unknown missing value strategy: {strategy}, no action taken.")
            logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise CustomException(e, sys)

    def _handle_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Removing duplicates using strategy: {self.duplicate_strategy}, subset: {self.duplicate_subset}")
            if self.duplicate_strategy == 'drop':
                df = df.drop_duplicates(subset=self.duplicate_subset, keep=False)
            elif self.duplicate_strategy == 'keep_first':
                df = df.drop_duplicates(subset=self.duplicate_subset, keep='first')
            elif self.duplicate_strategy == 'keep_last':
                df = df.drop_duplicates(subset=self.duplicate_subset, keep='last')
            elif self.duplicate_strategy == 'none':
                logger.info("No duplicate removal performed (strategy: none)")
            else:
                logger.warning(f"Unknown duplicate handling strategy: {self.duplicate_strategy}, no action taken.")
            logger.info(f"Shape after duplicate removal: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error handling duplicate rows: {e}")
            raise CustomException(e, sys)

    def _convert_length_to_seconds(self, x):
        if pd.isna(x) or x == '':
            return np.nan
        parts = str(x).split(':')
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                return float(x)
        except Exception:
            return np.nan

    def _clean_loudness_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            loud_col = None
            for col in df.columns:
                if 'loudness' in col.lower():
                    loud_col = col
                    break
            if loud_col:
                df[loud_col] = df[loud_col].astype(str).str.replace(self.loudness_suffix_to_remove, '', case=False, regex=False)
                df[loud_col] = pd.to_numeric(df[loud_col], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error cleaning loudness column: {e}")
            raise CustomException(e, sys)

    def _process_release_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Row-wise release date cleaning:
        - Reads each row's date as string.
        - Cleans ordinal suffixes.
        - Completes incomplete months (e.g., '2020-05' -> '2020-05-01').
        - Converts to datetime.
        - Extracts features as per config.
        """
        try:
            date_col = None
            for col in df.columns:
                if 'release' in col.lower() and 'date' in col.lower():
                    date_col = col
                    break

            if date_col:
                logger.info(f"Processing release date column: {date_col}")

                def clean_and_complete_date(val):
                    if pd.isna(val):
                        return np.nan
                    s = str(val).strip()
                    # Remove ordinal suffixes (st, nd, rd, th) from day part if present
                    # Replace e.g. '21st', '2nd', '3rd', '4th' with just the number
                    s = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', s, flags=re.IGNORECASE)
                    s = s.strip()
                    # If only year
                    if s.isdigit() and len(s) == 4:
                        return f"{s}-01-01"
                    # If year-month (YYYY-MM)
                    if len(s) == 7 and s[:4].isdigit() and s[5:7].isdigit() and s[4] == '-':
                        return f"{s}-01"
                    return s

                df[date_col] = df[date_col].apply(clean_and_complete_date)
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)

                # Extract features
                if self.release_date_feature_type == 'year':
                    df['release_year'] = df[date_col].dt.year.astype('Int64')
                    logger.info("Extracted release year from dates")
                elif self.release_date_feature_type == 'month':
                    df['release_month'] = df[date_col].dt.month.astype('Int64')
                    logger.info("Extracted release month from dates")
                elif self.release_date_feature_type == 'day':
                    df['release_day'] = df[date_col].dt.day.astype('Int64')
                    logger.info("Extracted release day from dates")
                elif self.release_date_feature_type == 'days_since_epoch':
                    df['days_since_epoch'] = (df[date_col] - pd.Timestamp('1970-01-01')).dt.days.astype('Int64')
                    logger.info("Calculated days since epoch from dates")

                missing_dates = df[date_col].isna().sum()
                logger.info(f"Release date processing complete. Missing values: {missing_dates}")

            return df
        except Exception as e:
            logger.error(f"Error processing release date: {e}")
            raise CustomException(e, sys)

    def _encode_key_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'Key' in df.columns:
                if self.key_encoding == 'one-hot':
                    key_dummies = pd.get_dummies(df['Key'], prefix='Key')
                    df = pd.concat([df, key_dummies], axis=1)
                elif self.key_encoding == 'label':
                    key_map = {k: i for i, k in enumerate(sorted(df['Key'].dropna().unique()))}
                    df['Key_encoded'] = df['Key'].map(key_map).fillna(-1).astype(int)
            return df
        except Exception as e:
            logger.error(f"Error encoding key column: {e}")
            raise CustomException(e, sys)

    def _map_explicit_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'Explicit' in df.columns:
                df['Explicit_bin'] = df['Explicit'].map(self.explicit_mapping)
            return df
        except Exception as e:
            logger.error(f"Error mapping explicit column: {e}")
            raise CustomException(e, sys)

    def _clean_emotion_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and standardizes the emotion/target column:
        - Lowercases and strips whitespace.
        - Applies mapping from config if provided.
        - Groups rare emotions under 'rare' based on threshold.
        """
        try:
            col = self.target_column
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .replace(self.emotion_cleaning_map or {})
                )
                counts = df[col].value_counts()
                rare_emotions = counts[counts < self.rare_emotion_threshold].index
                df[col] = df[col].where(~df[col].isin(rare_emotions), 'rare')
            return df
        except Exception as e:
            logger.error(f"Error cleaning emotion column: {e}")
            raise CustomException(e, sys)

    def _clean_genre_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'Genre' in df.columns:
                df['Genre'] = df['Genre'].str.lower().str.strip()
                if self.genre_wrap_as_list:
                    df['Genre'] = df['Genre'].apply(
                        lambda x: [g.strip() for g in str(x).split(self.genre_split_delimiter)] if pd.notna(x) and x != '' else []
                    )
            return df
        except Exception as e:
            logger.error(f"Error cleaning genre column: {e}")
            raise CustomException(e, sys)

    def _clean_artists_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean artist columns efficiently using vectorized operations and convert to lists
        """
        try:
            artist_columns = ['Artist(s)']
            for col in artist_columns:
                if col in df.columns:
                    # Use vectorized string operations instead of apply
                    mask = df[col].notna()
                    df.loc[mask, col] = df.loc[mask, col].str.strip()
                    df[col] = df[col].fillna('Unknown Artist')
                    
                    # Convert comma-separated artists to list
                    df[col] = df[col].str.split(',').apply(lambda x: [artist.strip() for artist in x] if isinstance(x, list) else ['Unknown Artist'])
                    
                    if 'Artists' in col:
                        # Get count from list length
                        df[f'{col}_count'] = df[col].str.len().fillna(1).astype('int8')
            
            return df
        except Exception as e:
            logger.error(f"Error cleaning artist columns: {e}")
            raise CustomException(e, sys)
            
            
    def _clean_similar_artist_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean similar artist columns using vectorized operations and convert to lists.
        """
        try:
            similar_cols = ['Similar Artist 1', 'Similar Artist 2', 'Similar Artist 3']
            for col in similar_cols:
                if col in df.columns:
                    mask = df[col].notna()
                    df.loc[mask, col] = df.loc[mask, col].str.strip()
                    df[col] = df[col].fillna('No Similar Artists')
                    # Convert comma-separated similar artists to list
                    df[col] = df[col].str.split(',').apply(
                        lambda x: [artist.strip() for artist in x] if isinstance(x, list) else ['No Similar Artists']
                    )
                    # Optionally, add a count column
                    df[f'{col}_count'] = df[col].str.len().fillna(1).astype('int8')
            return df
        except Exception as e:
            logger.error(f"Error cleaning similar artist columns: {e}")
            raise CustomException(e, sys)

    def _downcast_numerical_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for col in df.select_dtypes(include=['int64', 'int32']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64', 'float32']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            return df
        except Exception as e:
            logger.error(f"Error optimizing data types: {e}")
            raise CustomException(e, sys)

    def clean_data(self) -> pd.DataFrame:
        try:
            df = self._load_data()
            df = self._handle_missing_values(df)
            df = self._handle_duplicate_rows(df)
            df['Length_sec'] = df['Length'].apply(self._convert_length_to_seconds) if 'Length' in df.columns else df.get('Length_sec', np.nan)
            df = self._clean_loudness_column(df)
            df = self._process_release_date(df)
            df = self._encode_key_column(df)
            df = self._map_explicit_column(df)
            df = self._clean_emotion_column(df)
            df = self._clean_genre_column(df)
            df = self._clean_artists_columns(df)
            df = self._clean_similar_artist_columns(df)
            df = self._downcast_numerical_dtypes(df)
            logger.info(f"Final cleaned data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error in data cleaning pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data()
        output_path = cleaner.processed_data_dir / "cleaned_spotify_data.csv"
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path}")
    except CustomException as ce:
        logger.error(f"Custom Exception occurred: {ce}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        sys.exit(1)
