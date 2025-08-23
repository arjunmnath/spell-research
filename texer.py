import pandas as pd
import argparse
from typing import List

def generate_latex_with_pandas(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        cols_to_drop: List[str] = ['f1_macro', 'f1_micro']
        if 'feature' in df.columns:
            cols_to_drop.append('feature')

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Note: Columns {cols_to_drop} found and dropped.")

        f1_col = 'f1' if 'f1' in df.columns else 'f1_score'
        
        if f1_col not in df.columns:
            return "Error: No 'f1' or 'f1_score' column found for sorting."

        # Ensure the F1 score column is numeric for correct sorting
        df[f1_col] = pd.to_numeric(df[f1_col], errors='coerce')
        df.dropna(subset=[f1_col], inplace=True)

        # 3. Sort by F1 score and select the top 5 models
        top_5_models = df.sort_values(by=f1_col, ascending=False).head(10)
        # 4. Generate LaTeX code from the processed DataFrame
    
        

        for col in top_5_models.select_dtypes(include='number').columns:
            top_5_models[col] = top_5_models[col].apply(lambda x: f"{x:.4f}")

        latex_code = top_5_models.to_latex(
            index=False,
            caption="Top 5 Models by Weighted F1 Score",
            label="tab:top5_models",
            position="h!",
            escape=True
        )
        return latex_code

    except FileNotFoundError:
        return f"Error: The file at '{csv_path}' was not found."
    except pd.errors.EmptyDataError:
        return "Error: The CSV file is empty."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a CSV file into a LaTeX table for the top 5 models by F1 score."
    )
    parser.add_argument(
        "csv_filepath",
        type=str,
        help="The path to the input CSV file."
    )
    args = parser.parse_args()
    latex_output = generate_latex_with_pandas(args.csv_filepath)
    print(latex_output)
