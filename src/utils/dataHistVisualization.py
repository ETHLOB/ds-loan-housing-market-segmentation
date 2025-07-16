import matplotlib.pyplot as plt

from numpy import ceil
from pandas import DataFrame

from typing import Literal

def visualize_histograms(Frame: DataFrame, column_type: Literal['number', 'object'] = 'number', bins: int = 30) -> None:
    """
    Visualiza histogramas de las variables numéricas u objetos en un DataFrame con una estrucura partuclar definida en este ejercicio

    Args:
        data_processed (_type_): _description_
    """
    # Now we create histograms for all numeric variables in a single visualization
    numeric_cols = Frame.select_dtypes(include = [column_type]).columns
    n_cols = len(numeric_cols)

    # Calculate optimal grid dimensions and create subplots
    n_subplot_cols = min(3, n_cols) # max 3 columns per row
    n_rows = int(ceil(n_cols / 3)) # calculate number of rows needed given number of columns

    fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
        
    # We plot a histogram for each numeric variable we got!
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        
        if column_type == 'number':
            # Create histogram charts
            Frame.loc[:, col].hist(bins = bins, alpha = 0.7, color = plt.cm.Set3(i), ax = ax, edgecolor = 'black', linewidth = 0.5)
            
            # Add labels and title
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{col} values', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            
            # Add statistics text
            mean_val = Frame.loc[:, col].mean()
            median_val = Frame.loc[:, col].median()
            ax.axvline(mean_val, color = 'red', linestyle = '--', alpha = 0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color = 'blue', linestyle = '--', alpha = 0.8, label=f'Median: {median_val:.2f}')
            ax.legend(fontsize=8)
        
        else:
            # Create bar charts
            value_counts = Frame.loc[:, col].value_counts()
            bars = ax.bar(range(len(value_counts)), value_counts.values, alpha = 0.7, color = plt.cm.Set3(i), edgecolor='black', linewidth=0.5)
            
            # Add labels and title
            ax.set_title(f'{col}', fontsize = 12, fontweight = 'bold')
            ax.set_xlabel(f'{col} categories', fontsize = 8)
            ax.set_ylabel('Frequency', fontsize=8)
            
            # Set x-axis labels
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
            
            # Add value labels on top of bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8
                )
            
        # Grid
        ax.grid(True, alpha=0.3)

    # Hide empty subplots if any
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Histograma de las variables numéricas', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return 