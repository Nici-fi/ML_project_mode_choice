import pandas as pd
import matplotlib.pyplot as plt
import sklearn

def plot_stacked_percentage_bar_chart(df, feature, target):

    def format_label(label):
        return ' '.join(word.capitalize() for word in label.split('_'))

    # Create a crosstab of the specified feature against the target
    crosstab = pd.crosstab(df[feature], df[target])

    # Normalize the crosstab to sum to 1
    crosstab_norm = crosstab.div(crosstab.sum(1).astype(float), axis=0)

    # Automatically generate title and labels with formatting
    title = f'Distribution of {format_label(target)} by {format_label(feature)}'

    # Plot the normalized crosstab and capture the Axes object
    ax = crosstab_norm.plot(kind='bar', stacked=True, figsize=(10, 6), title=title)

    # Set axis labels
    ax.set_xlabel(format_label(feature))
    ax.set_ylabel(format_label(target))

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
