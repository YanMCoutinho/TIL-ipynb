import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

class ChartGenerator:
    """
    A class for generating various charts from a pandas DataFrame.
    """

    def __init__(self, df):
        """
        Initialize with a pandas DataFrame.

        Parameters:
        - df: pandas DataFrame containing the data.
        """
        self.df = df

    def line_chart(self, x, y, title=None, xlabel=None, ylabel=None, legend=None):
        """
        Create a line chart using DataFrame columns.

        Parameters:
        - x: string, name of the column for x-axis.
        - y: string, name of the column for y-axis.
        - title: (optional) chart title.
        - xlabel: (optional) label for x-axis.
        - ylabel: (optional) label for y-axis.
        - legend: (optional) legend label for the line.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        ax.plot(self.df[x], self.df[y], label=legend)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if legend: ax.legend()
        plt.tight_layout()
        return fig, ax

    def bar_chart(self, category_col, value_col, title=None, xlabel=None, ylabel=None, rotation=0):
        """
        Create a bar chart using DataFrame columns.

        Parameters:
        - category_col: string, name of the column for categories.
        - value_col: string, name of the column for values.
        - title: (optional) chart title.
        - xlabel: (optional) label for x-axis.
        - ylabel: (optional) label for y-axis.
        - rotation: (optional) rotation angle for x-tick labels.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        # build labels: join multiple cols or take single
        if isinstance(category_col, (list, tuple)):
            labels = (
                self.df[category_col]
                .astype(str)
                .agg(' - '.join, axis=1)
            )
        else:
            labels = self.df[category_col].astype(str)
        values = self.df[value_col]
        ax.bar(labels, values)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if rotation:
            plt.setp(ax.get_xticklabels(), rotation=rotation)
        plt.tight_layout()
        return fig, ax

    def scatter_chart(self, x, y, title=None, xlabel=None, ylabel=None, color=None):
        """
        Create a scatter chart using DataFrame columns.

        Parameters:
        - x: string, name of the column for x values.
        - y: string, name of the column for y values.
        - title: (optional) chart title.
        - xlabel: (optional) label for x-axis.
        - ylabel: (optional) label for y-axis.
        - color: (optional) marker color.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        ax.scatter(self.df[x], self.df[y], c=color)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig, ax

    def histogram(self, column, bins=10, title=None, xlabel=None, ylabel=None):
        """
        Create a histogram using a DataFrame column.

        Parameters:
        - column: string, name of the column to plot.
        - bins: (optional) number of histogram bins.
        - title: (optional) chart title.
        - xlabel: (optional) label for x-axis.
        - ylabel: (optional) label for y-axis.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        ax.hist(self.df[column], bins=bins)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig, ax

    def pie_chart(self, label_col, size_col, title=None, autopct='%1.1f%%'):
        """
        Create a pie chart using DataFrame columns.

        Parameters:
        - label_col: string, name of the column for labels.
        - size_col: string, name of the column for sizes.
        - title: (optional) chart title.
        - autopct: (optional) format string for percentages.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        ax.pie(self.df[size_col], labels=self.df[label_col], autopct=autopct)
        if title: ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    def box_plot(self, columns, title=None, ylabel=None):
        """
        Create a box plot for one or more DataFrame columns.

        Parameters:
        - columns: list of strings, names of the columns to include.
        - title: (optional) chart title.
        - ylabel: (optional) label for y-axis.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        ax.boxplot([self.df[col] for col in columns], labels=columns)
        if title: ax.set_title(title)
        if ylabel: ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig, ax

    def heatmap(self, title=None, cmap='viridis'):
        """
        Create a heatmap of the entire DataFrame.

        Parameters:
        - title: (optional) chart title.
        - cmap: (optional) colormap name.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        cax = ax.imshow(self.df.values, cmap=cmap)
        ax.set_xticks(range(len(self.df.columns)))
        ax.set_xticklabels(self.df.columns)
        ax.set_yticks(range(len(self.df.index)))
        ax.set_yticklabels(self.df.index)
        if title: ax.set_title(title)
        fig.colorbar(cax)
        plt.tight_layout()
        return fig, ax

    def area_chart(self, x, ys, labels=None, title=None, xlabel=None, ylabel=None):
        """
        Create an area (stack) chart using DataFrame columns.

        Parameters:
        - x: string, name of the column for x-axis.
        - ys: list of strings, names of the columns for stack areas.
        - labels: (optional) labels for each layer.
        - title: (optional) chart title.
        - xlabel: (optional) label for x-axis.
        - ylabel: (optional) label for y-axis.

        Returns:
        - fig, ax: Matplotlib figure and axes objects.
        """
        fig, ax = plt.subplots()
        data = [self.df[col] for col in ys]
        ax.stackplot(self.df[x], *data, labels=labels)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if labels: ax.legend()
        plt.tight_layout()
        return fig, ax

    def scatter_matrix(self, columns=None, diagonal='hist', figsize=(8, 8), **kwargs):
        """
        Create a scatter matrix (pair plot) for DataFrame columns.

        Parameters:
        - columns: list of strings, names of the columns to include (defaults to all).
        - diagonal: (optional) type of plot on diagonal ('hist', 'kde').
        - figsize: (optional) figure size tuple.
        - kwargs: additional plotting keyword arguments.

        Returns:
        - fig, axes: Matplotlib figure and axes array.
        """
        data = self.df[columns] if columns else self.df
        axes = scatter_matrix(data, diagonal=diagonal, figsize=figsize, **kwargs)
        fig = axes[0, 0].get_figure()
        plt.tight_layout()
        return fig, axes
