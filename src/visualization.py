# ============================================================================= #
# >>> Module of functions and classes for creating graphs and visualizing data. #                                        
# ============================================================================= #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
# Graphics:
import matplotlib.pyplot as plt
import seaborn as sns
# Data manipulation and visualization:
# Pandas
import pandas as pd
import numpy as np
from scipy.stats import skew

# ======================================================== #
# Graphics Data - Class                                    #
# ======================================================== #

class GraphicsData:

    # Initialize Class
    def __init__(
        self, 
        data : pd.DataFrame,

    ):
        try:
            # Entry checks
            if data.empty:
                raise ValueError('The provided DataFrame is empty.')

            self.data = data

        except Exception  as e:
            print(f'[Error] Failed to load Dataframe : {str(e)}')

    # ======================================================== #
    # Initializer Subplot Grid - Function                      #
    # ======================================================== #
    def _initializer_subplot_grid(
        self, 
        num_columns, 
        figsize_per_row,
        h_size: int = 25
    ):
        """
        Initializes and returns a standardized matplotlib subplot grid layout.

        This utility method calculates the required number of rows based on 
        the number of variables in the dataset and the desired number of 
        columns per row. It then creates a grid of subplots accordingly and 
        applies a consistent styling.

        Args:
            num_columns (int): Number of subplots per row.
            figsize_per_row (int): Vertical size (height) per row in the final figure.

        Returns:
            tuple:
                - fig (matplotlib.figure.Figure): The full matplotlib figure object.
                - ax (np.ndarray of matplotlib.axes._subplots.AxesSubplot): Flattened array of subplot axes.
        """
        num_vars = len(self.data.columns)
        num_rows = (num_vars + num_columns - 1) // num_columns

        plt.rc('font', size = 12)
        fig, ax = plt.subplots(num_rows, num_columns, figsize = (h_size, num_rows * figsize_per_row))
        ax = ax.flatten()
        sns.set(style = 'whitegrid')

        return fig, ax
    
    # ======================================================== #
    # Finalize Subplot Layout - Function                       #
    # ======================================================== #
    def _finalize_subplot_layout(
        self,
        fig,
        ax,
        i: int,
        title: str = None,
        fontsize: int = 30,
    ):
        """
        Finalizes and displays a matplotlib figure by adjusting layout and removing unused subplots.

        This method is used after plotting multiple subplots to:
        - Remove any unused axes in the grid.
        - Set a central title for the entire figure.
        - Automatically adjust spacing and layout for better readability.
        - Display the resulting plot.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure object containing the subplots.
            ax (np.ndarray of matplotlib.axes.Axes): Array of axes (flattened) for all subplots.
            i (int): Index of the last used subplot (all subplots after this will be removed).
            title (str, optional): Title to be displayed at the top of the entire figure.
            fontsize (int, optional): Font size of the overall title. Default is 30.
        """
        for j in range(i + 1, len(ax)):
                fig.delaxes(ax[j])
        
        plt.suptitle(title, fontsize = fontsize, fontweight = 'bold')
        plt.tight_layout(rect = [0, 0, 1, 0.97])
        plt.show()

    # ======================================================== #
    # Format Single AX - Function                              #
    # ======================================================== #
    def _format_single_ax(
        self,
        ax,
        title: str = None,
        fontsize: int = 20,
        linewidth: float = 0.9,
        feature_skew: float = 0,
        pct_outliers: float = 0
    ):

        """
        Applies standard formatting to a single subplot axis.

        This method configures a single axis by:
        - Setting the title with specified font size and bold style.
        - Hiding the x and y axis labels.
        - Adding dashed grid lines for both axes with configurable line width.

        Args:
            ax (matplotlib.axes.Axes): The axis to be formatted.
            title (str, optional): Title text for the axis. Defaults to None.
            fontsize (int, optional): Font size for the title. Defaults to 20.
            linewidth (float, optional): Width of the dashed grid lines. Defaults to 0.9.
        """
        ax.set_title(title, fontsize = fontsize, fontweight = 'bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(axis = 'y', which = 'major', linestyle = '--', linewidth = linewidth)
        ax.grid(axis = 'x', which = 'major', linestyle = '--', linewidth = linewidth)
        
        # Visual maker of normality (Skewness = 0)
        if abs(feature_skew) > 1:
            ax.set_facecolor('#fdf2e9') # Light orange background to indicate high asymmetry
        
        # Visual maker of Outliers (Pct > 5%)
        if pct_outliers > 5.0:
            ax.set_facecolor('#fffbe6')
      


    def numerical_histograms(
        self, 
        num_columns: int = 3,
        figsize_per_row: int = 6,
        color: str = '#3498db',
        hue: str = None,
        hue_order: list = None,
        palette: list = ['#1abc9c', '#ff6b6b'] ,
        title: str = 'Histograms of Numerical Variables',
    ):
        """
        Plots histograms with KDE (Kernel Density Estimation) for all numerical columns in the dataset.

        Optionally groups the histograms by a categorical target variable using different colors (hue).
        Useful for visualizing the distribution of numerical features and how they differ between groups.

        Args:
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height of each row in inches (controls vertical spacing).
            color (str): Default color for histograms when `hue` is not specified.
            hue (str, optional): Name of the column used for grouping (e.g., 'churn_target'). Must be categorical.
            palette (list): List of colors for hue levels. Only used if `hue` is provided.
            title (str): Title of the entire figure layout.

        Raises:
            Exception: If plotting fails due to missing columns, incorrect types, or rendering errors.
        """
        try:

            # Order HUE
            if hue:
                hue_order = sorted(self.data[hue].dropna().unique())

            # Entry checks
            numeric_cols = self.data.select_dtypes(include = 'number').columns.tolist()
            if hue and hue in numeric_cols:
                numeric_cols.remove(hue)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)

            
            for i, column in enumerate(numeric_cols):

                
                # Calculate Skewness for the title
                # dropna() is necessary for the statistical calculation not to fail
                feature_skew = self.data[column].dropna().skew()
                
                sns.histplot(
                    data = self.data,
                    x = column,
                    kde = True,
                    hue = hue,
                    hue_order = hue_order,
                    palette = palette if hue else None,
                    edgecolor = 'black',
                    line_kws = {'linewidth': 2},
                    alpha = 0.4 if hue else 0.7,
                    color = None if hue else color,
                    ax = ax[i],
                )

        
                # Line Means for Group
                if hue:

                    for j, target_val in enumerate(hue_order):

                        # Filter
                        subset_data = self.data[self.data[hue] == target_val][column]
                        mean_val = subset_data.mean()

                        # Color 
                        line_color = palette[j] if palette and j < len(palette) else 'black'

                        # Plot Line
                        ax[i].axvline(
                            mean_val,
                            color = line_color,
                            linestyle = '--',
                            linewidth = 2.5,
                            label = f'Mean({target_val}: {mean_val:.2f})'
                        )
                    ax[i].legend(fontsize = 10)

                # Config Ax's
                self._format_single_ax(ax[i], title = f'Variable: {column}. \n Skewness: {feature_skew:.2f}', feature_skew = feature_skew)
                

                    
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[Error] Failed to generate numeric histograms: {str(e)}.')
    

    # ======================================================== #
    # Numerical Boxplots - Function                            #
    # ======================================================== #
    def numerical_boxplots(
        self, 
        hue: str = None, 
        num_columns: int = 3,
        figsize_per_row: int = 6,
        palette: list = ['#1abc9c', '#ff6b6b'],
        color: str = '#1abc9c',
        showfliers: bool = False,
        showmeans: bool = False,
        title: str = 'Boxplots of Numerical Variables',
        legend: list = []
    ):
        """
        Plots boxplots for each numerical variable in the dataset.

        Optionally groups the boxplots by a categorical hue variable (e.g., churn target), 
        allowing for comparison of distributions between groups. Helps identify outliers, 
        skewness, and variability in each feature.

        Args:
            hue (str, optional): Column name to group the boxplots (e.g., 'churn_target').
                                If None, individual boxplots are created without grouping.
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height (in inches) of each row of plots.
            palette (list): Color palette to use when `hue` is provided.
            color (str): Single color to use when `hue` is not specified.
            showfliers (bool): Whether to display outlier points in the boxplots (default: False).
            title (str): Overall title for the subplot grid.
            legend (list): Custom legend labels to replace default tick labels when `hue` is present.

        Raises:
            ValueError: If the hue column is not found in the DataFrame.
            Exception: If plotting fails due to unexpected issues.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not in the DataFrame.")

            numeric_cols = self.data.select_dtypes(include = 'number').columns.tolist()
            if hue and hue in numeric_cols:
                numeric_cols.remove(hue)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row, h_size = 25)

            for i, column in enumerate(numeric_cols):

                    data_col = self.data[column].dropna()

                    # Calcule of Outliers (Rule of Tukey)
                    Q1 = data_col.quantile(0.25)
                    Q3 = data_col.quantile(0.75)
                    median = data_col.quantile(0.50)
                    IQR = Q3 - Q1
                    lower_fence = Q1 -1.5 * IQR
                    upper_fence = Q3 + 1.5 * IQR

                    # Count Outliers
                    outliers = data_col[(data_col < lower_fence) | (data_col > upper_fence)]
                    n_outliers = len(outliers)
                    pct_outliers = (n_outliers / len(data_col)) * 100

                    sns.boxplot(
                        data = self.data,
                        x = hue if hue else column,
                        y = column if hue else None,
                        hue = hue if hue else None,
                        palette = palette if hue else None,
                        color = None if hue else color,
                        showfliers = showfliers,
                        showmeans = showmeans,
                        #orient = 'h',
                        meanprops = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
                        fliersize = 3,
                        #legend = False,
                        ax = ax[i]
                    )

                    # Config Ax's
                    if len(legend) > 0:
                        ax[i].set_xticks([l for l in range(0, len(legend))])
                        ax[i].set_xticklabels(legend, fontsize = 14, fontweight = 'bold')
                    
                    if ax[i].get_legend():
                        ax[i].legend_.remove()
                    self._format_single_ax(ax[i], f'Variable: {column}\nOutliers: {n_outliers} ({pct_outliers:.1f}%)', pct_outliers = pct_outliers) 
                    ax[i].set_yticklabels([])
                    #sns.despine(ax = ax[i], top = True, right = True, left = True, bottom = True)
            
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e: 
            print(f'[ERROR] Failed to generate numerical boxplots: {str(e)}.')
    
    # ======================================================== #
    # Categorical Countplots - Function                        #
    # ======================================================== #
    def categorical_countplots(
        self,
        hue: str = None,
        num_columns: int = 3,
        figsize_per_row: int = 7,
        palette: list = ['#1abc9c', '#ff6b6b'],
        color: str = '#8e44ad',
        title: str = 'Countplots of Categorical Variables '
    ):
        """
        Plots countplots for all categorical variables in the dataset.

        Optionally groups the bars using a hue column (e.g., 'churn_target'), allowing 
        visual comparison of class distributions between different categories. Annotates
        each bar with its percentage frequency.

        Args:
            hue (str, optional): Name of the column used to group bars (e.g., target variable).
                                If None, no grouping is applied.
            num_columns (int): Number of plots per row in the subplot grid.
            figsize_per_row (int): Height (in inches) of each subplot row.
            palette (list): List of colors to use when `hue` is specified.
            color (str): Default color to use when `hue` is not provided.
            title (str): General title for the entire plot grid.

        Raises:
            ValueError: If the hue column is not found in the DataFrame.
            Exception: If the plot generation fails for unexpected reasons.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not found in the DataFrame.")

            categorical_cols = self.data.select_dtypes(include = ['object', 'category', 'int8']).columns.tolist()
            if hue and hue in categorical_cols :
                categorical_cols.remove(hue)
            
            # Config Ax's
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row, h_size = 30)

            for i, column in enumerate(categorical_cols):
                sns.countplot(
                    data = self.data,
                    x = column,
                    hue = hue if hue else None,
                    palette = palette if hue else None,
                    color = None if hue else color,
                    edgecolor = 'white' if hue else 'black',
                    saturation = 1,
                    alpha = 0.8,
                    legend = False,
                    ax = ax[i]
                )
                
                total = len(self.data[column])
                for p in ax[i].patches:
                    height = p.get_height()
                    if height == 0:
                        continue
                    percentage = f'{100 * height / total:.1f}%'
                    x = p.get_x() + p.get_width() / 1.95
                    y = height
                    ax[i].annotate(
                        percentage,
                        (x, y),
                        ha = 'center',
                        va = 'bottom',
                        fontsize = 16,
                        fontweight = 'bold',
                        color = 'black'
                    )

                # Config Ax's
                self._format_single_ax(ax[i], f'Variable: {column}')
                ax[i].set_xticks(range(len(ax[i].get_xticklabels())))
                ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize = 16)
                
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[ERROR] Failed to generate categorical countplots: {str(e)}')


    # ======================================================== #
    # Categorical Bar Percentages - Function                   #
    # ======================================================== #
    def categorical_bar_percentages(
        self,
        hue: str ,
        palette: list = ['#1abc9c', '#ff6b6b'],
        num_columns: int = 3,
        figsize_per_row: int = 8,
        title: str = 'Barplots Of The Individual Rate Percentages Of Each Column Class'
    ):
        """
        Plots barplots of churn percentages per class of each categorical variable.

        This method calculates the percentage distribution of a binary target (`hue`)
        within each category of all categorical columns in the dataset, and visualizes
        these percentages as barplots.

        Args:
            hue (str): Name of the binary target column (e.g., 'churn_target').
            palette (list, optional): List of colors for the hue classes.
                Defaults to ['#b0ff9d', '#db5856'].
            num_columns (int): Number of subplots per row in the grid.
            figsize_per_row (int): Height (in inches) allocated per subplot row.
            title (str): Overall title for the figure.

        Raises:
            ValueError: If `hue` is not found in the DataFrame.
            Exception: For other errors during computation or plotting.

        Returns:
            None: Displays the plot directly.
        """
        try:
            # Entry checks
            if hue and hue not in self.data.columns:
                raise ValueError(f"Column '{hue}' not found in the DataFrame.")
            categorical_cols = self.data.select_dtypes(include = ['object', 'category', 'int8']).columns.tolist()
            if hue and hue in categorical_cols:
                categorical_cols.remove(hue)

            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row, h_size = 30)

            for i, column in enumerate(categorical_cols):
                
                total_churn_per_class = self.data.groupby(column, observed = True)[hue].count().reset_index(name = f'total_count_class')

                result = (
                    self.data.groupby([column, hue], observed = True)[hue]
                    .count()
                    .reset_index(name = 'frequency')
                    .merge(total_churn_per_class, on = column)
                )
                result['percentage_per_class'] = round((result['frequency'] / result['total_count_class']) * 100, 2)

                sns.barplot(
                    data=result,
                    x = column,
                    y = 'percentage_per_class',
                    hue = hue,
                    palette = palette,
                    edgecolor = 'white',
                    saturation = 1,
                    legend = False,
                    ax = ax[i]
                )

                # Annotate bars
                for p in ax[i].patches:
                    height = p.get_height()
                    percentage = f'{height:.1f}%'
                    x = p.get_x() + p.get_width() / 2
                    ax[i].annotate(
                        percentage,
                        (x, height),
                        ha='center',
                        va='bottom',
                        fontsize=14,
                        fontweight = 'bold',
                        color='black'
                    )

                # Config Ax's
                self._format_single_ax(ax[i], f'Variable: {column}')
                ax[i].set_xticks(range(len(ax[i].get_xticklabels())))
                ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize = 16)
            
            # Show Graphics
            self._finalize_subplot_layout(fig, ax, i, title = title)
        except Exception as e:
            print(f'[ERROR] Failed to generate percentage barplots: {str(e)}.')

    # ======================================================== #
    # Target Analysis - Function                               #
    # ======================================================== #
    def plot_target_analysis(
        self, 
        target_col: str,
        title: str = 'Target Variable Distribution',
        colors: list =['#1abc9c', '#ff6b6b'],
        palette: str = 'RdYlBu_r',
        figsize: tuple = (14, 6)
    ):
        
        """
        This function generates an analytical dashboard for the target variable, combining relative and absolute views.

        This function creates a figure with two subplots:

        1. Donut Chart: To visualize class balance (%).
        2. Bar Plot: To visualize the absolute volume (N) of each class.

        The function automatically adjusts the color palette: it uses a fixed list for binary problems and a Seaborn palette for multi-class problems, ensuring visual consistency between the two graphs.

        Arguments:
        target_col (str): Name of the target column in the DataFrame.
        Ex: 'turnover', 'pattern', 'fraud'.
        title (str, optional): Main title of the panel.
        Default: 'Target Variable Distribution'.
        colors (list, optional): List of hexadecimal cores for binary problems.
        Default: ['#1abc9c' (Teal), '#ff6b6b' (Coral)].
        palette (str, optional): Seaborn palette name for multiclass problems (>2 classes).

        Default: 'RdYlBu_r'.
        figsize (tuple, optional): Figure dimensions (width, height).

        Default: (14, 6).

        Increases:
        ValueError: If `target_col` was not found in the DataFrame columns.
        Exception: For generic plotting errors (e.g., corrupted data).

        Returns:
        None: Displays the graph directly via plt.show().
        """

        try:
            # Entry checks
            if target_col not in self.data.columns:
                raise ValueError(f"Target '{target_col}' not found in Dataframe.")

            # Prepraration of Data (Ordered by Frequency: Highest -> Lowest)
            counts = self.data[target_col].value_counts()
            labels = counts.index
            values = counts.values

            # Color Logic
            if len(labels) <=2:
                final_colors = colors
            else:
                final_colors = sns.color_palette(palette, n_colors = len(labels))

            # Define AX and Fig
            fig, ax = plt.subplots(1, 2, figsize = figsize)
            plt.suptitle(title, fontsize = 20, fontweight = 'bold', y = 1.02)

            # Plot 1: Donut Chart
            wedges, texts, autotexts = ax[0].pie(
                values, 
                labels = labels,
                autopct = '%1.1f%%',
                startangle = 90,
                colors = final_colors,
                pctdistance = 0.80,
                wedgeprops =  dict(width = 0.4, edgecolor = 'white'),
                textprops = {'fontsize': 14}
            )

            ax[0].set_title(f'Class Balance (%)', fontsize = 16, fontweight = 'bold')
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_weight('bold')

            # Plot 2: Bar Plot
            sns.barplot(
                x = labels,
                y = values,
                hue = labels,
                palette = final_colors,
                order = labels,
                hue_order = labels,
                legend = False,
                ax = ax[1],
                edgecolor = 'black'
            )

            ax[1].set_title(f'Absolute Frequecy (N)', fontsize = 16, fontweight = 'bold')
            ax[1].set_ylabel('Count') 
            ax[1].grid(axis = 'y', linestyle = '--', alpha = 0.4)
            ax[1].grid(axis = 'x', linestyle = '--', alpha = 0.4)

            # Annotate of values for blarplot
            for p in ax[1].patches:
                height = p.get_height()
                if height > 0:
                    ax[1].annotate(
                    f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha = 'center', 
                    va = 'bottom',
                    fontsize = 14,
                    fontweight = 'bold',
                    color = 'black'
                    )
            
            plt.tight_layout(rect = [0, 0, 1, 0.97])
            plt.show()
        except Exception as e:
            print(f'[ERROR] Failed to plot target analysis: {str(e)}')

    # ======================================================== #
    # Correlation Heatmap - Function                           #
    # ======================================================== #
    def correlation_heatmap(
        self,
        title: str = None,
        cmap: str = 'RdBu_r',
        h_size: int = 20,
        v_size: int = 15
    ):
        """
        Plots a heatmap showing the correlation matrix among the numerical columns.

        This method computes the correlation matrix of the dataset and displays it as a heatmap,
        with annotations showing the correlation coefficients.

        Args:
            title (str, optional): Title for the heatmap plot. Defaults to None.
            cmap (str, optional): Colormap to use for the heatmap. Defaults to 'coolwarm'.

        Raises:
            Exception: If the heatmap generation or plotting fails.
        """
        try:
            # Select only the desired columns
            corr_data = self.data.corr()
            sns.set_style('white')
            # Define AX and Fig
            plt.rc('font', size = 15)
            fig, ax = plt.subplots(figsize = (h_size, v_size))
            
            sns.heatmap(
                corr_data,
                mask = np.triu(np.ones_like(corr_data, dtype = bool)),
                annot = True,
                cmap = cmap,
                center =  0,
                vmax = 1,
                vmin = -1,
                fmt = '.2f',
                linewidths = .5,
                cbar_kws = {'shrink': .8},
                ax = ax
            )
            # Config Ax's and Show Graphics
            ax.set_title(title, fontsize = 20, fontweight = 'bold')
            plt.tight_layout(rect = [0, 0, 1, 0.97])
            plt.show()
        except Exception as e:
            print(f'[Error] Failed to generate correlation heatmap: {str(e)}.')


    # ======================================================== #
    # Models Performance Barplots - Function                   #
    # ======================================================== #
    def models_performance_barplots(
        self,
        models_col: str = None,
        palette = None,
        title: str = 'Models Performance Comparison',
        num_columns: int = 1,
        figsize_per_row: int = 9
    ):
        """
        Generates bar plots to compare the performance of multiple models across different metrics.

        Args:
            models_col (str, optional): Column name containing model identifiers.
            palette (list or seaborn color palette, optional): Color palette for the bar plots.
                Defaults to a 'viridis' palette if None.
            title (str, optional): Main title for the figure. Defaults to 'Models Performance Comparison'.
            num_columns (int, optional): Number of subplot columns. Defaults to 1.
            figsize_per_row (int, optional): Height of each subplot row in inches. Defaults to 9.

        Raises:
            Exception: If there is an error generating the bar plots.

        Notes:
            - The method expects `self.data` to contain one column for model names
            and one or more columns with numeric performance metrics.
            - Each subplot will represent a different metric.
        """
        try:

            # Define palette
            if palette is None:
                palette = sns.color_palette('viridis', len(self.data[models_col].unique()))
            
            # Define AX and Fig
            fig, ax = self._initializer_subplot_grid(num_columns, figsize_per_row)
            # Ax Flatten
            ax = ax.flatten()

            # Iterate over metrics (excluding 'Model')
            for i, column in enumerate(self.data.drop(columns = models_col).columns):

                barplot = sns.barplot(
                    data = self.data,
                    x = models_col,
                    y = column,
                    hue = models_col,
                    dodge = False,
                    edgecolor = 'white',
                    saturation = 1,
                    palette = palette,
                    ax = ax[i]
                )

                # Formatting axis
                self._format_single_ax(ax[i], title = column, fontsize = 25)
                ax[i].tick_params(axis = 'x', labelsize = 20)
                ax[i].set_yticklabels([])
                sns.set(style = 'whitegrid')

                # Add values on bars
                for v in barplot.patches:
                    barplot.annotate(
                        f'{v.get_height():.4f}',
                        (v.get_x() + v.get_width() / 2., v.get_height() / 1.06),
                        ha = 'center',
                        va = 'top',
                        xytext = (0, 0),
                        textcoords = 'offset points',
                        fontsize = 20,
                        fontweight = 'bold',
                        color = 'white'
                    )

            # Finalize plot
            self._finalize_subplot_layout(fig, ax, i, title = title)     
        
        except Exception as e:
                print(f'[ERROR] Failed to generate model performance barplots: {str(e)}.')

    # ======================================================== #
    # Plot KDE Predictions - Function                          #
    # ======================================================== #                
    def plot_kde_predictions(
        self,
        palette: list = ['#12e193', '#feb308'],
        predictions: str = None,
        labels: str = None,
        title: str = 'Prediction Probabilities'
    ):
        """
        Plots the probability distributions of predictions using Kernel Density Estimation (KDE).

        Args:
            palette (list, optional): List of colors for each class. Defaults to ['#12e193', '#feb308'].
            predictions (str, optional): Column name containing the predicted probabilities.
            labels (str, optional): Column name containing the true labels.
            title (str, optional): Title for the plot. Defaults to 'Prediction Probabilities'.

        Raises:
            Exception: If there is an error generating the KDE plot.

        Notes:
            - The method expects `self.data` to contain the prediction probabilities and true labels.
            - KDE plots are useful for visualizing class separation in probabilistic predictions.
        """
        try:

            # Creating figures and setting font size
            plt.rc('font', size = 10)
            fig, ax = plt.subplots(figsize = (12, 4))

            sns.kdeplot(
                data = self.data,
                x = predictions,
                hue = labels,
                fill = True,
                alpha = 0.4,
                bw_adjust = 1,
                palette = palette,
                linewidth = 1,
                ax = ax
            )

            # Axis and title adjustments
            ax.set_title(title, fontsize = 14)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_yticklabels([])

            ## Grade and style
            ax.grid(axis = 'y', linestyle = '--', linewidth = 0.3)
            ax.grid(axis = 'x', linestyle = '--', linewidth = 0.3)
            sns.set(style = 'whitegrid')
            sns.despine(ax = ax, top = True, right = True, left = True, bottom = False)

            # Show Graphics
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f'[ERROR] Failed to generate prediction probability graph {str(e)}.')

    # ======================================================== #
    # Plot ROC and Precision Curves Function                   #
    # ======================================================== #   
    def plot_roc_pr_curves(
        preds, 
        labels
    ):
        """
        Plots the Receiver Operating Characteristic (ROC) curve and the Precision-Recall (PR) curve
        for a binary classification model, along with their respective AUC metrics.

        Args:
            preds (Tensor or array-like): Predicted probabilities for the positive class.
            labels (Tensor or array-like): True binary labels (0 or 1).

        Raises:
            Exception: If there is an error generating or plotting the curves.

        Notes:
            - Computes and plots:
                * ROC curve with AUROC (Area Under ROC Curve).
                * Precision-Recall curve with AUPRC (Area Under Precision-Recall Curve).
            - Uses TorchMetrics for metric computation.
            - Useful for evaluating model performance, especially in imbalanced datasets.
        """
        try:
            # Metrics

            # ROC
            fpr, tpr, _ = BinaryROC()(preds, labels)
            auroc_value = BinaryAUROC()(preds, labels).item()

            # Precision-Recall
            precision_vals, recall_vals, _ = BinaryPrecisionRecallCurve()(preds, labels)
            auprc_value = BinaryAveragePrecision()(preds, labels).item()

            # Ax and Fig
            fig, axes = plt.subplots(1, 2, figsize = (12, 5))

            # ROC Curve
            axes[0].plot(fpr, tpr, color = '#1f77b4', lw = 2, label = f'ROC (AUROC = {auroc_value:.3f})')
            axes[0].plot([0, 1], [0, 1], color = 'gray', linestyle = '--', label = 'Random (AUROC = 0.5)')
            axes[0].set_xlabel('False Positive Rate', fontsize = 12)
            axes[0].set_ylabel('True Positive Rate', fontsize = 12)
            axes[0].set_title('ROC Curve', fontsize = 14, fontweight = 'bold')
            axes[0].legend(loc = 'lower right', fontsize = 10)

            # PR Curve
            axes[1].plot(recall_vals, precision_vals, color = 'darkorange', lw = 2, label = f'PR Curve (AUPRC = {auprc_value:.3f})')
            axes[1].set_xlabel('Recall', fontsize = 12)
            axes[1].set_ylabel('Precision', fontsize = 12)
            axes[1].set_title('Precision-Recall Curve', fontsize = 14, fontweight = 'bold')
            axes[1].legend(loc='upper right', fontsize = 10)

            # Grid
            for ax in axes:
                ax.grid(True, linestyle = '--', linewidth = 0.5)

            # Show Graphics
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f'[ERROR] Failed to generate ROC and PR curves: {str(e)}.')
