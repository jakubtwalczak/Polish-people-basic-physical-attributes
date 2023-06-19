# Imports.

import pandas as pd
import numpy as np
import random
import scipy.stats as scs
import seaborn as sns
import matplotlib.pyplot as plt

# Class to generate distributions.

class Distribution:
    """
    A class for generating samples from different probability distributions.

    Attributes:
        avg (float): The average value of the distribution.
        stdev (float): The standard deviation of the distribution.
        samples (int): The number of samples to generate.
        degrees_of_freedom (int): The degrees of freedom for the t-distribution (samples - 1).
        distribution_type (str): The type of distribution to generate samples from (default: "normal").
        skewness (float): The skewness parameter for the skewnorm distribution.

    Methods:
        generate_samples(): Generates samples from the specified distribution.

    """

    def __init__(self, avg, stdev, samples):
        """
        Initialize a Distribution object.

        Args:
            avg (float): The average value of the distribution.
            stdev (float): The standard deviation of the distribution.
            samples (int): The number of samples to generate.

        """
        self.avg = avg
        self.stdev = stdev
        self.samples = samples
        self._degrees_of_freedom = samples - 1
        self._distribution_type = "normal"
        self._skewness = None

    @property
    def degrees_of_freedom(self):
        """
        Get the degrees of freedom for the t-distribution.

        Returns:
            int: The degrees of freedom.

        """
        return self._degrees_of_freedom

    @property
    def distribution_type(self):
        """
        Get the current distribution type.

        Returns:
            str: The distribution type.

        """
        return self._distribution_type

    @distribution_type.setter
    def distribution_type(self, value):
        """
        Set the distribution type.

        Args:
            value (str): The distribution type.

        Raises:
            ValueError: If an invalid distribution type is provided.

        """
        if value.lower() in ["normal", "t", "skewnorm"]:
            self._distribution_type = value.lower()
        else:
            raise ValueError("Invalid distribution type. Available options: normal, t, skewnorm.")

    @property
    def skewness(self):
        """
        Get the skewness parameter for the skewnorm distribution.

        Returns:
            float: The skewness parameter.

        """
        return self._skewness

    @skewness.setter
    def skewness(self, value):
        """
        Set the skewness parameter for the skewnorm distribution.

        Args:
            value (float): The skewness parameter.

        """
        self._skewness = value

    def generate_samples(self):
        """
        Generate samples from the specified distribution.

        Returns:
            list: The generated samples.

        """
        distribution = []
        for i in range(self.samples):
            random_p = random.random()
            if self.distribution_type == "normal":
                random_value = scs.norm.ppf(random_p, loc=self.avg, scale=self.stdev)
            elif self.distribution_type == "t":
                random_value = scs.t.ppf(random_p, df=self.degrees_of_freedom, loc=self.avg, scale=self.stdev)
            elif self.distribution_type == "skewnorm":
                random_value = scs.skewnorm.ppf(random_p, self.skewness, loc=self.avg, scale=self.stdev)
            distribution.append(random_value)
        return distribution


# Generating male data.


male_avg_height = 180.69
male_stdev_height = 7.048
male_avg_bmi = 27.4
male_stdev_bmi = 4

heights_distribution_male = Distribution(male_avg_height, male_stdev_height, 5000)
bmis_distribution_male = Distribution(male_avg_bmi, male_stdev_bmi, 5000)

heights_distribution_male.distribution_type = "t"

bmis_distribution_male.distribution_type = "t"

heights_male = heights_distribution_male.generate_samples()
bmis_male = bmis_distribution_male.generate_samples()


data_m = {
    'Height': heights_male,
    'BMI': bmis_male,
    'Sex': 'Male'
}

df_male = pd.DataFrame(data_m)
print(df_male)


df_male['Height'] = round(df_male['Height'], 1)
df_male['BMI'] = round(df_male['BMI'], 2)
print(df_male)

df_male['Weight'] = round(df_male['BMI'] * (((df_male['Height'])/100)**2), 1)
df_male

print(df_male.info())

print(df_male.describe())


# Generating female data.

female_avg_height = 165.78
female_stdev_height = 6.535
female_avg_bmi = 26.1
female_stdev_bmi = 3.57

heights_distribution_female = Distribution(female_avg_height, female_stdev_height, 5000)
bmis_distribution_female = Distribution(female_avg_bmi, female_stdev_bmi, 5000)

heights_distribution_female.distribution_type = "t"

bmis_distribution_female.distribution_type = "t"

heights_female = heights_distribution_female.generate_samples()
bmis_female = bmis_distribution_female.generate_samples()

data_f = {
    'Height': heights_female,
    'BMI': bmis_female,
    'Sex': 'Female'
}

df_female = pd.DataFrame(data_f)
df_female['Height'] = round(df_female['Height'], 1)
df_female['BMI'] = round(df_female['BMI'], 2)
df_female['Weight'] = round(df_female['BMI'] * (((df_female['Height'])/100)**2), 1)
print(df_female)

print(df_female.info())

print(df_female.describe())

# Joining the dataframes.

df = pd.concat([df_male, df_female])
print(df)

print(df.info())

print(df.describe())


df.to_csv('Polish_height_weight_bmi.csv', index=False)

df.reset_index(drop=True, inplace=True)
print(df)


# Visualization and analysis.

continuous_cols = list(df.drop(columns='Sex').columns)
print(continuous_cols)


# KDE plots.

rotation = 45
legend = (1, 1)
figsize = (6, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.kdeplot(x=col, data=df,
                     hue='Sex', fill=True, multiple='layer')
    sns.move_legend(ax, "upper left", bbox_to_anchor=legend)
    plt.xticks(rotation=rotation)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# ECDF plots.

rotation = 45
legend = (1, 1)
figsize = (6, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.ecdfplot(x=col, data=df_male)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.title(f'CDF of {col} in male set')
    median = df_male[col].median()
    plt.axvline(median, color='red', linestyle='--')
    plt.axhline(0.5, color='blue', linestyle='--')

plt.tight_layout()
plt.show()


rotation = 45
legend = (1, 1)
figsize = (6, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.ecdfplot(x=col, data=df_female)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.title(f'CDF of {col} in female set')
    median = df_female[col].median()
    plt.axvline(median, color='red', linestyle='--')
    plt.axhline(0.5, color='blue', linestyle='--')

plt.tight_layout()
plt.show()


# Boxplots.


rotation = 45
figsize = (4, 5)

fig, axes = plt.subplots(1, 3, figsize=(12, 5))  

for i, col in enumerate(continuous_cols):
    ax = axes[i]
    ax = sns.boxplot(x='Sex', y=col, data=df, fliersize=3, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# Scatterplots.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax1 = axes[0]
sns.scatterplot(x='Height', y='Weight', hue='Sex', data=df, ax=ax1)
sns.regplot(x='Height', y='Weight', data=df[df['Sex'] == 'Male'], ax=ax1, line_kws={'linestyle': 'solid', 'color': 'green'})
sns.regplot(x='Height', y='Weight', data=df[df['Sex'] == 'Female'], ax=ax1, line_kws={'linestyle': 'solid', 'color': 'purple'})
ax1.set_xlabel('Height')
ax1.set_ylabel('Weight')
ax1.set_title('Weight vs Height')
ax1.legend()

ax2 = axes[1]
sns.scatterplot(x='Height', y='BMI', hue='Sex', data=df, ax=ax2)
sns.regplot(x='Height', y='BMI', data=df[df['Sex'] == 'Male'], ax=ax2, line_kws={'linestyle': 'solid', 'color': 'green'})
sns.regplot(x='Height', y='BMI', data=df[df['Sex'] == 'Female'], ax=ax2, line_kws={'linestyle': 'solid', 'color': 'purple'})
ax2.set_xlabel('Height')
ax2.set_ylabel('BMI')
ax2.set_title('BMI vs Height')
ax2.legend()

ax3 = axes[2]
sns.scatterplot(x='Weight', y='BMI', hue='Sex', data=df, ax=ax3)
sns.regplot(x='Weight', y='BMI', data=df[df['Sex'] == 'Male'], ax=ax3, line_kws={'linestyle': 'solid', 'color': 'green'})
sns.regplot(x='Weight', y='BMI', data=df[df['Sex'] == 'Female'], ax=ax3, line_kws={'linestyle': 'solid', 'color': 'purple'})
ax3.set_xlabel('Weight')
ax3.set_ylabel('BMI')
ax3.set_title('BMI vs Weight')
ax3.legend()

plt.tight_layout()
plt.show()


# Correlation matrices.

correlation_matrix = df_male[continuous_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of correlations - male set')
plt.show()


correlation_matrix = df_female[continuous_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of correlations - female set')
plt.show()


# Descriptive statistics plots.

three_sigmas = [0.0015, 0.0235, 0.16, 0.5, 0.84, 0.975, 0.9985]

plt.figure(figsize=(10, 6))
sns.heatmap(df_male.describe(percentiles=three_sigmas).drop('count'), annot=True, fmt=".3f", cmap='coolwarm')
plt.xlabel('Statistic')
plt.ylabel('Variable')
plt.title('Male set descriptive statistics')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df_female.describe(percentiles=three_sigmas).drop('count'), annot=True, fmt=".3f", cmap='coolwarm')
plt.xlabel('Statistic')
plt.ylabel('Variable')
plt.title('Female set descriptive statistics')
plt.show()


# Definition of analytic functions.

def normality_testing(data, columns: list):
    for col in columns:
        column_data = data[col]
        stat_sw, p_value_sw = scs.shapiro(column_data)
        stat_ap, p_value_ap = scs.normaltest(column_data)
        stats_and = scs.anderson(column_data, dist='norm')
        skewness = scs.skew(column_data)
        kurtosis = scs.kurtosis(column_data)
    
        print(f"Column '{col}':\n")
        print(f"Shapiro-Wilk test statistic: {stat_sw}")
        print(f"Shapiro-Wilk p-value: {p_value_sw}")
        print("------------------------------------")
        print(f"D'Agostino-Pearson test statistic: {stat_ap}")
        print(f"D'Agostino-Pearson p-value: {p_value_ap}")
        print("------------------------------------")
        print(f"Anderson-Darling test statistic: {stats_and.statistic}")
        print(f"Anderson-Darling critical values: {stats_and.critical_values}")
        print(f"Anderson-Darling significance level: {stats_and.significance_level}")
        print("------------------------------------")
        print(f"Skewness: {skewness}")
        print(f"Kurtosis: {kurtosis}")
        print("------------------------------------\n")
        
def distributions_comparison(x, y, columns: list):
    for col in columns:
        data_x = x[col]
        data_y = y[col]
        stat, p_value = scs.ttest_ind(data_x, data_y)
        ks_stat, ks_p_value = scs.ks_2samp(data_x, data_y)
        print(f"Column {col}:")
        print(f"T-test statistic: {stat}")
        print(f"T-test p-value: {p_value}")
        print("------------------------------------")
        print(f"KS test statistic: {ks_stat}")
        print(f"KS test p-value: {ks_p_value}")
        print("------------------------------------\n")



normality_testing(df_male, continuous_cols)

normality_testing(df_female, continuous_cols)

distributions_comparison(df_male, df_female, continuous_cols)


# Comparison - football teams vs dataset.
# Male football team.

male_polish_football_team = {
    'Height': [196, 187, 195, 185, 189, 187, 189, 189, 190, 183,
               182, 189, 185, 187, 185, 174, 180, 175, 186, 185,
               180, 179, 184, 183, 176, 173],
    'Weight': [92, 83, 87, 79, 83, 80, 84, 76, 86, 75, 74, 83, 75,
               82, 79, 62, 76, 68, 82, 79, 76, 69, 74, 78, 68, 66]
}

df_mfoot = pd.DataFrame.from_dict(male_polish_football_team)
print(df_mfoot)


df_mfoot['BMI'] = df_mfoot['Weight'] / ((df_mfoot['Height']/100)**2)
df_mfoot['BMI'] = round(df_mfoot['BMI'], 2)
print(df_mfoot)


plt.figure(figsize=(10, 6))
sns.heatmap(df_mfoot.describe().drop('count'), annot=True, fmt=".3f", cmap='coolwarm')
plt.xlabel('Statistic')
plt.ylabel('Variable')
plt.title("Male footballers' set descriptive statistics")
plt.show()


rotation = 45
legend = (1, 1)
figsize = (5, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.histplot(x=col, data=df_mfoot, bins=8)
    plt.xticks(rotation=rotation)
    plt.title(f'Histogram of {col} - male footballers')

plt.tight_layout()
plt.show()


rotation = 45
legend = (1, 1)
figsize = (6, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.ecdfplot(x=col, data=df_mfoot)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.title(f"CDF of {col} in male footballers' set")
    median = df_mfoot[col].median()
    plt.axvline(median, color='red', linestyle='--')
    plt.axhline(0.5, color='blue', linestyle='--')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax1 = axes[0]
sns.scatterplot(x='Height', y='Weight', data=df_mfoot, ax=ax1)
sns.regplot(x='Height', y='Weight', data=df_mfoot, ax=ax1, line_kws={'linestyle': 'solid', 'color': 'green'})
ax1.set_xlabel('Height')
ax1.set_ylabel('Weight')
ax1.set_title('Weight vs Height')

ax2 = axes[1]
sns.scatterplot(x='Height', y='BMI', data=df_mfoot, ax=ax2)
sns.regplot(x='Height', y='BMI', data=df_mfoot, ax=ax2, line_kws={'linestyle': 'solid', 'color': 'green'})
ax2.set_xlabel('Height')
ax2.set_ylabel('BMI')
ax2.set_title('BMI vs Height')

ax3 = axes[2]
sns.scatterplot(x='Weight', y='BMI', data=df_mfoot, ax=ax3)
sns.regplot(x='Weight', y='BMI', data=df_mfoot, ax=ax3, line_kws={'linestyle': 'solid', 'color': 'green'})
ax3.set_xlabel('Weight')
ax3.set_ylabel('BMI')
ax3.set_title('BMI vs Weight')

plt.tight_layout()
plt.show()


correlation_matrix = df_mfoot[continuous_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of correlations - male set')
plt.show()

normality_testing(df_mfoot, continuous_cols)

distributions_comparison(df_mfoot, df_male, continuous_cols)

# Female football team.

female_best_polish_footballers = {
    'Height': [174, 170, 170, 174, 160, 161, 170, 162, 174, 166,
               160, 158, 172, 171, 180, 180, 166, 183, 169, 168,
               170, 170, 183, 168, 168, 154, 167], 
    'Weight': [76, 56, 60, 61, 51, 52, 55, 59, 58, 57, 50, 50,
               58, 58, 63, 62, 59, 69, 58, 56, 60, 68, 69, 56, 
               56, 50, 58]
}

df_femfoot = pd.DataFrame.from_dict(female_best_polish_footballers)
print(df_femfoot)

df_femfoot['BMI'] = df_femfoot['Weight'] / ((df_femfoot['Height']/100)**2)
df_femfoot['BMI'] = round(df_femfoot['BMI'], 2)
print(df_femfoot)


plt.figure(figsize=(10, 6))
sns.heatmap(df_femfoot.describe().drop('count'), annot=True, fmt=".3f", cmap='coolwarm')
plt.xlabel('Statistic')
plt.ylabel('Variable')
plt.title("Female footballers' set descriptive statistics")
plt.show()

rotation = 45
legend = (1, 1)
figsize = (5, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.histplot(x=col, data=df_femfoot, bins=8)
    plt.xticks(rotation=rotation)
    plt.title(f'Histogram of {col} - female footballers')

plt.tight_layout()
plt.show()


rotation = 45
legend = (1, 1)
figsize = (6, 3.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 3.5)) 

for i, col in enumerate(continuous_cols):
    plt.subplot(1, 3, i+1) 
    ax = sns.ecdfplot(x=col, data=df_femfoot)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.title(f"CDF of {col} in female footballers' set")
    median = df_femfoot[col].median()
    plt.axvline(median, color='red', linestyle='--')
    plt.axhline(0.5, color='blue', linestyle='--')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax1 = axes[0]
sns.scatterplot(x='Height', y='Weight', data=df_femfoot, ax=ax1)
sns.regplot(x='Height', y='Weight', data=df_femfoot, ax=ax1, line_kws={'linestyle': 'solid', 'color': 'green'})
ax1.set_xlabel('Height')
ax1.set_ylabel('Weight')
ax1.set_title('Weight vs Height')

ax2 = axes[1]
sns.scatterplot(x='Height', y='BMI', data=df_femfoot, ax=ax2)
sns.regplot(x='Height', y='BMI', data=df_femfoot, ax=ax2, line_kws={'linestyle': 'solid', 'color': 'green'})
ax2.set_xlabel('Height')
ax2.set_ylabel('BMI')
ax2.set_title('BMI vs Height')

ax3 = axes[2]
sns.scatterplot(x='Weight', y='BMI', data=df_femfoot, ax=ax3)
sns.regplot(x='Weight', y='BMI', data=df_femfoot, ax=ax3, line_kws={'linestyle': 'solid', 'color': 'green'})
ax3.set_xlabel('Weight')
ax3.set_ylabel('BMI')
ax3.set_title('BMI vs Weight')

plt.tight_layout()
plt.show()


correlation_matrix = df_femfoot[continuous_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of correlations - male set')
plt.show()


normality_testing(df_femfoot, continuous_cols)


distributions_comparison(df_femfoot, df_female, continuous_cols)
