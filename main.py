import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print("Sample Data:")
print(titanic_data.head())
print()

# Basic statistics
print("Basic Statistics:")
print(titanic_data.describe())
print()

# Extracting deck information from the Cabin column
titanic_data['Deck'] = titanic_data['Cabin'].str.slice(0, 1)

# Survival Rate by Gender and Pclass (Stacked Bar Chart)
plt.figure(figsize=(10, 6))
# Group by Passenger Class and Gender to find Survival Rate
survival_by_class_gender = titanic_data.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
# Plot the data as a stacked bar chart
survival_by_class_gender.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Survival Rate by Gender and Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.legend(title='Gender')
plt.xticks(rotation=0)
plt.show()

# Age Distribution by Gender (Histogram)
plt.figure(figsize=(10, 6))
# Plotting histograms of Age distribution for each Gender
for gender in titanic_data['Sex'].unique():
    subset = titanic_data[titanic_data['Sex'] == gender]
    age = subset['Age'].dropna()
    plt.hist(age, bins=30, density=True, alpha=0.5, label=gender)

plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# Fare and Age Relationship (Scatter Plot)
plt.figure(figsize=(8, 6))
# Scatter plot of Fare against Age for each Passenger Class
classes = titanic_data['Pclass'].unique()
colors = ['orange', 'blue', 'green']
for i, passenger_class in enumerate(classes):
    subset = titanic_data[titanic_data['Pclass'] == passenger_class]
    plt.scatter(subset['Age'], subset['Fare'], label=passenger_class, alpha=0.5, color=colors[i])

plt.title('Fare and Age Relationship')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Class')
plt.show()

# Boxplot of Fare by Pclass and Embarked
plt.figure(figsize=(10, 6))
# Creating boxplots of Fare for each Class and Embarkation location
classes = titanic_data['Pclass'].unique()
embarked_locs = titanic_data['Embarked'].unique()
colors = ['orange', 'blue', 'green']
for i, passenger_class in enumerate(classes):
    for j, embarked_loc in enumerate(embarked_locs):
        subset = titanic_data[(titanic_data['Pclass'] == passenger_class) & (titanic_data['Embarked'] == embarked_loc)]
        fares = subset['Fare'].dropna()
        plt.boxplot(fares, positions=[i + j * 0.3], widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]))

plt.title('Fare by Class and Embarked')
plt.xlabel('Class')
plt.ylabel('Fare')
plt.xticks(range(len(classes)), classes)
plt.legend(embarked_locs, title='Embarked')
plt.show()

# Countplot of survivors by Deck
plt.figure(figsize=(8, 6))
# Count of Passengers by Deck based on Cabin information
deck_counts = titanic_data['Deck'].value_counts()
plt.bar(deck_counts.index, deck_counts.values, color='purple')
plt.title('Count of Passengers by Deck')
plt.xlabel('Deck')
plt.ylabel('Count')
plt.show()

# Pairplot of selected features (Age, Fare, Pclass) with Survived as hue
plt.figure(figsize=(10, 6))
# Pairwise scatter plots of selected features with Survived as hue
selected_features = ['Age', 'Fare', 'Pclass']
colors = ['red' if val == 1 else 'blue' for val in titanic_data['Survived']]
for i, feature in enumerate(selected_features):
    for j, other_feature in enumerate(selected_features):
        if i != j:
            plt.scatter(titanic_data[feature], titanic_data[other_feature], c=colors, alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel(other_feature)

plt.title('Pair Plot of Selected Features')
plt.show()
