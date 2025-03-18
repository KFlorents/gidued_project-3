import csv
import random
from datetime import datetime, timedelta
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Generate data for 50 characters
NUM_ROWS = 1000

# Create the CSV file
OUTPUT_FILE = "troop_movements.csv"


def choose_a_side(home_world):
    """
    Randomly choose an empire or resistance side based on the likelihood 
    that someone from that world would join the rebel alliance.
    Args:
        home_world (dict): The home world data for the character.
    Returns:
        str: The empire or resistance side.
    """
    if home_world["rebel_likelihood"] > random.random():
        return "resistance"
    else:
        return "empire"


# Load home world data from JSON file
with open("home_worlds.json") as json_file:
    home_worlds = json.load(json_file)

# Generate data rows
data_rows = []
for i in range(1, NUM_ROWS + 1):
    # Generate random values for each column
    timestamp = datetime.now() - timedelta(seconds=i)
    unit_id = i
    unit_type = random.choice(
        ["stormtrooper", "tie_fighter", "at-st", "x-wing",
            "resistance_soldier", "at-at", "tie_silencer", "unknown"]
    )
    location_x = random.randint(1, 10)
    location_y = random.randint(1, 10)
    destination_x = random.randint(1, 10)
    destination_y = random.randint(1, 10)

    # Select a random home world from the available options
    home_world = random.choice(home_worlds)
    home_world_name = home_world["name"]
    empire_or_resistance = choose_a_side(home_world)

    # Create the data row
    data_row = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        unit_id,
        unit_type,
        empire_or_resistance,
        location_x,
        location_y,
        destination_x,
        destination_y,
        home_world_name,
    ]

    # Add the data row to the list
    data_rows.append(data_row)

# Write the data to the CSV file
with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["timestamp", "unit_id", "unit_type", "empire_or_resistance", "location_x", "location_y", "destination_x",
         "destination_y", "homeworld"]
    )
    writer.writerows(data_rows)

print("Data generation complete.")


#Reading data from csv into pandas df
df = pd.read_csv('troop_movements.csv')

#groupe data for counts of empire vs resistence
print (df)


grouped_empire_res = df["empire_or_resistance"].value_counts().reset_index()
grouped_count_homeworld = df["homeworld"].value_counts().reset_index()
grouped_count_unit_type = df["unit_type"].value_counts().reset_index()
print(grouped_empire_res)
print(grouped_count_homeworld)
print(grouped_count_unit_type)

# Create a new feature 'is_resistance'
df["is_resistance"] = df["empire_or_resistance"] == "resistance"

print(df)


plt.figure(figsize=(8,6))
sns.barplot(data=grouped_empire_res, x="empire_or_resistance", y="count", palette="viridis")

plt.title("Distribution of Empire vs Resistance")
plt.xlabel("empire_or_resistance")
plt.ylabel("count")

# plt.show()


# Prediction model
X_encoded = pd.get_dummies(df[["homeworld", "unit_type"]])
y = df["empire_or_resistance"].map({"empire": 0, "resistance": 1})

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=["Empire", "Resistance"])


# Get feature importances
importances = model.feature_importances_
feature_importances = pd.DataFrame({"Feature": X_encoded.columns, "Importance": importances})

# Sort features
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

#Creat a bar plot for feature importances
plt.figure(figsize=(10,6))
sns.barplot(data=feature_importances, x="Feature", y="Importance", palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=90)
plt.show()

print(f"Accuracy: {accuracy:.2f} ")
print("\nClassification Report:\n", classification_rep)

# Save the model
model_filename = "trained_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved successfully as {model_filename}")


