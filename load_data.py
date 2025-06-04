import pandas as pd

# Step 1: Load your dataset
df = pd.read_csv('your_data.csv')

# Step 2: Combine symptoms into one string per row
df['all_symptoms'] = df[df.columns[:-1]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

# Step 3: Keep only needed columns
df_clean = df[['all_symptoms', 'disease']]

# ✅ Step 4: Add extra manually created variations
extra_data = [
    {"all_symptoms": "fever,headache,cough", "disease": "Common Cold"},
    {"all_symptoms": "high temperature,head pain,throat irritation", "disease": "Common Cold"},
    {"all_symptoms": "breathing difficulty,wheezing", "disease": "Asthma"},
    {"all_symptoms": "chest pain,shortness of breath", "disease": "Heart Disease"}
]

# Convert to DataFrame and append
extra_df = pd.DataFrame(extra_data)
df_clean = pd.concat([df_clean, extra_df], ignore_index=True)

# Step 5: Save to CSV for use in training
df_clean.to_csv('cleaned_data.csv', index=False)

# Preview
print(df_clean.head())
