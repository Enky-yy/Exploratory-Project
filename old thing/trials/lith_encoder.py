import joblib
from sklearn.preprocessing import LabelEncoder

# Define lithology categories (include all types used in your dataset)
lithologies = [
    "Limestone",
    "Sandstone",
    "Shale",
    "Granite",
    "Basalt",
    "Marble",
    "Dolomite",
    "Slate",
    "Quartzite"
]

# Initialize LabelEncoder
lith_encoder = LabelEncoder()

# Fit encoder on lithology list
lith_encoder.fit(lithologies)

# Save encoder to file
joblib.dump(lith_encoder, "lith_encoder.pkl")

print("✅ lith_encoder.pkl saved successfully!")
print("Classes:", list(lith_encoder.classes_))
