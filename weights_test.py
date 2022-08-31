import json

with open("theme_weights.json", "r") as f:
    data = json.load(f)

print(data.keys())

print(data["Naive"])