import requests
import pandas as pd

# URL of the JSON data
url = "https://freeslotmania.com/stats.json"

# Fetch the JSON data from the URL
response = requests.get(url)
data = response.json()

# Conversion mapping for the result values
result_conversion = {
    1: "one",
    2: "two",
    10: "ten",
    5: "five",
    400: "crazytime",
    300: "coinflip",
    200: "pachinko",
    100: "cashhunt"
}

# Extract the first 5 entries of result, multiplier, and total_payout
extracted_data = []
for entry in data['data'][:5]:
    result = entry.get('result')
    multiplier = entry.get('multiplier')
    total_payout = entry.get('total_payout')
    # Convert the result to its corresponding word
    result_word = result_conversion.get(result, "unknown")
    extracted_data.append({
        "result": result_word,
        "multiplier": multiplier,
        "total_payout": total_payout
    })

# Create a DataFrame to display the data
df = pd.DataFrame(extracted_data)

# Print the DataFrame
print(df)
