# %%
import requests
import yaml
import pandas as pd
import os

## PATH
main_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code")

export_path = os.path.join(main_dir, 
                           "ASN-DSA-T5", 
                           "33-ST",
                           "Homework",
                           "data",
                           "holidays.csv")

# %%
# load the credentials from YAML file
with open(os.path.join(main_dir, 'credentials.yaml'), 'r') as file:
    credentials = yaml.safe_load(file)

# extract the API key
api_key = credentials['feriados_api']['key']


# %%
# Map IBGE codes to their respective states
ibge_state_map = {
    '3550308': 'SP',  # São Paulo
    '3304557': 'RJ',  # Rio de Janeiro
    3106200: 'MG',  # Belo Horizonte
    4106902: 'PR',  # Curitiba
    4205407: 'SC',  # Florianópolis
    4304902: 'RS',  # Porto Alegre
    2927408: 'BA',  # Salvador
    2611606: 'PE',  # Recife
    2304400: 'CE',  # Fortaleza
    5300108: 'DF',  # Brasília
}


# %%
# Just get SP and RJ to save API calls
states = ['SP', 'RJ']
ibge = ['3550308', '3304557'] 
all_holidays = []
api_calls = 0

for year in range(2018, 2021):
    print(f"Year {year}")
    
    # National
    url = f"https://feriadosapi.com/api/v1/feriados/nacionais?ano={year}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    api_calls += 1
    
    if response.status_code == 200:
        data = response.json()
        # Access the 'feriados' key from the response
        if 'feriados' in data and isinstance(data['feriados'], list):
            for h in data['feriados']:
                h['year'] = year
                h['type'] = 'national'
                h['state'] = ''
                h['ibge_code'] = ''
            all_holidays.extend(data['feriados'])
            print(f"  National: {len(data['feriados'])}")
        else:
            print(f"  National: Unexpected response format")
            print(f"  Response keys: {data.keys()}")
    
    # States
    for state in states:
        url = f"https://feriadosapi.com/api/v1/feriados/estado/{state}?ano={year}"
        response = requests.get(url, headers=headers)
        api_calls += 1
        
        if response.status_code == 200:
            data = response.json()
            # Access the 'feriados' key from the response
            if 'feriados' in data and isinstance(data['feriados'], list):
                for h in data['feriados']:
                    h['year'] = year
                    h['type'] = 'state'
                    h['state'] = state
                    h['ibge_code'] = ''
                all_holidays.extend(data['feriados'])
                print(f"  {state}: {len(data['feriados'])}")
            else:
                print(f"  {state}: Unexpected response format")
                print(f"  Response keys: {data.keys()}")

    # Capitals
    for code_ibge in ibge:
        url = f"https://feriadosapi.com/api/v1/feriados/cidade/{code_ibge}?ano={year}"
        response = requests.get(url, headers=headers)
        api_calls += 1

        if response.status_code == 200:
            data = response.json()
            if 'feriados' in data and isinstance(data['feriados'], list):
                # get the state from the IBGE code
                state_map = ibge_state_map.get(code_ibge, 'Unknown')

            # Access the 'feriados' key from the response
            if 'feriados' in data and isinstance(data['feriados'], list):
                for h in data['feriados']:
                    h['year'] = year
                    h['type'] = 'capital'
                    h['state'] = state_map
                    h['ibge_code'] = code_ibge
                all_holidays.extend(data['feriados'])
                print(f"  {code_ibge}: {len(data['feriados'])}")
            else:
                print(f"  {code_ibge}: Unexpected response format")
                print(f"  Response keys: {data.keys()}")
                {'error': 'Invalid IBGE code'}
        else:
            print(f"{code_ibge}: {response.json()}") 


print(response.json())




print(f"\nTotal API calls: {api_calls}")

# Save to CSV
if all_holidays:
    df = pd.DataFrame(all_holidays)
    df.to_csv(export_path, index=False, encoding='utf-8')
    print(f"Saved {len(df)} holidays to {export_path}")
    
    # Show sample
    print("\nSample data:")
    print(df[['data', 'nome', 'tipo', 'year', 'type', 'state']].head())
else:
    print("No data collected")


# %%


