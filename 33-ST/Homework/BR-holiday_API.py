# %%
import requests
import yaml
import pandas as pd
import os
from datetime import datetime, timedelta

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
# Map IBGE codes to their respective states
ibge_state_map = {
    # North Region (Região Norte)
    # '1200401': 'AC',  # Rio Branco
    # '1600303': 'AP',  # Macapá
    # '1302603': 'AM',  # Manaus
    # '1501402': 'PA',  # Belém
    # '1100205': 'RO',  # Porto Velho
    # '1400100': 'RR',  # Boa Vista
    # '1721000': 'TO',  # Palmas

    # Northeast Region (Região Nordeste)
    # '2704302': 'AL',  # Maceió
    # '2927408': 'BA',  # Salvador
    # '2304400': 'CE',  # Fortaleza
    # '2111300': 'MA',  # São Luís
    # '2507507': 'PB',  # João Pessoa
    # '2611606': 'PE',  # Recife
    # '2211001': 'PI',  # Teresina
    # '2408102': 'RN',  # Natal
    # '2800308': 'SE',  # Aracaju

    # Central-West Region (Região Centro-Oeste)
    '5300108': 'DF',  # Brasília (Federal District)
    # '5208707': 'GO',  # Goiânia
    # '5103403': 'MT',  # Cuiabá
    # '5002704': 'MS',  # Campo Grande

    # Southeast Region (Região Sudeste)
    '3205309': 'ES',  # Vitória
    '3106200': 'MG',  # Belo Horizonte
    '3304557': 'RJ',  # Rio de Janeiro
    '3550308': 'SP',  # São Paulo

    # South Region (Região Sul)
    '4106902': 'PR',  # Curitiba
    '4205407': 'SC',  # Florianópolis
    '4314902': 'RS',  # Porto Alegre
}

# %%
# Main Parameters
ibge_codes = list(ibge_state_map.keys())
years = range(2018,2022)
states = list(ibge_state_map.values())

# %%
# load the credentials from YAML file
def load_api_key():
    """Load API key from credentials file"""
    with open(os.path.join(main_dir, 'credentials.yaml'), 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials['feriados_api']['key']


# %% 
# add carnaval tuesday or monday
def add_carnival_tuesday(df):
    """Check if Carnival is Monday or Tuesday, then add the missing day"""
    carnival_rows = df[df['nome'].str.contains('Carnaval', case=False, na=False)]
    
    if len(carnival_rows) == 0:
        print("Warning: No Carnival found in data")
        return df
    
    # Get unique years
    years_with_carnival = carnival_rows['year'].unique()
    
    for year in years_with_carnival:
        carnival_row = carnival_rows[carnival_rows['year'] == year].iloc[0]
        
        # Get the date
        carnival_date = datetime.strptime(carnival_row['data'], '%d/%m/%Y')
        day_of_week = carnival_date.strftime('%A')  # Monday, Tuesday, etc.
        
        print(f"  Year {year}: Carnival is on {day_of_week}")
        
        # Check if it's Monday or Tuesday
        if day_of_week == 'Monday':
            # It's Monday, add Tuesday
            tuesday_date = carnival_date + timedelta(days=1)
            tuesday_str = tuesday_date.strftime('%d/%m/%Y')
            
            tuesday_exists = df[(df['data'] == tuesday_str) & (df['year'] == year)]
            if len(tuesday_exists) == 0:
                tuesday_row = carnival_row.copy()
                tuesday_row['data'] = tuesday_str
                tuesday_row['nome'] = 'Carnaval (Terça-feira)'
                df.loc[len(df)] = tuesday_row
                print(f"    Added Carnival Tuesday: {tuesday_str}")
        
        elif day_of_week == 'Tuesday':
            # It's Tuesday, add Monday
            monday_date = carnival_date - timedelta(days=1)
            monday_str = monday_date.strftime('%d/%m/%Y')
            
            monday_exists = df[(df['data'] == monday_str) & (df['year'] == year)]
            if len(monday_exists) == 0:
                monday_row = carnival_row.copy()
                monday_row['data'] = monday_str
                monday_row['nome'] = 'Carnaval (Segunda-feira)'
                df.loc[len(df)] = monday_row
                print(f"    Added Carnival Monday: {monday_str}")
        
        else:
            print(f"    Warning: Carnival is on {day_of_week} (expected Monday or Tuesday)")
    
    return df


# %%
def fetch_national_holidays(year, api_key):
    """Fetch national holidays for a given year"""
    url = f"https://feriadosapi.com/api/v1/feriados/nacionais"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "ano": year,
        "facultativos": True  # True includes facultative holidays
    }
    response = requests.get(url, params=params, headers=headers)
    
    holidays = []
    if response.status_code == 200:
        data = response.json()
        if 'feriados' in data and isinstance(data['feriados'], list):
            for h in data['feriados']:
                h['year'] = year
                h['type'] = 'national'
                h['state'] = ''
                h['ibge_code'] = ''
            holidays = data['feriados']
            print(f"  National: {len(holidays)}")
        else:
            print(f"  National: Unexpected response format")
    else:
        print(f"  National: Error {response.status_code}")
    return holidays


# %%
def fetch_state_holidays(states, year, api_key):
    """Fetch state holidays for given states and year"""
    headers = {"Authorization": f"Bearer {api_key}"}
    all_holidays = []
    
    for state in states:
        url = f"https://feriadosapi.com/api/v1/feriados/estado/{state}"
        params = {
            "ano": year,
            "facultativos": True  # True includes facultative holidays
        }
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
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
        else:
            print(f"  {state}: Error {response.status_code}")
    
    return all_holidays


# %%
def fetch_city_holidays(ibge_codes, year, api_key):
    """Fetch city holidays for given IBGE codes and year"""
    headers = {"Authorization": f"Bearer {api_key}"}
    all_holidays = []
    
    for code_ibge in ibge_codes:
        url = f"https://feriadosapi.com/api/v1/feriados/cidade/{code_ibge}"
        params = {
            "ano": year,
            "facultativos": True  # True includes facultative holidays
        }
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'feriados' in data and isinstance(data['feriados'], list):
                state_map = ibge_state_map.get(code_ibge, 'Unknown')
                for h in data['feriados']:
                    h['year'] = year
                    h['type'] = 'capital'
                    h['state'] = state_map
                    h['ibge_code'] = code_ibge
                all_holidays.extend(data['feriados'])
                print(f"  {code_ibge}: {len(data['feriados'])}")
            else:
                print(f"  {code_ibge}: Unexpected response format")
        else:
            print(f"  {code_ibge}: Error {response.status_code}")
    
    return all_holidays


# %%
# Main Functions - choose whcih holidays to get from API
# -----------------------------------------------------------------------------
def get_all_holidays(years, states, ibge_codes, api_key):
    """Get ALL holidays: national + state + city"""
    all_holidays = []
    api_calls = 0
    
    for year in years:
        print(f"\nYear {year}")
        
        # National
        holidays = fetch_national_holidays(year, api_key)
        all_holidays.extend(holidays)
        api_calls += 1
        
        # States
        holidays = fetch_state_holidays(states, year, api_key)
        all_holidays.extend(holidays)
        api_calls += len(states)
        
        # Cities
        holidays = fetch_city_holidays(ibge_codes, year, api_key)
        all_holidays.extend(holidays)
        api_calls += len(ibge_codes)
    
    print(f"\nTotal API calls: {api_calls}")
    return all_holidays


def get_national_only(years, api_key):
    """Get ONLY national holidays"""
    all_holidays = []
    api_calls = 0
    
    for year in years:
        print(f"\nYear {year}")
        holidays = fetch_national_holidays(year, api_key)
        all_holidays.extend(holidays)
        api_calls += 1
    
    print(f"\nTotal API calls: {api_calls}")
    return all_holidays


def get_state_only(states, years, api_key):
    """Get ONLY state holidays for given states"""
    all_holidays = []
    api_calls = 0
    
    for year in years:
        print(f"\nYear {year}")
        holidays = fetch_state_holidays(states, year, api_key)
        all_holidays.extend(holidays)
        api_calls += len(states)
    
    print(f"\nTotal API calls: {api_calls}")
    return all_holidays


def get_city_only(ibge_codes, years, api_key):
    """Get ONLY city holidays for given IBGE codes"""
    all_holidays = []
    api_calls = 0
    
    for year in years:
        print(f"\nYear {year}")
        holidays = fetch_city_holidays(ibge_codes, year, api_key)
        all_holidays.extend(holidays)
        api_calls += len(ibge_codes)
    
    print(f"\nTotal API calls: {api_calls}")
    return all_holidays


def get_national_and_state(years, states, api_key):
    """Get national + state holidays (no cities)"""
    all_holidays = []
    api_calls = 0
    
    for year in years:
        print(f"\nYear {year}")
        
        # National
        holidays = fetch_national_holidays(year, api_key)
        all_holidays.extend(holidays)
        api_calls += 1
        
        # States
        holidays = fetch_state_holidays(states, year, api_key)
        all_holidays.extend(holidays)
        api_calls += len(states)
    
    print(f"\nTotal API calls: {api_calls}")
    return all_holidays


def get_national_and_city(years, ibge_codes, api_key):
    """Get national + city holidays (no states)"""
    all_holidays = []
    api_calls = 0
    
    for year in years:
        print(f"\nYear {year}")
        
        # National
        holidays = fetch_national_holidays(year, api_key)
        all_holidays.extend(holidays)
        api_calls += 1
        
        # Cities
        holidays = fetch_city_holidays(ibge_codes, year, api_key)
        all_holidays.extend(holidays)
        api_calls += len(ibge_codes)
    
    print(f"\nTotal API calls: {api_calls}")
    return all_holidays


def save_holidays(holidays, export_path_override=None):
    """Save holidays to CSV and return DataFrame"""
    save_path = export_path_override if export_path_override is not None else export_path
    
    if not holidays:
        print("No data to save")
        return pd.DataFrame()
    
    df = pd.DataFrame(holidays)
    
    # Add Carnival Tuesday if Carnival exists
    df = add_carnival_tuesday(df)
    
    # Sort by date
    df['date_obj'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df = df.sort_values('date_obj').drop('date_obj', axis=1)
    
    # Save to CSV
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\nSaved {len(df)} holidays to {save_path}")
    
    # Show sample
    print("\nSample data:")
    print(df[['data', 'nome', 'tipo', 'year', 'type', 'state']].head())
    
    return df


# %%
# USAGE EXAMPLES
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Load API key
    API_KEY = load_api_key()
    
    # Define your parameters
    # YEARS, STATES, and IBGE_CODES are already defined above
    
    # Choose ONE of the following:
    
    # Option 1: Get ALL holidays
    # holidays = get_all_holidays(years, states, ibge_codes, API_KEY)
    
    # Option 2: Get ONLY national holidays
    # holidays = get_national_only(years, API_KEY)
    
    # Option 3: Get ONLY state holidays
    # holidays = get_state_only(states, years, API_KEY)
    
    # Option 4: Get ONLY city holidays
    holidays = get_city_only(ibge_codes, years, API_KEY)
    
    # Option 5: Get national + state only
    # holidays = get_national_and_state(years, states, API_KEY)
    
    # Option 6: Get national + city only
    # holidays = get_national_and_city(years, ibge_codes, API_KEY)
    
    # Save the results
    df = save_holidays(holidays)

# %%
