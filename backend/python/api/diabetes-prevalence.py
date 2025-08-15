import requests
import plotly.express as px

# WHO API URL for diabetes prevalence (age-standardized)
url = 'https://ghoapi.azureedge.net/api/NCD_DIABETES_PREVALENCE_AGESTD'

def render_diabetes_map():
    try:
        # Fetch data from the WHO API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()  # Parse the JSON response

        # Process the data to get the most recent year for each country
        country_data = {}
        for record in data['value']:
            country = record.get('SpatialDim')  # Country code
            year = record.get('TimeDim')       # Year
            prevalence = record.get('NumericValue')  # Diabetes prevalence

            # Only process records with valid data
            if country and year and prevalence is not None:
                # Update with the most recent year's data
                if country not in country_data or year > country_data[country]['year']:
                    country_data[country] = {
                        'year': year,
                        'prevalence': prevalence
                    }

        # Prepare data for choropleth map
        formatted_data = [
            {
                'country_code': country,
                'year': details['year'],
                'prevalence': details['prevalence']
            }
            for country, details in country_data.items()
        ]

        # Create the choropleth map using Plotly
        fig = px.choropleth(
            formatted_data,
            locations='country_code',  # Country codes in ISO-3166-1 format
            color='prevalence',
            hover_name='country_code',
            color_continuous_scale='Viridis',  # Color scale
            labels={'prevalence': 'Diabetes Prevalence (%)'},
            title="Diabetes Prevalence by Country"
        )

        # Display the map interactively
        fig.show()

    except requests.exceptions.RequestException as e:
        print(f'Failed to fetch data from WHO API: {e}')

if __name__ == '__main__':
    render_diabetes_map()
