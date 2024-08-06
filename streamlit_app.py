import streamlit as st
import requests

# Title of the app
st.title('Random Quotes Generator')

# Fetch a random quote from the API
def get_random_quote():
    response = requests.get('https://api.quotable.io/random')
    if response.status_code == 200:
        quote = response.json()
        return f"\"{quote['content']}\" - {quote['author']}"
    else:
        return "Failed to fetch a quote."

# Display a button to get a new quote
if st.button('Get a Random Quote'):
    quote = get_random_quote()
    st.write(quote)
else:
    st.write('Click the button to get a random quote!')

# Footer
st.write('Powered by the [Quotable API](https://github.com/lukePeavey/quotable)')
