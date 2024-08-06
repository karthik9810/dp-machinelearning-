import streamlit as st
import requests

# Title of the app
st.title('Random Jokes Generator')

# Fetch a random joke from the API
def get_random_joke():
    response = requests.get('https://official-joke-api.appspot.com/random_joke')
    if response.status_code == 200:
        joke = response.json()
        return f"{joke['setup']} - {joke['punchline']}"
    else:
        return "Failed to fetch a joke."

# Display a button to get a new joke
if st.button('Get a Random Joke'):
    joke = get_random_joke()
    st.write(joke)
else:
    st.write('Click the button to get a random joke!')

# Footer
st.write('Powered by the [Official Joke API](https://github.com/15Dkatz/official_joke_api)')

