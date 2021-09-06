import requests
import streamlit
import time


prev_sent = ""
streamlit.title("Answer bot")
html_temp="""

"""

streamlit.markdown(html_temp)
sent=streamlit.text_input("Enter your sentence, for example 'are you going to marry me?'")
prediction=""

if streamlit.button("Ask") or (prev_sent != sent):
    data = {'bot': sent}
    res = requests.post("http://localhost:8080/bot", json=data)
    task_id = res.json()['task_id']
    status = "IN_PROGRESS"
    while status != "DONE":
        time.sleep(0.5)
        r = requests.get('http://localhost:8080/bot/{}'.format(task_id))
        status = r.json()['status']

    prediction=r.json()['result']
streamlit.success("Answer : {}".format(prediction))