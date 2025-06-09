from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


# Directory structure scaffold (to create in filesystem):
# GreenRoboAdvisor/
# ├── app/
# │   ├── __init__.py
# │   ├── routes.py
# │   ├── models.py
# │   ├── forms.py
# │   ├── utils.py
# │   ├── robo/
# │   │   ├── Green_Robo_Advisor_Class.py
# │   │   └── Green_Robo_Advisor_main.py
# │   ├── templates/
# │   │   ├── layout.html
# │   │   ├── index.html
# │   │   ├── login.html
# │   │   ├── register.html
# │   │   ├── form.html
# │   │   ├── results.html
# │   │   └── history.html
# │   └── static/
# │       ├── css/
# │       └── js/
# ├── Green_ETF_Selection.xlsx
# ├── config.py
# └── requirements.txt