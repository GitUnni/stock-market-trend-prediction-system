import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("app/.env")
sender_email = os.getenv("sender_email")
sender_password = os.getenv("sender_password")
print(sender_email)
print(sender_password)

TODAY        = datetime.now()
TODAY_STR    = TODAY.strftime("%B %d, %Y")
print(TODAY_STR)