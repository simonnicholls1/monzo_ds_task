import pydata_google_auth

class BigQueryConnection:

    def __init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/drive',
        ]

        self.credentials = pydata_google_auth.get_user_credentials(self.SCOPES,auth_local_webserver=False,)

    def get_credentials(self):
        return self.credentials