import pandas_gbq
import pydata_google_auth


class NPSDAO():

    def __init__(self, credentials):
        self.credentials = credentials

    def get_nps_data(self):
        nps_df = pandas_gbq.read_gbq(
            "select * from monzo_userresearch.nps limit 100",
            project_id='analytics-take-home-test',
            credentials=self.credentials,
        )

        return nps_df

