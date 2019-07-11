import pandas_gbq
import pydata_google_auth


class CSATDAO:

    def __init__(self, credentials):
        self.credentials = credentials

    def get_csat_data(self):

        csat_df = pandas_gbq.read_gbq(
            "select * from monzo_userresearch.csat",
            project_id='analytics-take-home-test',
            credentials=self.credentials,
        )

        return csat_df