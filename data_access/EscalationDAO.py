import pandas_gbq
import pydata_google_auth


class EscalationDAO():

    def __init__(self, credentials):
        self.credentials = credentials

    def get_escalation_data(self):
        escalations_df = pandas_gbq.read_gbq(
            "select * from monzo_userresearch.escalations limit 100",
            project_id='analytics-take-home-test',
            credentials=self.credentials,
        )

        return escalations_df
