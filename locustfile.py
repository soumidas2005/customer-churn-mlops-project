from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def test_api(self):
        self.client.post("/predict", json={
            "tenure": 12,
            "MonthlyCharges": 70,
            "TotalCharges": 1000,
            "support_tickets": 2,
            "usage_ratio": 0.5
        })