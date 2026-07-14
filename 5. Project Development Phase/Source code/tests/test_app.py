import unittest

from app import app


class SmartLenderAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Lender", response.data)

    def test_prediction_page_loads(self):
        response = self.client.get("/predict")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Loan Eligibility Form", response.data)

    def test_architecture_page_loads(self):
        response = self.client.get("/architecture")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Machine Learning Pipeline", response.data)

    def test_health_check_reports_model_ready(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertTrue(response.json["model_available"])

    def test_prediction_post_returns_result(self):
        response = self.client.post(
            "/predict",
            data={
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": "6200",
                "CoapplicantIncome": "2200",
                "LoanAmount": "150",
                "Loan_Amount_Term": "360",
                "Credit_History": "1",
                "Property_Area": "Semiurban",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Confidence", response.data)


if __name__ == "__main__":
    unittest.main()
