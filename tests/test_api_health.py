import unittest

try:
    from fastapi.testclient import TestClient
    from apps.api.app.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    TestClient = None
    app = None


@unittest.skipUnless(HAS_FASTAPI, "FastAPI/httpx not installed in current environment")
class TestAPIHealth(unittest.TestCase):
    """Test case verifying health and readiness endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """Test GET /health returns 200 and status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "emergency-vision-api")
        self.assertIn("timestamp", data)

    def test_readiness_check(self):
        """Test GET /health/ready returns 200 and ready status."""
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ready")
        self.assertIn("dependencies", data)


if __name__ == "__main__":
    unittest.main()
