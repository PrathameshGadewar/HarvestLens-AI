import unittest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import Base, engine, SessionLocal
from backend.app.security import hash_password, verify_password, create_access_token, decode_access_token
from backend.app.models import User

client = TestClient(app)


class TestAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    def test_password_hashing(self):
        pwd = "securepassword123"
        hashed = hash_password(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("wrongpassword", hashed))

    def test_jwt_token_flow(self):
        data = {"user_id": 999, "username": "testuser", "role": "farmer"}
        token = create_access_token(data)
        decoded = decode_access_token(token)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.user_id, 999)
        self.assertEqual(decoded.username, "testuser")
        self.assertEqual(decoded.role, "farmer")

    def test_demo_login_endpoint(self):
        res = client.post("/api/auth/demo-login", json={"role": "farmer"})
        self.assertEqual(res.status_code, 200)
        json_data = res.json()
        self.assertIn("access_token", json_data)
        self.assertEqual(json_data["role"], "farmer")

        res_buyer = client.post("/api/auth/demo-login", json={"role": "buyer"})
        self.assertEqual(res_buyer.status_code, 200)
        self.assertEqual(res_buyer.json()["role"], "buyer")

        res_admin = client.post("/api/auth/demo-login", json={"role": "admin"})
        self.assertEqual(res_admin.status_code, 200)
        self.assertEqual(res_admin.json()["role"], "admin")

    def test_user_registration_and_login(self):
        import uuid
        uid = uuid.uuid4().hex[:6]
        user_payload = {
            "email": f"farmer_{uid}@example.com",
            "username": f"farmer_{uid}",
            "password": "mypassword123",
            "full_name": f"Farmer {uid}",
            "role": "farmer",
            "farm_name": "Sunrise Orchards",
            "phone": "+91 98765 00000"
        }
        # Register
        reg_res = client.post("/api/auth/register", json=user_payload)
        self.assertEqual(reg_res.status_code, 200)
        token = reg_res.json()["access_token"]

        # Check me
        me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["username"], user_payload["username"])

        # Login
        login_res = client.post("/api/auth/login", json={
            "username_or_email": user_payload["email"],
            "password": "mypassword123"
        })
        self.assertEqual(login_res.status_code, 200)
        self.assertIn("access_token", login_res.json())

        # Failed login
        fail_res = client.post("/api/auth/login", json={
            "username_or_email": user_payload["email"],
            "password": "wrongpassword"
        })
        self.assertEqual(fail_res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
